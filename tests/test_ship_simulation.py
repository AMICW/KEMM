import os
import json
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np

from kemm.core.types import KEMMConfig as RuntimeKEMMConfig
from reporting_config import PublicationStyle, ShipPlotConfig, interactive_bundle_path
from ship_simulation.config import DemoConfig, KEMMConfig, apply_experiment_profile, build_default_config, build_default_demo_config
from ship_simulation.core.collision_risk import RiskBreakdown
from ship_simulation.core.environment import GridScalarField, GridVectorField
from ship_simulation.optimizer.episode import PlanningEpisodeResult, PlanningStepResult, RollingHorizonPlanner
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.kemm_solver import ShipKEMMOptimizer
from ship_simulation.optimizer.selection import select_representative_index
from ship_simulation.scenario.encounter import ChannelBoundary, KeepOutZone
from ship_simulation.scenario.generator import ScenarioGenerator
from ship_simulation import run_report as run_report_module
from ship_simulation.visualization import (
    ExperimentSeries,
    open_figure_bundle,
    save_control_time_series,
    save_change_timeline_panel,
    save_convergence_statistics,
    save_distribution_violin,
    save_dynamic_avoidance_snapshots,
    save_environment_overlay,
    save_parallel_coordinates,
    save_pareto_3d_with_knee,
    save_pareto_projection_panel,
    save_radar_chart,
    save_risk_breakdown_time_series,
    save_route_planning_panel,
    save_route_bundle_gallery,
    save_run_statistics_panel,
    save_safety_envelope_plot,
    save_scenario_gallery,
    save_spatiotemporal_plot,
    save_summary_dashboard,
)
from ship_simulation.visualization.report_plots import _projection_front_order


matplotlib.use("Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


class ShipSimulationSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = build_default_config()
        self.scenario = ScenarioGenerator(self.config).generate("crossing")
        self.interface = ShipOptimizerInterface(self.scenario, self.config)

    def _make_series(self, episode: PlanningEpisodeResult, *, label: str = "Random", color: str = "tab:orange") -> ExperimentSeries:
        stats = {
            "n_runs": 1.0,
            "success_rate": 1.0 if episode.final_evaluation.reached_goal else 0.0,
            "success_rate_std": 0.0,
            "minimum_clearance": float(episode.analysis_metrics.get("minimum_clearance", 0.0)),
            "minimum_clearance_std": 0.0,
            "minimum_ship_distance": float(episode.analysis_metrics.get("minimum_ship_distance", 0.0)),
            "minimum_ship_distance_std": 0.0,
            "runtime": float(episode.analysis_metrics.get("runtime", 0.0)),
            "runtime_std": 0.0,
        }
        return ExperimentSeries(
            label=label,
            color=color,
            episode=episode,
            histories=[episode.convergence_history],
            distribution_metrics=[episode.analysis_metrics],
            repeated_statistics=stats,
        )

    def _stub_episode(self, scenario_key: str, *, optimizer_name: str = "kemm") -> PlanningEpisodeResult:
        scenario = ScenarioGenerator(self.config).generate(scenario_key)
        interface = ShipOptimizerInterface(scenario, self.config)
        decision = interface.build_context().initial_guess
        evaluation = interface.simulate(decision)
        history = [{
            "generation": 0.0,
            "best_fuel": float(evaluation.objectives[0]),
            "best_time": float(evaluation.objectives[1]),
            "best_risk": float(evaluation.objectives[2]),
            "best_weighted_score": float(np.mean(evaluation.objectives)),
        }]
        step = PlanningStepResult(
            step_index=0,
            start_time=0.0,
            optimizer_name=optimizer_name,
            runtime_s=0.0,
            scenario=scenario,
            best_decision=decision.copy(),
            selected_evaluation=evaluation,
            pareto_decisions=np.asarray([decision], dtype=float),
            pareto_objectives=np.asarray([evaluation.objectives], dtype=float),
            history=history,
            applied_changes=[],
        )
        metrics = dict(evaluation.analysis_metrics)
        metrics.update(
            {
                "fuel": float(evaluation.objectives[0]),
                "time": float(evaluation.objectives[1]),
                "risk": float(evaluation.objectives[2]),
                "intrusion_time": float(evaluation.risk.intrusion_time),
                "runtime": 0.0,
                "planning_steps": 1.0,
            }
        )
        return PlanningEpisodeResult(
            scenario_name=scenario.name,
            optimizer_name=optimizer_name,
            steps=[step],
            final_evaluation=evaluation,
            pareto_objectives=np.asarray([evaluation.objectives], dtype=float),
            pareto_decisions=np.asarray([decision], dtype=float),
            knee_index=0,
            knee_objectives=evaluation.objectives.copy(),
            snapshots=[],
            analysis_metrics=metrics,
            convergence_history=history,
            terminated_reason="stub",
            experiment_profile="baseline",
            change_history=[],
            problem_config=self.config,
        )

    def test_optimizer_context_matches_problem_shape(self):
        context = self.interface.build_context()
        self.assertEqual(context.n_obj, 3)
        self.assertEqual(context.n_var, self.config.num_intermediate_waypoints * 3)

        obj_func = self.interface.make_objective_function(context)
        pop = np.vstack([context.initial_guess, context.initial_guess])
        values = obj_func(pop, 0.0)
        self.assertEqual(values.shape, (2, 4))

    def test_small_kemm_run_reaches_goal(self):
        demo = DemoConfig(
            optimizer_name="kemm",
            kemm=KEMMConfig(
                pop_size=16,
                generations=4,
                refresh_interval=2,
                seed=7,
                inject_initial_guess=True,
                initial_guess_copies=3,
                initial_guess_jitter_ratio=0.02,
            ),
        )
        result = ShipKEMMOptimizer(self.interface, demo).optimize()
        self.assertEqual(result.population.shape[1], self.interface.build_context().n_var)
        self.assertEqual(result.fitness.shape[1], 4)
        self.assertTrue(result.best_evaluation.reached_goal)

    def test_ship_kemm_reuses_persistent_algorithm_across_replans(self):
        demo = DemoConfig(
            optimizer_name="kemm",
            kemm=KEMMConfig(
                pop_size=16,
                generations=3,
                seed=9,
                use_change_response=True,
                reuse_solver_state_across_replans=True,
                inject_initial_guess=True,
                initial_guess_copies=3,
                initial_guess_jitter_ratio=0.02,
            ),
        )
        solver = ShipKEMMOptimizer(self.interface, demo)
        first = solver.optimize(change_time=0.0)
        algo_id = id(solver._algo)
        self.assertTrue(first.best_evaluation.reached_goal)

        shifted_scenario = self.scenario.with_updated_states(
            replace(self.scenario.own_ship.initial_state, x=self.scenario.own_ship.initial_state.x + 120.0),
            [replace(target.initial_state) for target in self.scenario.target_ships],
            name_suffix="[shifted]",
        )
        shifted_interface = ShipOptimizerInterface(shifted_scenario, self.config)
        second = solver.optimize(interface=shifted_interface, change_time=180.0)

        self.assertEqual(id(solver._algo), algo_id)
        self.assertGreaterEqual(len(solver._algo.change_diagnostics_history), 1)
        self.assertEqual(solver._algo.last_change_diagnostics.time_step, 1)
        self.assertEqual(second.population.shape[1], shifted_interface.build_context().n_var)

    def test_ship_kemm_resets_algorithm_state_between_replans_by_default(self):
        demo = DemoConfig(
            optimizer_name="kemm",
            kemm=KEMMConfig(
                pop_size=16,
                generations=3,
                seed=9,
                use_change_response=True,
                inject_initial_guess=True,
                initial_guess_copies=3,
                initial_guess_jitter_ratio=0.02,
            ),
        )
        solver = ShipKEMMOptimizer(self.interface, demo)
        solver.optimize(change_time=0.0)
        first_algo_id = id(solver._algo)

        shifted_scenario = self.scenario.with_updated_states(
            replace(self.scenario.own_ship.initial_state, x=self.scenario.own_ship.initial_state.x + 120.0),
            [replace(target.initial_state) for target in self.scenario.target_ships],
            name_suffix="[shifted]",
        )
        shifted_interface = ShipOptimizerInterface(shifted_scenario, self.config)
        solver.optimize(interface=shifted_interface, change_time=180.0)

        self.assertEqual(solver._solve_count, 1)
        self.assertIsNone(solver._algo.last_change_diagnostics)

    def test_simulate_uses_evaluation_cache_for_repeated_decisions(self):
        problem = self.interface.problem
        decision = problem.initial_guess()
        with patch.object(problem.own_ship_model, "simulate_route", wraps=problem.own_ship_model.simulate_route) as own_wrap:
            with patch.object(problem.target_ship_model, "simulate_constant_velocity", wraps=problem.target_ship_model.simulate_constant_velocity) as target_wrap:
                first = problem.simulate(decision)
                second = problem.simulate(decision)
        self.assertEqual(own_wrap.call_count, 1)
        self.assertEqual(target_wrap.call_count, 0)
        np.testing.assert_allclose(first.objectives, second.objectives)

    def test_environment_batch_api_matches_scalar_sampling(self):
        environment = self.interface.problem.environment
        positions = np.array(
            [
                [0.0, -480.0],
                [1200.0, -900.0],
                [2600.0, -150.0],
                [4100.0, 600.0],
                [5600.0, 1200.0],
            ],
            dtype=float,
        )
        times = np.array([0.0, 45.0, 90.0, 135.0, 180.0], dtype=float)

        scalar_expected = np.asarray([environment.scalar_risk_at(position, time_s) for position, time_s in zip(positions, times)], dtype=float)
        vector_expected = np.asarray([environment.vector_field_at(position, time_s) for position, time_s in zip(positions, times)], dtype=float)
        drift_expected = np.asarray([environment.drift_velocity(position, time_s) for position, time_s in zip(positions, times)], dtype=float)

        np.testing.assert_allclose(environment.scalar_risk_series(positions, times), scalar_expected, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(environment.vector_field_series(positions, times), vector_expected, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(environment.drift_series(positions, times), drift_expected, atol=1e-9, rtol=0.0)

    def test_ship_risk_model_batch_path_matches_scalar_reference(self):
        problem = self.interface.problem
        evaluation = problem.simulate(problem.initial_guess())
        own_traj = evaluation.own_trajectory
        targets = evaluation.target_trajectories
        target_names = [target.name for target in self.scenario.target_ships]
        model = problem.risk_model
        batch = model.evaluate(
            own_traj,
            targets,
            static_obstacles=self.scenario.static_obstacles,
            static_obstacle_descriptors=problem._static_obstacle_descriptors,
            environment=problem.environment,
            colreg_roles=self.scenario.metadata.colreg_roles,
            target_names=target_names,
        )

        count = len(own_traj.times)
        dt = own_traj.times[1] - own_traj.times[0] if count > 1 else 0.0
        risk = np.zeros(count, dtype=float)
        clearance_series = np.full(count, np.inf, dtype=float)
        static_clearance_series = np.full(count, np.inf, dtype=float)
        ship_distance_series = np.full(count, np.inf, dtype=float)
        dcpa_series = np.full(count, np.inf, dtype=float)
        tcpa_series = np.full(count, np.inf, dtype=float)

        for idx in range(count):
            obstacle_clearance = model._obstacle_clearance(
                own_traj.positions[idx],
                descriptors=problem._static_obstacle_descriptors,
            )
            static_clearance_series[idx] = obstacle_clearance
            obstacle_risk = model._clearance_risk(obstacle_clearance)
            environment_risk = min(2.0, problem.environment.scalar_risk_at(own_traj.positions[idx], own_traj.times[idx]))

            step_domain = 0.0
            step_scale = 1.0
            step_dcpa_risk = 0.0
            step_ship_distance = float("inf")
            step_dcpa = float("inf")
            step_tcpa = float("inf")
            for target_idx, target in enumerate(targets):
                sample_idx = min(idx, len(target.times) - 1)
                ship_distance = float(np.linalg.norm(target.positions[sample_idx] - own_traj.positions[idx]))
                step_ship_distance = min(step_ship_distance, ship_distance)
                domain_value = model.instantaneous_domain_risk(
                    own_position=own_traj.positions[idx],
                    own_heading=own_traj.headings[idx],
                    target_position=target.positions[sample_idx],
                )
                dcpa, tcpa = model._dcpa_tcpa(own_traj, target, sample_idx)
                dcpa_value = model._dcpa_risk(dcpa, tcpa)
                scale = model._colreg_scale(self.scenario.metadata.colreg_roles.get(target_names[target_idx], ""))
                if domain_value * scale > step_domain * step_scale:
                    step_domain = domain_value
                    step_scale = scale
                step_dcpa_risk = max(step_dcpa_risk, dcpa_value)
                step_dcpa = min(step_dcpa, dcpa)
                step_tcpa = min(step_tcpa, tcpa)

            clearance_series[idx] = min(obstacle_clearance, step_ship_distance)
            ship_distance_series[idx] = step_ship_distance
            dcpa_series[idx] = step_dcpa
            tcpa_series[idx] = step_tcpa
            risk[idx] = (
                self.config.domain_risk_weight * step_domain
                + self.config.dcpa_risk_weight * step_dcpa_risk
                + self.config.obstacle_risk_weight * obstacle_risk
                + self.config.environment_risk_weight * environment_risk
            ) * step_scale

        self.assertAlmostEqual(batch.max_risk, float(np.max(risk)), places=9)
        self.assertAlmostEqual(batch.mean_risk, float(np.mean(risk)), places=9)
        self.assertAlmostEqual(batch.intrusion_time, float(np.sum(risk >= 1.0) * dt), places=9)
        self.assertAlmostEqual(batch.min_clearance, float(np.min(clearance_series)), places=9)
        self.assertAlmostEqual(batch.min_dcpa, float(np.min(dcpa_series[np.isfinite(dcpa_series)])), places=9)
        self.assertAlmostEqual(batch.min_tcpa, float(np.min(tcpa_series[np.isfinite(tcpa_series)])), places=9)

    def test_local_scoring_penalizes_clearance_shortfall_in_objectives(self):
        problem = self.interface.problem
        base = problem.simulate(problem.initial_guess())
        sample_count = len(base.own_trajectory.times)

        def make_risk(min_clearance: float) -> RiskBreakdown:
            risk_series = np.full(sample_count, 0.08, dtype=float)
            clearance_series = np.full(sample_count, min_clearance, dtype=float)
            ship_distance_series = np.full(sample_count, max(min_clearance, 220.0), dtype=float)
            dcpa_series = np.full(sample_count, 150.0, dtype=float)
            tcpa_series = np.full(sample_count, 45.0, dtype=float)
            zeros = np.zeros(sample_count, dtype=float)
            return RiskBreakdown(
                max_risk=0.12,
                mean_risk=0.08,
                intrusion_time=0.0,
                min_clearance=min_clearance,
                min_dcpa=150.0,
                min_tcpa=45.0,
                min_static_clearance=min_clearance,
                min_ship_distance=float(ship_distance_series.min()),
                risk_series=risk_series,
                domain_risk_series=risk_series.copy(),
                dcpa_risk_series=zeros.copy(),
                obstacle_risk_series=zeros.copy(),
                environment_risk_series=zeros.copy(),
                clearance_series=clearance_series,
                static_clearance_series=clearance_series.copy(),
                ship_distance_series=ship_distance_series,
                dcpa_series=dcpa_series,
                tcpa_series=tcpa_series,
                colreg_scale_series=np.ones(sample_count, dtype=float),
            )

        with patch.object(problem.risk_model, "evaluate", return_value=make_risk(240.0)):
            safe = problem.score_trajectory_bundle(base.own_trajectory, base.target_trajectories)
        with patch.object(problem.risk_model, "evaluate", return_value=make_risk(40.0)):
            unsafe = problem.score_trajectory_bundle(base.own_trajectory, base.target_trajectories)

        self.assertGreater(unsafe.objectives[0], safe.objectives[0])
        self.assertGreater(unsafe.objectives[1], safe.objectives[1])
        self.assertGreater(unsafe.objectives[2], safe.objectives[2])

    def test_terminal_progress_penalty_does_not_pollute_risk_or_cv(self):
        problem = self.interface.problem
        sample_count = len(problem.simulate(problem.initial_guess()).own_trajectory.times)
        zeros = np.zeros(sample_count, dtype=float)
        safe_clearance = np.full(sample_count, 260.0, dtype=float)
        ship_distance = np.full(sample_count, 320.0, dtype=float)
        dcpa_series = np.full(sample_count, 180.0, dtype=float)
        tcpa_series = np.full(sample_count, 60.0, dtype=float)
        risk = RiskBreakdown(
            max_risk=0.12,
            mean_risk=0.08,
            intrusion_time=0.0,
            min_clearance=260.0,
            min_dcpa=180.0,
            min_tcpa=60.0,
            min_static_clearance=260.0,
            min_ship_distance=320.0,
            risk_series=np.full(sample_count, 0.08, dtype=float),
            domain_risk_series=np.full(sample_count, 0.08, dtype=float),
            dcpa_risk_series=zeros.copy(),
            obstacle_risk_series=zeros.copy(),
            environment_risk_series=zeros.copy(),
            clearance_series=safe_clearance,
            static_clearance_series=safe_clearance.copy(),
            ship_distance_series=ship_distance,
            dcpa_series=dcpa_series,
            tcpa_series=tcpa_series,
            colreg_scale_series=np.ones(sample_count, dtype=float),
        )
        near_obj, near_penalties = problem.compose_objectives(
            fuel=100.0,
            total_time=120.0,
            risk=risk,
            terminal_distance=20.0,
        )
        far_obj, far_penalties = problem.compose_objectives(
            fuel=100.0,
            total_time=120.0,
            risk=risk,
            terminal_distance=2000.0,
        )
        self.assertGreater(far_obj[0], near_obj[0])
        self.assertGreater(far_obj[1], near_obj[1])
        self.assertAlmostEqual(float(far_obj[2]), float(near_obj[2]))
        self.assertAlmostEqual(float(far_penalties["cv"]), float(near_penalties["cv"]))

    def test_heuristic_seed_vectors_include_distinct_safe_detours(self):
        seeds = self.interface.problem.heuristic_seed_vectors(8)
        self.assertGreaterEqual(len(seeds), 4)
        rounded = np.unique(np.round(seeds, decimals=3), axis=0)
        self.assertGreaterEqual(len(rounded), 4)
        base = self.interface.problem.initial_guess()
        self.assertTrue(any(np.linalg.norm(seed - base) > 1e-6 for seed in seeds[1:]))

    def test_ship_kemm_runtime_config_is_forwarded_to_algorithm(self):
        demo = DemoConfig(
            optimizer_name="kemm",
            kemm=KEMMConfig(
                pop_size=14,
                generations=2,
                seed=5,
                runtime=replace(
                    RuntimeKEMMConfig(),
                    benchmark_aware_prior=False,
                    enable_memory=False,
                    enable_prediction=False,
                    enable_transfer=False,
                    enable_adaptive=False,
                ),
            ),
        )
        solver = ShipKEMMOptimizer(self.interface, demo)
        solver.optimize(change_time=0.0)
        self.assertIsNotNone(solver._algo)
        self.assertFalse(solver._algo.config.benchmark_aware_prior)
        self.assertFalse(solver._algo.config.enable_memory)
        self.assertFalse(solver._algo.config.enable_prediction)
        self.assertFalse(solver._algo.config.enable_transfer)
        self.assertFalse(solver._algo.config.enable_adaptive)

    def test_rolling_episode_records_multiple_planning_steps(self):
        demo = DemoConfig(random_search_samples=12)
        demo.episode.local_horizon = 180.0
        demo.episode.execution_horizon = 60.0
        demo.episode.max_replans = 3
        episode = RollingHorizonPlanner(self.scenario, self.config, demo).run("random")
        self.assertGreaterEqual(len(episode.steps), 2)
        self.assertEqual(episode.final_evaluation.objectives.shape, (3,))
        self.assertIn("minimum_clearance", episode.analysis_metrics)
        self.assertIn("minimum_dcpa", episode.analysis_metrics)
        self.assertIn("minimum_tcpa", episode.analysis_metrics)
        self.assertIn("max_commanded_yaw_rate", episode.analysis_metrics)
        self.assertTrue(np.all(np.diff(episode.final_evaluation.own_trajectory.times) > 0.0))

    def test_scenario_supports_explicit_obstacle_and_grid_layers(self):
        head_on = ScenarioGenerator(self.config).generate("head_on")
        self.assertTrue(any(isinstance(obstacle, ChannelBoundary) for obstacle in head_on.static_obstacles))

        crossing = ScenarioGenerator(self.config).generate("crossing")
        self.assertTrue(any(isinstance(obstacle, KeepOutZone) for obstacle in crossing.static_obstacles))
        self.assertTrue(any(isinstance(layer, GridScalarField) for layer in crossing.scalar_fields))
        self.assertTrue(any(isinstance(layer, GridVectorField) for layer in crossing.vector_fields))

        harbor = ScenarioGenerator(self.config).generate("harbor_clutter")
        self.assertGreaterEqual(len(harbor.static_obstacles), 11)
        self.assertGreaterEqual(len(harbor.target_ships), 2)
        self.assertGreaterEqual(sum(isinstance(obstacle, KeepOutZone) for obstacle in harbor.static_obstacles), 3)
        self.assertGreaterEqual(sum(isinstance(obstacle, ChannelBoundary) for obstacle in harbor.static_obstacles), 2)
        self.assertGreaterEqual(sum(isinstance(obstacle, GridScalarField) for obstacle in harbor.scalar_fields), 2)
        self.assertGreaterEqual(sum(isinstance(obstacle, GridVectorField) for obstacle in harbor.vector_fields), 1)

    def test_scenario_generation_config_controls_harbor_clutter_density(self):
        tuned_config = build_default_config()
        tuned = tuned_config.scenario_generation.harbor_clutter
        tuned.circular_obstacle_limit = 5
        tuned.polygon_obstacle_limit = 2
        tuned.target_limit = 2
        tuned.circular_radius_scale = 0.5
        default_harbor = ScenarioGenerator(self.config).generate("harbor_clutter")
        tuned_harbor = ScenarioGenerator(tuned_config).generate("harbor_clutter")

        default_circles = [obstacle for obstacle in default_harbor.static_obstacles if hasattr(obstacle, "radius")]
        tuned_circles = [obstacle for obstacle in tuned_harbor.static_obstacles if hasattr(obstacle, "radius")]
        self.assertEqual(len(tuned_harbor.target_ships), 2)
        self.assertEqual(len(tuned_circles), 5)
        self.assertEqual(sum(isinstance(obstacle, KeepOutZone) for obstacle in tuned_harbor.static_obstacles), 2)
        self.assertLess(np.mean([obstacle.radius for obstacle in tuned_circles]), np.mean([obstacle.radius for obstacle in default_circles]))

    def test_seeded_scenario_family_is_reproducible_and_configurable(self):
        tuned_config = build_default_config()
        crossing_cfg = tuned_config.scenario_generation.crossing
        crossing_cfg.family_name = "crossing_family_a"
        crossing_cfg.scenario_seed = 17
        crossing_cfg.geometry_jitter_m = 120.0
        crossing_cfg.traffic_heading_jitter_deg = 6.0
        crossing_cfg.current_direction_jitter_deg = 8.0
        crossing_cfg.difficulty_scale = 1.25

        scenario_a = ScenarioGenerator(tuned_config).generate("crossing")
        scenario_b = ScenarioGenerator(tuned_config).generate("crossing")

        other_config = build_default_config()
        other_crossing = other_config.scenario_generation.crossing
        other_crossing.family_name = "crossing_family_b"
        other_crossing.scenario_seed = 23
        other_crossing.geometry_jitter_m = 120.0
        other_crossing.traffic_heading_jitter_deg = 6.0
        other_crossing.current_direction_jitter_deg = 8.0
        other_crossing.difficulty_scale = 1.25
        scenario_c = ScenarioGenerator(other_config).generate("crossing")

        np.testing.assert_allclose(scenario_a.own_ship.initial_state.position(), scenario_b.own_ship.initial_state.position())
        self.assertAlmostEqual(scenario_a.target_ships[0].initial_state.heading, scenario_b.target_ships[0].initial_state.heading)
        self.assertFalse(np.allclose(scenario_a.own_ship.initial_state.position(), scenario_c.own_ship.initial_state.position()))
        self.assertEqual(scenario_a.metadata.family, "crossing_family_a")
        self.assertIsNotNone(scenario_a.metadata.layout_seed)

    def test_experiment_profile_applies_dynamic_changes_to_steps(self):
        tuned_config = build_default_config()
        apply_experiment_profile(tuned_config, "shock")
        scenario = ScenarioGenerator(tuned_config).generate("harbor_clutter")
        demo = DemoConfig(random_search_samples=8)
        demo.episode.local_horizon = 180.0
        demo.episode.execution_horizon = 60.0
        demo.episode.max_replans = 3
        episode = RollingHorizonPlanner(scenario, tuned_config, demo).run("random")
        self.assertEqual(episode.experiment_profile, "shock")
        self.assertGreaterEqual(len(episode.change_history or []), 1)
        self.assertGreaterEqual(episode.analysis_metrics.get("scheduled_change_count", 0.0), 1.0)
        changed_steps = [step for step in episode.steps if step.applied_changes]
        self.assertTrue(changed_steps)
        self.assertTrue(any("Sudden channel closure" in str(change.get("label")) for change in changed_steps[0].applied_changes))
        self.assertGreater(len(changed_steps[0].scenario.static_obstacles), len(scenario.static_obstacles))

    def test_kemm_episode_avoids_static_obstacles_and_records_knee(self):
        demo = DemoConfig(
            optimizer_name="kemm",
            random_search_samples=12,
            kemm=KEMMConfig(
                pop_size=20,
                generations=6,
                refresh_interval=3,
                seed=11,
                inject_initial_guess=True,
                initial_guess_copies=4,
                initial_guess_jitter_ratio=0.02,
            ),
        )
        demo.episode.local_horizon = 240.0
        demo.episode.execution_horizon = 80.0
        demo.episode.max_replans = 6
        episode = RollingHorizonPlanner(self.scenario, self.config, demo).run("kemm")
        self.assertGreaterEqual(len(episode.steps), 2)
        self.assertGreaterEqual(episode.analysis_metrics["minimum_clearance"], 0.0)
        self.assertIn("minimum_ship_distance", episode.analysis_metrics)
        np.testing.assert_allclose(episode.pareto_objectives, episode.steps[-1].pareto_objectives)
        self.assertIsNotNone(episode.knee_index)
        self.assertIsNotNone(episode.knee_objectives)

    def test_representative_selection_prefers_safe_candidate(self):
        evaluations = [
            SimpleNamespace(
                reached_goal=False,
                terminal_distance=120.0,
                risk=SimpleNamespace(max_risk=0.35),
                analysis_metrics={"minimum_clearance": -12.0},
            ),
            SimpleNamespace(
                reached_goal=False,
                terminal_distance=240.0,
                risk=SimpleNamespace(max_risk=0.22),
                analysis_metrics={"minimum_clearance": 260.0},
            ),
        ]
        objectives = np.array(
            [
                [10.0, 10.0, 10.0],
                [12.0, 12.0, 12.0],
            ],
            dtype=float,
        )
        chosen = select_representative_index(objectives, evaluations, (1.0, 1.0, 1.0), safety_clearance=180.0)
        self.assertEqual(chosen, 1)

    def test_report_dashboard_is_generated(self):
        decision = self.interface.build_context().initial_guess
        evaluation = self.interface.simulate(decision)
        series = [ExperimentSeries(label="Initial", result=evaluation, color="tab:blue")]
        tmp_dir = Path("ship_simulation/outputs/test_artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            output = tmp_dir / "dashboard.png"
            plot_config = ShipPlotConfig(
                style=PublicationStyle(dpi=180, title_size=12, label_size=10),
                dashboard_figsize=(12.0, 8.0),
            )
            save_summary_dashboard(output, self.scenario, series, plot_config=plot_config)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                try:
                    output.unlink()
                except PermissionError:
                    pass

    def test_core_ship_figure_pack_is_generated(self):
        demo = DemoConfig(random_search_samples=10)
        demo.episode.local_horizon = 180.0
        demo.episode.execution_horizon = 60.0
        demo.episode.max_replans = 3
        episode = RollingHorizonPlanner(self.scenario, self.config, demo).run("random")
        series = [self._make_series(episode)]
        histories = {"Random": [episode.convergence_history]}
        distributions = {"Random": [episode.analysis_metrics]}
        tmp_dir = Path("ship_simulation/outputs/test_artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            tmp_dir / "environment_overlay.png",
            tmp_dir / "route_planning_panel.png",
            tmp_dir / "change_timeline.png",
            tmp_dir / "scenario_gallery.png",
            tmp_dir / "route_bundle_gallery.png",
            tmp_dir / "snapshots.png",
            tmp_dir / "spatiotemporal.png",
            tmp_dir / "control_timeseries.png",
            tmp_dir / "pareto3d.png",
            tmp_dir / "pareto_projection.png",
            tmp_dir / "risk_breakdown.png",
            tmp_dir / "safety_envelope.png",
            tmp_dir / "parallel.png",
            tmp_dir / "radar.png",
            tmp_dir / "convergence.png",
            tmp_dir / "distribution.png",
            tmp_dir / "run_statistics.png",
            tmp_dir / "dashboard_pack.png",
        ]
        plot_config = ShipPlotConfig(style=PublicationStyle(dpi=140, title_size=11, label_size=9))
        try:
            save_environment_overlay(outputs[0], self.scenario, series, plot_config=plot_config)
            save_route_planning_panel(outputs[1], self.scenario, series, plot_config=plot_config)
            save_change_timeline_panel(outputs[2], self.scenario.name, episode, plot_config=plot_config)
            save_scenario_gallery(outputs[3], {"crossing": self.scenario}, plot_config=plot_config)
            save_route_bundle_gallery(outputs[4], {"crossing": self.scenario}, {"crossing": {"kemm": [episode], "nsga_style": [episode]}}, plot_config=plot_config)
            save_dynamic_avoidance_snapshots(outputs[5], self.scenario, episode, plot_config=plot_config)
            save_spatiotemporal_plot(outputs[6], self.scenario, episode, plot_config=plot_config)
            save_control_time_series(outputs[7], self.scenario.name, episode, plot_config=plot_config)
            save_pareto_3d_with_knee(outputs[8], self.scenario.name, episode, plot_config=plot_config)
            save_pareto_projection_panel(outputs[9], self.scenario.name, episode, plot_config=plot_config)
            save_risk_breakdown_time_series(outputs[10], self.scenario.name, episode, plot_config=plot_config)
            save_safety_envelope_plot(outputs[11], self.scenario.name, episode, plot_config=plot_config)
            save_parallel_coordinates(outputs[12], self.scenario.name, series, plot_config=plot_config)
            save_radar_chart(outputs[13], self.scenario.name, series, plot_config=plot_config)
            save_convergence_statistics(outputs[14], self.scenario.name, histories, plot_config=plot_config)
            save_distribution_violin(outputs[15], self.scenario.name, distributions, plot_config=plot_config)
            save_run_statistics_panel(outputs[16], self.scenario.name, series, plot_config=plot_config)
            save_summary_dashboard(outputs[17], self.scenario, series, histories_by_label=histories, metrics_by_label=distributions, plot_config=plot_config)
            for output in outputs:
                self.assertTrue(output.exists())
                self.assertGreater(output.stat().st_size, 0)
        finally:
            for output in outputs:
                if output.exists():
                    try:
                        output.unlink()
                    except PermissionError:
                        pass

    def test_ship_plot_falls_back_without_scienceplots(self):
        decision = self.interface.build_context().initial_guess
        evaluation = self.interface.simulate(decision)
        series = [ExperimentSeries(label="Initial", result=evaluation, color="tab:blue")]
        output = Path("ship_simulation/outputs/test_artifacts/scienceplots_fallback.png")
        plot_config = ShipPlotConfig(
            style=PublicationStyle(
                dpi=140,
                title_size=11,
                label_size=9,
                use_scienceplots=True,
                science_styles=("science", "no-latex"),
            )
        )
        try:
            with patch("reporting_config.HAS_SCIENCEPLOTS", False):
                save_environment_overlay(output, self.scenario, series, plot_config=plot_config)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
        finally:
            if output.exists():
                try:
                    output.unlink()
                except PermissionError:
                    pass

    def test_interactive_ship_artifacts_are_generated_and_reopenable(self):
        demo = DemoConfig(random_search_samples=10)
        demo.episode.local_horizon = 180.0
        demo.episode.execution_horizon = 60.0
        demo.episode.max_replans = 3
        episode = RollingHorizonPlanner(self.scenario, self.config, demo).run("random")
        tmp_dir = Path("ship_simulation/outputs/test_artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pareto_output = tmp_dir / "interactive_pareto3d.png"
        spatiotemporal_output = tmp_dir / "interactive_spatiotemporal.png"
        route_output = tmp_dir / "interactive_route_panel.png"
        view_output = tmp_dir / "interactive_pareto3d_view.png"
        plot_config = ShipPlotConfig(
            style=PublicationStyle(dpi=140, title_size=11, label_size=9),
            interactive_figures=True,
            interactive_html=True,
        )
        bundle_path = interactive_bundle_path(pareto_output)
        html_path = pareto_output.with_suffix(".html")
        spatiotemporal_html = spatiotemporal_output.with_suffix(".html")
        route_bundle = interactive_bundle_path(route_output)
        route_html = route_output.with_suffix(".html")
        try:
            save_pareto_3d_with_knee(pareto_output, self.scenario.name, episode, plot_config=plot_config)
            save_spatiotemporal_plot(spatiotemporal_output, self.scenario, episode, plot_config=plot_config)
            save_route_planning_panel(route_output, self.scenario, [self._make_series(episode)], plot_config=plot_config)
            self.assertTrue(pareto_output.exists())
            self.assertTrue(bundle_path.exists())
            self.assertTrue(html_path.exists())
            self.assertTrue(spatiotemporal_output.exists())
            self.assertTrue(spatiotemporal_html.exists())
            self.assertIn("Mean speed", html_path.read_text(encoding="utf-8"))
            self.assertIn("Own ship", spatiotemporal_html.read_text(encoding="utf-8"))
            self.assertTrue(route_output.exists())
            self.assertFalse(route_bundle.exists())
            self.assertFalse(route_html.exists())
            open_figure_bundle(bundle_path, elev=24.0, azim=132.0, save_path=view_output, show=False)
            self.assertTrue(view_output.exists())
            self.assertGreater(view_output.stat().st_size, 0)
        finally:
            for output in [pareto_output, bundle_path, html_path, spatiotemporal_output, spatiotemporal_html, route_output, route_bundle, route_html, view_output]:
                if output.exists():
                    try:
                        output.unlink()
                    except PermissionError:
                        pass

    def test_pareto_projection_helper_orders_front_and_handles_empty_input(self):
        points = np.array(
            [
                [3.0, 6.0],
                [2.0, 7.0],
                [1.5, 4.0],
                [4.0, 3.0],
                [2.5, 5.5],
            ],
            dtype=float,
        )
        order = _projection_front_order(points)
        self.assertGreater(order.size, 0)
        self.assertTrue(np.all(np.diff(points[order, 0]) >= 0.0))
        self.assertEqual(_projection_front_order(np.zeros((0, 2), dtype=float)).size, 0)

    def test_default_report_scenarios_include_harbor_clutter(self):
        recorded: list[str] = []
        noop_names = [
            "save_environment_overlay",
            "save_route_planning_panel",
            "save_change_timeline_panel",
            "save_scenario_gallery",
            "save_route_bundle_gallery",
            "save_dynamic_avoidance_snapshots",
            "save_spatiotemporal_plot",
            "save_control_time_series",
            "save_pareto_3d_with_knee",
            "save_pareto_projection_panel",
            "save_risk_breakdown_time_series",
            "save_safety_envelope_plot",
            "save_parallel_coordinates",
            "save_radar_chart",
            "save_convergence_statistics",
            "save_distribution_violin",
            "save_run_statistics_panel",
            "save_summary_dashboard",
            "_write_csv",
            "_write_json",
            "_write_markdown",
            "_write_figure_inventory",
        ]

        class DummyGenerator:
            def __init__(self, config):
                self.config = config

            def generate(self, key: str):
                recorded.append(key)
                return ScenarioGenerator(self.config).generate(key)

        class DummyPlanner:
            def __init__(self, scenario, config, demo_config):
                self.scenario = scenario

            def run(self, optimizer_name="kemm"):
                scenario_key = self.scenario.name.lower().replace(" ", "_").replace("-", "_")
                return self_test._stub_episode(scenario_key, optimizer_name=optimizer_name)

        demo = DemoConfig()
        demo.n_runs = 1
        demo.episode_cache_enabled = False
        self_test = self
        tmp_root = Path("ship_simulation/outputs/test_artifacts/default_report_stub")
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(run_report_module, "ScenarioGenerator", DummyGenerator))
                stack.enter_context(patch.object(run_report_module, "RollingHorizonPlanner", DummyPlanner))
                for name in noop_names:
                    stack.enter_context(patch.object(run_report_module, name, lambda *args, **kwargs: None))
                run_report_module.generate_report_with_config(
                    config=self.config,
                    demo_config=demo,
                    output_root=tmp_root,
                    plot_config=ShipPlotConfig(style=PublicationStyle(dpi=120)),
                    scenario_keys=None,
                    verbose=False,
                    n_runs=1,
                )
        finally:
            for path in sorted(tmp_root.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            if tmp_root.exists():
                try:
                    tmp_root.rmdir()
                except OSError:
                    pass
        self.assertIn("harbor_clutter", recorded)

    def test_default_demo_config_uses_full_tuned_profile(self):
        demo = build_default_demo_config()
        self.assertEqual(demo.scenario_profiles.active_profile_name, "full_tuned")

    def test_report_episode_cache_reuses_cached_episodes(self):
        planner_calls: list[str] = []

        class DummyPlanner:
            def __init__(self, scenario, config, demo_config):
                self.scenario = scenario

            def run(self, optimizer_name="kemm"):
                planner_calls.append(optimizer_name)
                scenario_key = self.scenario.name.lower().replace(" ", "_").replace("-", "_")
                return self_test._stub_episode(scenario_key, optimizer_name=optimizer_name)

        demo = build_default_demo_config()
        demo.n_runs = 1
        demo.report_algorithms = ("random",)
        demo.render_workers = 1
        self_test = self
        tmp_root = Path("ship_simulation/outputs/test_artifacts") / f"episode_cache_stub_{int(np.random.randint(1_000_000))}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            with patch.object(run_report_module, "RollingHorizonPlanner", DummyPlanner):
                run_report_module.generate_report_with_config(
                    config=self.config,
                    demo_config=demo,
                    output_root=tmp_root,
                    plot_config=ShipPlotConfig(style=PublicationStyle(dpi=120)),
                    scenario_keys=["crossing"],
                    algorithm_keys=["random"],
                    verbose=False,
                    n_runs=1,
                    render_figures=False,
                )
                first_metadata = json.loads((tmp_root / "raw" / "report_metadata.json").read_text(encoding="utf-8"))
                run_report_module.generate_report_with_config(
                    config=self.config,
                    demo_config=demo,
                    output_root=tmp_root,
                    plot_config=ShipPlotConfig(style=PublicationStyle(dpi=120)),
                    scenario_keys=["crossing"],
                    algorithm_keys=["random"],
                    verbose=False,
                    n_runs=1,
                    render_figures=False,
                )
                second_metadata = json.loads((tmp_root / "raw" / "report_metadata.json").read_text(encoding="utf-8"))
                changed_demo = replace(demo, random_search_seed=demo.random_search_seed + 7)
                run_report_module.generate_report_with_config(
                    config=self.config,
                    demo_config=changed_demo,
                    output_root=tmp_root,
                    plot_config=ShipPlotConfig(style=PublicationStyle(dpi=120)),
                    scenario_keys=["crossing"],
                    algorithm_keys=["random"],
                    verbose=False,
                    n_runs=1,
                    render_figures=False,
                )
                third_metadata = json.loads((tmp_root / "raw" / "report_metadata.json").read_text(encoding="utf-8"))
        finally:
            for path in sorted(tmp_root.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            if tmp_root.exists():
                try:
                    tmp_root.rmdir()
                except OSError:
                    pass

        self.assertEqual(planner_calls, ["random", "random"])
        self.assertEqual(first_metadata["episode_cache_hits"], 0)
        self.assertEqual(first_metadata["episode_cache_misses"], 1)
        self.assertEqual(second_metadata["episode_cache_hits"], 1)
        self.assertEqual(second_metadata["episode_cache_misses"], 0)
        self.assertEqual(third_metadata["episode_cache_hits"], 0)
        self.assertEqual(third_metadata["episode_cache_misses"], 1)

    def test_report_algorithm_registry_is_configurable(self):
        recorded_algorithms: list[str] = []
        noop_names = [
            "save_environment_overlay",
            "save_route_planning_panel",
            "save_change_timeline_panel",
            "save_scenario_gallery",
            "save_route_bundle_gallery",
            "save_dynamic_avoidance_snapshots",
            "save_spatiotemporal_plot",
            "save_control_time_series",
            "save_pareto_3d_with_knee",
            "save_pareto_projection_panel",
            "save_risk_breakdown_time_series",
            "save_safety_envelope_plot",
            "save_parallel_coordinates",
            "save_radar_chart",
            "save_convergence_statistics",
            "save_distribution_violin",
            "save_run_statistics_panel",
            "save_summary_dashboard",
            "_write_csv",
            "_write_markdown",
            "_write_figure_inventory",
        ]

        class DummyPlanner:
            def __init__(self, scenario, config, demo_config):
                self.scenario = scenario

            def run(self, optimizer_name="kemm"):
                recorded_algorithms.append(optimizer_name)
                scenario_key = self.scenario.name.lower().replace(" ", "_").replace("-", "_")
                return self_test._stub_episode(scenario_key, optimizer_name=optimizer_name)

        demo = DemoConfig()
        demo.n_runs = 1
        demo.report_algorithms = ("random", "kemm")
        demo.episode_cache_enabled = False
        self_test = self
        tmp_root = Path("ship_simulation/outputs/test_artifacts/algorithm_registry_stub")
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(run_report_module, "RollingHorizonPlanner", DummyPlanner))
                for name in noop_names:
                    stack.enter_context(patch.object(run_report_module, name, lambda *args, **kwargs: None))
                run_report_module.generate_report_with_config(
                    config=self.config,
                    demo_config=demo,
                    output_root=tmp_root,
                    plot_config=ShipPlotConfig(style=PublicationStyle(dpi=120)),
                    scenario_keys=["crossing"],
                    verbose=False,
                    n_runs=1,
                )
                metadata = (tmp_root / "raw" / "report_metadata.json").read_text(encoding="utf-8")
                representatives = (tmp_root / "raw" / "representative_runs.json").read_text(encoding="utf-8")
        finally:
            for path in sorted(tmp_root.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            if tmp_root.exists():
                try:
                    tmp_root.rmdir()
                except OSError:
                    pass
        self.assertEqual(recorded_algorithms, ["random", "kemm"])
        self.assertIn('"key": "random"', metadata)
        self.assertIn('"key": "kemm"', metadata)
        self.assertIn('"random"', representatives)
        self.assertIn('"kemm"', representatives)

    def test_figure_manifest_and_inventory_follow_registered_specs(self):
        tmp_root = Path("ship_simulation/outputs/test_artifacts/manifest_stub")
        tmp_root.mkdir(parents=True, exist_ok=True)
        manifest_path = tmp_root / "figure_manifest.json"
        inventory_path = tmp_root / "figure_inventory.md"
        try:
            manifest = run_report_module._figure_manifest(["crossing", "harbor_clutter"])
            run_report_module._write_json(manifest_path, manifest)
            run_report_module._write_figure_inventory(inventory_path, ["crossing", "harbor_clutter"])
            manifest_text = manifest_path.read_text(encoding="utf-8")
            inventory_text = inventory_path.read_text(encoding="utf-8")
            self.assertIn("change_timeline", manifest_text)
            self.assertIn("route_bundle_gallery.png", manifest_text)
            self.assertIn("crossing_change_timeline.png", inventory_text)
            self.assertIn("harbor_clutter_dashboard.png", inventory_text)
        finally:
            for output in [manifest_path, inventory_path]:
                if output.exists():
                    try:
                        output.unlink()
                    except PermissionError:
                        pass
            if tmp_root.exists():
                try:
                    tmp_root.rmdir()
                except OSError:
                    pass

    def test_population_evaluation_cache_deduplicates_identical_candidates(self):
        problem = self.interface.problem
        population = np.vstack([problem.initial_guess(), problem.initial_guess(), problem.initial_guess()])
        with patch.object(problem, "simulate", wraps=problem.simulate) as wrapped:
            values = problem.evaluate_population(population)
        self.assertEqual(values.shape, (3, 4))
        self.assertEqual(wrapped.call_count, 1)

    def test_aggregate_rows_include_confidence_interval_fields(self):
        first = self._stub_episode("crossing", optimizer_name="kemm")
        second = self._stub_episode("crossing", optimizer_name="kemm")
        second.final_evaluation.objectives = second.final_evaluation.objectives + np.array([0.5, 0.8, 0.01], dtype=float)
        first.analysis_metrics["runtime"] = 1.2
        second.analysis_metrics["runtime"] = 1.6
        rows = [
            run_report_module._episode_row("crossing", "kemm", 0, first),
            run_report_module._episode_row("crossing", "kemm", 1, second),
        ]
        aggregates = run_report_module._aggregate_rows(rows)
        self.assertEqual(len(aggregates), 1)
        payload = aggregates[0]
        self.assertIn("fuel_ci_low", payload)
        self.assertIn("fuel_ci_high", payload)
        self.assertIn("runtime_ci_low", payload)
        self.assertIn("runtime_ci_high", payload)
        self.assertIn("success_rate_ci_low", payload)
        self.assertIn("success_rate_ci_high", payload)

    def test_statistical_tests_compare_kemm_against_baseline(self):
        kemm_episode = self._stub_episode("crossing", optimizer_name="kemm")
        random_episode = self._stub_episode("crossing", optimizer_name="random")
        random_episode.final_evaluation.objectives = random_episode.final_evaluation.objectives + np.array([2.0, 2.0, 0.2], dtype=float)
        rows = [
            run_report_module._episode_row("crossing", "kemm", 0, kemm_episode),
            run_report_module._episode_row("crossing", "random", 0, random_episode),
        ]
        tests = run_report_module._build_statistical_tests(rows)
        self.assertTrue(any(item["comparison_optimizer"] == "Random" for item in tests))
        self.assertTrue(any(item["metric"] == "risk" for item in tests))

    def test_report_exports_strict_comparable_and_robustness_outputs(self):
        planner_calls: list[str] = []

        class DummyPlanner:
            def __init__(self, scenario, config, demo_config):
                self.scenario = scenario

            def run(self, optimizer_name="kemm"):
                planner_calls.append(optimizer_name)
                scenario_key = self.scenario.name.lower().replace(" ", "_").replace("-", "_")
                return self_test._stub_episode(scenario_key, optimizer_name=optimizer_name)

        demo = build_default_demo_config()
        demo.n_runs = 1
        demo.report_algorithms = ("kemm", "random")
        demo.episode_cache_enabled = False
        self_test = self
        tmp_root = Path("ship_simulation/outputs/test_artifacts") / f"strict_robust_stub_{int(np.random.randint(1_000_000))}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            with patch.object(run_report_module, "RollingHorizonPlanner", DummyPlanner):
                run_report_module.generate_report_with_config(
                    config=self.config,
                    demo_config=demo,
                    output_root=tmp_root,
                    plot_config=ShipPlotConfig(style=PublicationStyle(dpi=120)),
                    scenario_keys=["crossing"],
                    algorithm_keys=["kemm", "random"],
                    verbose=False,
                    n_runs=1,
                    render_figures=False,
                    strict_comparable=True,
                    robustness_sweep=True,
                    robustness_levels=[0.0, 0.5],
                    robustness_scenarios=["crossing"],
                )
            metadata = json.loads((tmp_root / "raw" / "report_metadata.json").read_text(encoding="utf-8"))
            robustness = json.loads((tmp_root / "raw" / "robustness_summary.json").read_text(encoding="utf-8"))
            algorithms = [item["key"] for item in metadata["algorithms"]]
            self.assertIn("random_matched", algorithms)
            self.assertTrue(metadata["strict_comparable"])
            self.assertTrue(metadata["robustness_sweep_enabled"])
            self.assertEqual(robustness["levels"], [0.0, 0.5])
            self.assertGreater(metadata["statistical_test_count"], 0)
        finally:
            for path in sorted(tmp_root.rglob("*"), reverse=True):
                if path.is_file():
                    try:
                        path.unlink()
                    except PermissionError:
                        pass
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            if tmp_root.exists():
                try:
                    tmp_root.rmdir()
                except OSError:
                    pass
        self.assertIn("random", planner_calls)


if __name__ == "__main__":
    unittest.main()
