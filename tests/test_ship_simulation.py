import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np

from reporting_config import PublicationStyle, ShipPlotConfig, interactive_bundle_path
from ship_simulation.config import DemoConfig, KEMMConfig, apply_experiment_profile, build_default_config
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
        )

    def test_optimizer_context_matches_problem_shape(self):
        context = self.interface.build_context()
        self.assertEqual(context.n_obj, 3)
        self.assertEqual(context.n_var, self.config.num_intermediate_waypoints * 3)

        obj_func = self.interface.make_objective_function(context)
        pop = np.vstack([context.initial_guess, context.initial_guess])
        values = obj_func(pop, 0.0)
        self.assertEqual(values.shape, (2, 3))

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
        self.assertEqual(result.fitness.shape[1], 3)
        self.assertTrue(result.best_evaluation.reached_goal)

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


if __name__ == "__main__":
    unittest.main()
