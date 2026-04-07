"""滚动重规划执行层。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Sequence
import time

import numpy as np

from ship_simulation.config import DemoConfig, ProblemConfig, ScenarioChangeStepConfig
from ship_simulation.core.collision_risk import RiskBreakdown
from ship_simulation.core.environment import GaussianScalarField, GridScalarField, GridVectorField, UniformVectorField, VortexVectorField
from ship_simulation.optimizer.baseline_solver import EvolutionaryOptimizationResult, ShipNSGAStyleOptimizer
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.kemm_solver import KEMMOptimizationResult, ShipKEMMOptimizer
from ship_simulation.optimizer.problem import EvaluationResult, ShipTrajectoryProblem
from ship_simulation.optimizer.selection import select_representative_index
from ship_simulation.scenario.encounter import ChannelBoundary, EncounterScenario, KeepOutZone
from ship_simulation.core.ship_model import ShipState, Trajectory


@dataclass
class PlanningSnapshot:
    time_s: float
    own_position: np.ndarray
    target_positions: list[np.ndarray]
    risk: float
    minimum_clearance: float


@dataclass
class PlanningStepResult:
    step_index: int
    start_time: float
    optimizer_name: str
    runtime_s: float
    scenario: EncounterScenario
    best_decision: np.ndarray
    selected_evaluation: EvaluationResult
    pareto_decisions: np.ndarray
    pareto_objectives: np.ndarray
    history: list[dict[str, float]]
    applied_changes: list[dict[str, object]]


@dataclass
class PlanningEpisodeResult:
    scenario_name: str
    optimizer_name: str
    steps: list[PlanningStepResult]
    final_evaluation: EvaluationResult
    pareto_objectives: np.ndarray
    pareto_decisions: np.ndarray
    knee_index: int | None
    knee_objectives: np.ndarray | None
    snapshots: list[PlanningSnapshot]
    analysis_metrics: dict[str, float]
    convergence_history: list[dict[str, float]]
    terminated_reason: str
    experiment_profile: str = "baseline"
    change_history: list[dict[str, object]] | None = None

    @property
    def result(self) -> EvaluationResult:
        return self.final_evaluation


def _merge_trajectory_segments(segments: Sequence[Trajectory]) -> Trajectory:
    if not segments:
        raise ValueError("At least one trajectory segment is required.")

    def _concat(name: str, axis: int = 0) -> np.ndarray:
        arrays = []
        for idx, segment in enumerate(segments):
            value = getattr(segment, name)
            arrays.append(value if idx == 0 else value[1:])
        return np.concatenate(arrays, axis=axis)

    return Trajectory(
        times=_concat("times"),
        positions=_concat("positions"),
        headings=_concat("headings"),
        speeds=_concat("speeds"),
        yaw_rates=_concat("yaw_rates"),
        commanded_yaw_rates=_concat("commanded_yaw_rates"),
        drift_vectors=_concat("drift_vectors"),
        waypoint_indices=_concat("waypoint_indices"),
        reached_goal=segments[-1].reached_goal,
        terminal_distance=float(segments[-1].terminal_distance),
    )


def _slice_risk_breakdown(risk: RiskBreakdown, sample_count: int, *, dt: float) -> RiskBreakdown:
    end = min(sample_count, len(risk.risk_series))
    risk_series = risk.risk_series[:end].copy()
    domain_risk_series = risk.domain_risk_series[:end].copy()
    dcpa_risk_series = risk.dcpa_risk_series[:end].copy()
    obstacle_risk_series = risk.obstacle_risk_series[:end].copy()
    environment_risk_series = risk.environment_risk_series[:end].copy()
    clearance_series = risk.clearance_series[:end].copy()
    static_clearance_series = risk.static_clearance_series[:end].copy()
    ship_distance_series = risk.ship_distance_series[:end].copy()
    dcpa_series = risk.dcpa_series[:end].copy()
    tcpa_series = risk.tcpa_series[:end].copy()
    colreg_scale_series = risk.colreg_scale_series[:end].copy()
    finite_ship = ship_distance_series[np.isfinite(ship_distance_series)]
    finite_dcpa = dcpa_series[np.isfinite(dcpa_series)]
    finite_tcpa = tcpa_series[np.isfinite(tcpa_series)]
    return RiskBreakdown(
        max_risk=float(np.max(risk_series)) if risk_series.size else 0.0,
        mean_risk=float(np.mean(risk_series)) if risk_series.size else 0.0,
        intrusion_time=float(np.sum(risk_series >= 1.0) * dt),
        min_clearance=float(np.min(clearance_series)) if clearance_series.size else float("inf"),
        min_dcpa=float(np.min(finite_dcpa)) if finite_dcpa.size else float("inf"),
        min_tcpa=float(np.min(finite_tcpa)) if finite_tcpa.size else float("inf"),
        min_static_clearance=float(np.min(static_clearance_series)) if static_clearance_series.size else float("inf"),
        min_ship_distance=float(np.min(finite_ship)) if finite_ship.size else float("inf"),
        risk_series=risk_series,
        domain_risk_series=domain_risk_series,
        dcpa_risk_series=dcpa_risk_series,
        obstacle_risk_series=obstacle_risk_series,
        environment_risk_series=environment_risk_series,
        clearance_series=clearance_series,
        static_clearance_series=static_clearance_series,
        ship_distance_series=ship_distance_series,
        dcpa_series=dcpa_series,
        tcpa_series=tcpa_series,
        colreg_scale_series=colreg_scale_series,
    )


def _merge_risk_segments(segments: Sequence[RiskBreakdown], *, dt: float) -> RiskBreakdown:
    if not segments:
        raise ValueError("At least one risk segment is required.")

    def _concat(name: str) -> np.ndarray:
        arrays = []
        for idx, segment in enumerate(segments):
            values = getattr(segment, name)
            arrays.append(values if idx == 0 else values[1:])
        return np.concatenate(arrays, axis=0)

    risk_series = _concat("risk_series")
    domain_risk_series = _concat("domain_risk_series")
    dcpa_risk_series = _concat("dcpa_risk_series")
    obstacle_risk_series = _concat("obstacle_risk_series")
    environment_risk_series = _concat("environment_risk_series")
    clearance_series = _concat("clearance_series")
    static_clearance_series = _concat("static_clearance_series")
    ship_distance_series = _concat("ship_distance_series")
    dcpa_series = _concat("dcpa_series")
    tcpa_series = _concat("tcpa_series")
    colreg_scale_series = _concat("colreg_scale_series")
    finite_ship = ship_distance_series[np.isfinite(ship_distance_series)]
    finite_dcpa = dcpa_series[np.isfinite(dcpa_series)]
    finite_tcpa = tcpa_series[np.isfinite(tcpa_series)]
    return RiskBreakdown(
        max_risk=float(np.max(risk_series)) if risk_series.size else 0.0,
        mean_risk=float(np.mean(risk_series)) if risk_series.size else 0.0,
        intrusion_time=float(np.sum(risk_series >= 1.0) * dt),
        min_clearance=float(np.min(clearance_series)) if clearance_series.size else float("inf"),
        min_dcpa=float(np.min(finite_dcpa)) if finite_dcpa.size else float("inf"),
        min_tcpa=float(np.min(finite_tcpa)) if finite_tcpa.size else float("inf"),
        min_static_clearance=float(np.min(static_clearance_series)) if static_clearance_series.size else float("inf"),
        min_ship_distance=float(np.min(finite_ship)) if finite_ship.size else float("inf"),
        risk_series=risk_series,
        domain_risk_series=domain_risk_series,
        dcpa_risk_series=dcpa_risk_series,
        obstacle_risk_series=obstacle_risk_series,
        environment_risk_series=environment_risk_series,
        clearance_series=clearance_series,
        static_clearance_series=static_clearance_series,
        ship_distance_series=ship_distance_series,
        dcpa_series=dcpa_series,
        tcpa_series=tcpa_series,
        colreg_scale_series=colreg_scale_series,
    )


class RollingHorizonPlanner:
    """将局部求解器包装成完整滚动重规划 episode。"""

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig, demo_config: DemoConfig):
        self.scenario = scenario
        self.config = config
        self.demo_config = demo_config
        self.dt = self.config.simulation.dt

    @staticmethod
    def _scale_scalar_layers(layers: Sequence[object], amplitude_scale: float) -> list[object]:
        tuned: list[object] = []
        for layer in layers:
            if isinstance(layer, GaussianScalarField):
                tuned.append(replace(layer, amplitude=float(layer.amplitude * amplitude_scale)))
            elif isinstance(layer, GridScalarField):
                tuned.append(replace(layer, values=np.asarray(layer.values, dtype=float) * amplitude_scale))
            else:
                tuned.append(layer)
        return tuned

    @staticmethod
    def _scale_vector_layers(layers: Sequence[object], speed_scale: float) -> list[object]:
        tuned: list[object] = []
        for layer in layers:
            if isinstance(layer, UniformVectorField):
                tuned.append(replace(layer, speed=float(layer.speed * speed_scale)))
            elif isinstance(layer, GridVectorField):
                tuned.append(
                    replace(
                        layer,
                        u_values=np.asarray(layer.u_values, dtype=float) * speed_scale,
                        v_values=np.asarray(layer.v_values, dtype=float) * speed_scale,
                    )
                )
            elif isinstance(layer, VortexVectorField):
                tuned.append(replace(layer, strength=float(layer.strength * speed_scale)))
            else:
                tuned.append(layer)
        return tuned

    @staticmethod
    def _scale_channel_boundary(obstacle: ChannelBoundary, y_scale: float) -> ChannelBoundary:
        vertices = np.asarray(obstacle.vertices, dtype=float)
        centroid = np.mean(vertices, axis=0)
        scaled = vertices.copy()
        scaled[:, 1] = centroid[1] + y_scale * (scaled[:, 1] - centroid[1])
        return replace(obstacle, vertices=scaled)

    def _inject_channel_closure(self, scenario: EncounterScenario, change: ScenarioChangeStepConfig) -> list[KeepOutZone]:
        xmin, xmax, ymin, ymax = scenario.area
        x_center = float(change.closure_center_x if change.closure_center_x is not None else xmin + 0.58 * (xmax - xmin))
        half_width = max(float(change.closure_width) * 0.5, 60.0)
        gap_center = ymin + float(change.closure_gap_center_ratio) * (ymax - ymin)
        gap_span = max(float(change.closure_gap_span_ratio) * (ymax - ymin), self.config.safety_clearance * 1.2)
        gap_half = gap_span * 0.5
        x0 = x_center - half_width
        x1 = x_center + half_width
        closures: list[KeepOutZone] = []
        lower_top = gap_center - gap_half
        upper_bottom = gap_center + gap_half
        if lower_top > ymin:
            closures.append(
                KeepOutZone(
                    name=f"Closure Lower Step {change.step_index}",
                    vertices=np.array([[x0, ymin], [x1, ymin], [x1, lower_top], [x0, lower_top]], dtype=float),
                    color="#f97316",
                )
            )
        if upper_bottom < ymax:
            closures.append(
                KeepOutZone(
                    name=f"Closure Upper Step {change.step_index}",
                    vertices=np.array([[x0, upper_bottom], [x1, upper_bottom], [x1, ymax], [x0, ymax]], dtype=float),
                    color="#f97316",
                )
            )
        return closures

    def _changes_for_step(self, step_index: int) -> list[ScenarioChangeStepConfig]:
        experiment = self.config.experiment
        if not experiment.enabled:
            return []
        return [change for change in experiment.change_schedule if int(change.step_index) == int(step_index)]

    def _apply_experiment_changes(
        self,
        scenario: EncounterScenario,
        config: ProblemConfig,
        step_index: int,
    ) -> tuple[EncounterScenario, ProblemConfig, list[dict[str, object]]]:
        scheduled = self._changes_for_step(step_index)
        if not scheduled:
            return scenario, config, []

        tuned_scenario = scenario
        tuned_config = config
        applied: list[dict[str, object]] = []
        for change in scheduled:
            tuned_targets = [
                replace(
                    target,
                    initial_state=replace(
                        target.initial_state,
                        heading=float(target.initial_state.heading + np.deg2rad(change.target_heading_delta_deg)),
                        speed=float(target.initial_state.speed * change.target_speed_scale),
                    ),
                )
                for target in tuned_scenario.target_ships
            ]
            tuned_obstacles = []
            for obstacle in tuned_scenario.static_obstacles:
                if isinstance(obstacle, ChannelBoundary) and abs(change.channel_width_scale - 1.0) > 1e-9:
                    tuned_obstacles.append(self._scale_channel_boundary(obstacle, change.channel_width_scale))
                else:
                    tuned_obstacles.append(obstacle)
            if change.inject_channel_closure:
                tuned_obstacles.extend(self._inject_channel_closure(tuned_scenario, change))
            tuned_scenario = replace(
                tuned_scenario,
                target_ships=tuned_targets,
                static_obstacles=tuned_obstacles,
                scalar_fields=self._scale_scalar_layers(tuned_scenario.scalar_fields, change.scalar_amplitude_scale),
                vector_fields=self._scale_vector_layers(tuned_scenario.vector_fields, change.vector_speed_scale),
            )
            tuned_config = replace(
                tuned_config,
                environment=replace(
                    tuned_config.environment,
                    current_speed=float(tuned_config.environment.current_speed * change.current_speed_scale),
                ),
            )
            applied.append(asdict(change))
        return tuned_scenario, tuned_config, applied

    def run(self, optimizer_name: str = "kemm") -> PlanningEpisodeResult:
        episode_cfg = self.demo_config.episode
        current_time = 0.0
        own_state = self.scenario.own_ship.initial_state
        target_states = [target.initial_state for target in self.scenario.target_ships]
        step_results: list[PlanningStepResult] = []
        own_segments: list[Trajectory] = []
        target_segments: list[list[Trajectory]] = [[] for _ in self.scenario.target_ships]
        risk_segments: list[RiskBreakdown] = []
        convergence_history: list[dict[str, float]] = []
        terminated_reason = "max_replans"

        for step_idx in range(episode_cfg.max_replans):
            goal_margin = max(
                self.config.simulation.arrival_tolerance,
                self.config.ship.length * 0.4,
            )
            local_cfg = replace(
                self.config,
                simulation=replace(self.config.simulation, horizon=episode_cfg.local_horizon),
            )
            local_scenario = self.scenario.with_updated_states(
                own_state,
                target_states,
                name_suffix=f"[step {step_idx}]",
            )
            local_scenario, local_cfg, applied_changes = self._apply_experiment_changes(local_scenario, local_cfg, step_idx)
            interface = ShipOptimizerInterface(local_scenario, local_cfg)
            solver_start = time.perf_counter()
            solve_result = self._solve_local_problem(interface, optimizer_name)
            runtime_s = time.perf_counter() - solver_start
            step_result = PlanningStepResult(
                step_index=step_idx,
                start_time=current_time,
                optimizer_name=optimizer_name,
                runtime_s=runtime_s,
                scenario=local_scenario,
                best_decision=solve_result.best_decision,
                selected_evaluation=solve_result.best_evaluation,
                pareto_decisions=solve_result.pareto_decisions,
                pareto_objectives=solve_result.pareto_objectives,
                history=list(solve_result.history),
                applied_changes=applied_changes,
            )
            step_results.append(step_result)
            convergence_history.extend(self._offset_history(solve_result.history, step_idx))

            if solve_result.best_evaluation.reached_goal or solve_result.best_evaluation.terminal_distance <= goal_margin:
                execute_count = len(solve_result.best_evaluation.own_trajectory.times)
            else:
                execute_count = min(
                    len(solve_result.best_evaluation.own_trajectory.times),
                    max(2, int(episode_cfg.execution_horizon / self.dt) + 1),
                )
            executed_own = self._slice_trajectory(
                solve_result.best_evaluation.own_trajectory,
                execute_count,
                time_offset=current_time,
            )
            own_segments.append(executed_own)
            risk_segments.append(_slice_risk_breakdown(solve_result.best_evaluation.risk, execute_count, dt=self.dt))
            for target_idx, traj in enumerate(solve_result.best_evaluation.target_trajectories):
                executed_target = self._slice_trajectory(traj, execute_count, time_offset=current_time)
                target_segments[target_idx].append(executed_target)

            own_state = executed_own.final_state()
            for target_idx, segments in enumerate(target_segments):
                target_states[target_idx] = segments[-1].final_state()
            current_time += float(executed_own.times[-1] - executed_own.times[0])

            goal = np.asarray(self.scenario.own_ship.goal, dtype=float)
            if float(np.linalg.norm(own_state.position() - goal)) <= goal_margin:
                terminated_reason = "reached_goal"
                break
            if solve_result.best_evaluation.risk.max_risk >= episode_cfg.stop_on_high_risk:
                terminated_reason = "risk_cutoff"
                break

        merged_own = _merge_trajectory_segments(own_segments)
        merged_targets = [_merge_trajectory_segments(segments) for segments in target_segments] if target_segments else []
        merged_own.reached_goal = float(np.linalg.norm(merged_own.positions[-1] - np.asarray(self.scenario.own_ship.goal, dtype=float))) <= max(
            self.config.simulation.arrival_tolerance,
            self.config.ship.length * 0.4,
        )
        merged_own.terminal_distance = float(np.linalg.norm(merged_own.positions[-1] - np.asarray(self.scenario.own_ship.goal, dtype=float)))
        scoring_problem = ShipTrajectoryProblem(self.scenario, self.config)
        merged_risk = _merge_risk_segments(risk_segments, dt=self.dt)
        terminal_distance = float(merged_own.terminal_distance)
        clearance = float(merged_risk.min_clearance)
        if np.isfinite(clearance):
            clearance_shortfall = max(0.0, self.config.safety_clearance - clearance)
            hard_intrusion = max(0.0, -clearance)
        else:
            clearance_shortfall = 0.0
            hard_intrusion = 0.0
        safety_penalty = (
            self.config.soft_clearance_penalty_per_meter * clearance_shortfall
            + self.config.hard_clearance_penalty_per_meter * hard_intrusion
        )
        terminal_fuel_penalty = terminal_distance * self.config.terminal_fuel_penalty_per_meter
        terminal_time_penalty = terminal_distance * self.config.terminal_time_penalty_per_meter
        terminal_risk_penalty = terminal_distance * self.config.terminal_risk_penalty_per_meter
        fuel = scoring_problem.fuel_model.integrate(merged_own) + terminal_fuel_penalty + 2.5 * safety_penalty
        total_time = float(merged_own.times[-1] - merged_own.times[0]) + terminal_time_penalty + 0.25 * safety_penalty
        collision_risk = (
            self.config.domain_risk_weight * float(merged_risk.max_risk)
            + (1.0 - self.config.domain_risk_weight) * float(merged_risk.mean_risk)
            + terminal_risk_penalty
            + 0.025 * safety_penalty
            + self.config.intrusion_risk_penalty_per_second * float(merged_risk.intrusion_time)
        )
        objectives = np.array([fuel, total_time, collision_risk], dtype=float)
        metrics = scoring_problem._analysis_metrics(merged_own, merged_risk)
        metrics["clearance_shortfall"] = clearance_shortfall
        metrics["hard_intrusion"] = hard_intrusion
        final_evaluation = EvaluationResult(
            objectives=objectives,
            own_trajectory=merged_own,
            target_trajectories=merged_targets,
            risk=merged_risk,
            reached_goal=merged_own.reached_goal,
            terminal_distance=terminal_distance,
            analysis_metrics=metrics,
        )
        final_evaluation.analysis_metrics["runtime"] = float(sum(step.runtime_s for step in step_results))
        final_evaluation.analysis_metrics["planning_steps"] = float(len(step_results))

        pareto_decisions, pareto_objectives = self._merge_pareto_sets(step_results)
        knee_index = self._detect_knee_point(pareto_objectives)
        knee_objectives = pareto_objectives[knee_index].copy() if knee_index is not None else None
        snapshots = self._build_snapshots(final_evaluation, episode_cfg.snapshot_count)
        analysis_metrics = dict(final_evaluation.analysis_metrics)
        analysis_metrics.update(
            {
                "fuel": float(final_evaluation.objectives[0]),
                "time": float(final_evaluation.objectives[1]),
                "risk": float(final_evaluation.objectives[2]),
                "intrusion_time": float(final_evaluation.risk.intrusion_time),
            }
        )
        if self.config.experiment.enabled:
            analysis_metrics["scheduled_change_count"] = float(sum(len(step.applied_changes) for step in step_results))
        return PlanningEpisodeResult(
            scenario_name=self.scenario.name,
            optimizer_name=optimizer_name,
            steps=step_results,
            final_evaluation=final_evaluation,
            pareto_objectives=pareto_objectives,
            pareto_decisions=pareto_decisions,
            knee_index=knee_index,
            knee_objectives=knee_objectives,
            snapshots=snapshots,
            analysis_metrics=analysis_metrics,
            convergence_history=convergence_history,
            terminated_reason=terminated_reason,
            experiment_profile=self.config.experiment.profile_name,
            change_history=[change for step in step_results for change in step.applied_changes],
        )

    def _solve_local_problem(self, interface: ShipOptimizerInterface, optimizer_name: str):
        key = optimizer_name.lower()
        if key == "kemm":
            return ShipKEMMOptimizer(interface=interface, demo_config=self.demo_config).optimize()
        if key == "nsga_style":
            return ShipNSGAStyleOptimizer(interface=interface, demo_config=self.demo_config).optimize()
        if key == "random":
            return self._run_random_baseline(interface)
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    def _run_random_baseline(self, interface: ShipOptimizerInterface):
        context = interface.build_context()
        rng = np.random.default_rng(self.demo_config.random_search_seed)
        lower = context.var_bounds[:, 0]
        upper = context.var_bounds[:, 1]
        decisions = rng.uniform(lower, upper, size=(self.demo_config.random_search_samples, context.n_var))
        decisions[0] = context.initial_guess
        fitness = context.evaluate_population(decisions)
        evaluations = [interface.simulate(individual) for individual in decisions]
        best_idx = select_representative_index(
            fitness,
            evaluations,
            self.config.objective_weights,
            safety_clearance=self.config.safety_clearance,
        )
        best_evaluation = evaluations[best_idx]
        pareto_idx = self._nondominated_indices(fitness)
        return EvolutionaryOptimizationResult(
            best_decision=decisions[best_idx].copy(),
            best_evaluation=best_evaluation,
            pareto_decisions=decisions[pareto_idx].copy(),
            pareto_objectives=fitness[pareto_idx].copy(),
            population=decisions.copy(),
            fitness=fitness.copy(),
            history=[{
                "generation": 0.0,
                "best_fuel": float(best_evaluation.objectives[0]),
                "best_time": float(best_evaluation.objectives[1]),
                "best_risk": float(best_evaluation.objectives[2]),
                "best_weighted_score": float(np.dot(
                    best_evaluation.objectives,
                    np.asarray(self.config.objective_weights, dtype=float) / max(float(np.sum(self.config.objective_weights)), 1e-9),
                )),
            }],
            runtime_s=0.0,
        )

    @staticmethod
    def _slice_trajectory(traj: Trajectory, sample_count: int, *, time_offset: float = 0.0) -> Trajectory:
        end = min(sample_count, len(traj.times))
        return Trajectory(
            times=traj.times[:end].copy() + float(time_offset),
            positions=traj.positions[:end].copy(),
            headings=traj.headings[:end].copy(),
            speeds=traj.speeds[:end].copy(),
            yaw_rates=traj.yaw_rates[:end].copy(),
            commanded_yaw_rates=traj.commanded_yaw_rates[:end].copy(),
            drift_vectors=traj.drift_vectors[:end].copy(),
            waypoint_indices=traj.waypoint_indices[:end].copy(),
            reached_goal=bool(traj.reached_goal and end == len(traj.times)),
            terminal_distance=float(traj.terminal_distance),
        )

    @staticmethod
    def _offset_history(history: Sequence[dict[str, float]], step_index: int) -> list[dict[str, float]]:
        return [dict(item, planning_step=float(step_index)) for item in history]

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b) and np.any(a < b)

    def _merge_pareto_sets(self, step_results: Sequence[PlanningStepResult]) -> tuple[np.ndarray, np.ndarray]:
        if not step_results:
            return np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float)
        final_step = step_results[-1]
        return final_step.pareto_decisions.copy(), final_step.pareto_objectives.copy()

    def _nondominated_indices(self, objectives: np.ndarray) -> np.ndarray:
        keep = []
        for i in range(len(objectives)):
            dominated = False
            for j in range(len(objectives)):
                if i != j and self._dominates(objectives[j], objectives[i]):
                    dominated = True
                    break
            if not dominated:
                keep.append(i)
        return np.asarray(keep, dtype=int)

    @staticmethod
    def _detect_knee_point(objectives: np.ndarray) -> int | None:
        if objectives.size == 0:
            return None
        span = np.ptp(objectives, axis=0)
        normalized = (objectives - objectives.min(axis=0)) / (span + 1e-9)
        scores = np.linalg.norm(normalized, axis=1)
        return int(np.argmin(scores))

    @staticmethod
    def _build_snapshots(evaluation: EvaluationResult, count: int) -> list[PlanningSnapshot]:
        if count <= 0 or len(evaluation.own_trajectory.times) == 0:
            return []
        indices = np.linspace(0, len(evaluation.own_trajectory.times) - 1, count, dtype=int)
        snapshots: list[PlanningSnapshot] = []
        for idx in indices:
            target_positions = [traj.positions[min(idx, len(traj.positions) - 1)].copy() for traj in evaluation.target_trajectories]
            snapshots.append(
                PlanningSnapshot(
                    time_s=float(evaluation.own_trajectory.times[idx]),
                    own_position=evaluation.own_trajectory.positions[idx].copy(),
                    target_positions=target_positions,
                    risk=float(evaluation.risk.risk_series[min(idx, len(evaluation.risk.risk_series) - 1)]),
                    minimum_clearance=float(evaluation.risk.clearance_series[min(idx, len(evaluation.risk.clearance_series) - 1)]),
                )
            )
        return snapshots
