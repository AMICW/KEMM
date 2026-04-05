"""滚动重规划执行层。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence
import time

import numpy as np

from ship_simulation.config import DemoConfig, ProblemConfig
from ship_simulation.optimizer.baseline_solver import EvolutionaryOptimizationResult, ShipNSGAStyleOptimizer
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.kemm_solver import KEMMOptimizationResult, ShipKEMMOptimizer
from ship_simulation.optimizer.problem import EvaluationResult, ShipTrajectoryProblem
from ship_simulation.optimizer.selection import select_representative_index
from ship_simulation.scenario.encounter import EncounterScenario
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


class RollingHorizonPlanner:
    """将局部求解器包装成完整滚动重规划 episode。"""

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig, demo_config: DemoConfig):
        self.scenario = scenario
        self.config = config
        self.demo_config = demo_config
        self.dt = self.config.simulation.dt

    def run(self, optimizer_name: str = "kemm") -> PlanningEpisodeResult:
        episode_cfg = self.demo_config.episode
        current_time = 0.0
        own_state = self.scenario.own_ship.initial_state
        target_states = [target.initial_state for target in self.scenario.target_ships]
        step_results: list[PlanningStepResult] = []
        own_segments: list[Trajectory] = []
        target_segments: list[list[Trajectory]] = [[] for _ in self.scenario.target_ships]
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
        final_evaluation = scoring_problem.score_trajectory_bundle(merged_own, merged_targets)
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
