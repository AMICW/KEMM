"""将船舶场景封装为标准多目标优化问题。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ship_simulation.config import ProblemConfig
from ship_simulation.core.collision_risk import RiskBreakdown, ShipDomainRiskModel
from ship_simulation.core.environment import EnvironmentField
from ship_simulation.core.fuel_model import FuelConsumptionModel
from ship_simulation.core.ship_model import NomotoShip, Trajectory
from ship_simulation.scenario.encounter import EncounterScenario


@dataclass
class EvaluationResult:
    """单个候选解的完整评估结果。"""

    objectives: np.ndarray
    own_trajectory: Trajectory
    target_trajectories: List[Trajectory]
    risk: RiskBreakdown
    reached_goal: bool
    terminal_distance: float
    analysis_metrics: dict[str, float] = field(default_factory=dict)


class ShipTrajectoryProblem:
    """船舶多目标轨迹规划问题。"""

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig):
        self.scenario = scenario
        self.config = config
        self.environment = EnvironmentField(
            config.environment,
            scalar_layers=list(scenario.scalar_fields),
            vector_layers=list(scenario.vector_fields),
        )
        self.own_ship_model = NomotoShip(config.ship, config.simulation, self.environment)
        self.target_ship_model = NomotoShip(config.ship, config.simulation, self.environment)
        self.risk_model = ShipDomainRiskModel(config.ship, config.domain, config)
        self.fuel_model = FuelConsumptionModel(config.environment)

        self.n_waypoints = config.num_intermediate_waypoints
        self.n_var = self.n_waypoints * 3
        self.n_obj = 3
        self.var_bounds = self._build_bounds()
        self._objective_cache: dict[bytes, np.ndarray] = {}

    def _build_bounds(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.scenario.area
        vmin, vmax = self.config.speed_bounds
        bounds: List[Tuple[float, float]] = []
        for _ in range(self.n_waypoints):
            bounds.extend([(xmin, xmax), (ymin, ymax), (vmin, vmax)])
        return np.asarray(bounds, dtype=float)

    def initial_guess(self) -> np.ndarray:
        start = self.scenario.own_ship.initial_state.position()
        goal = np.asarray(self.scenario.own_ship.goal, dtype=float)
        fractions = np.linspace(0.0, 1.0, self.n_waypoints + 2)[1:-1]
        vector = np.zeros(self.n_var, dtype=float)
        cruise_speed = float(np.mean(self.config.speed_bounds))
        for idx, frac in enumerate(fractions):
            waypoint = start + frac * (goal - start)
            base = idx * 3
            vector[base : base + 2] = waypoint
            vector[base + 2] = cruise_speed
        return vector

    def decode(self, decision_vector: Sequence[float]) -> Tuple[List[np.ndarray], List[float]]:
        vector = np.asarray(decision_vector, dtype=float)
        if vector.size != self.n_var:
            raise ValueError(f"Expected decision vector of length {self.n_var}, got {vector.size}.")
        waypoints: List[np.ndarray] = []
        speeds: List[float] = []
        for idx in range(self.n_waypoints):
            base = idx * 3
            waypoints.append(np.array([vector[base], vector[base + 1]], dtype=float))
            speeds.append(float(vector[base + 2]))
        waypoints.append(np.asarray(self.scenario.own_ship.goal, dtype=float))
        speeds.append(speeds[-1] if speeds else float(np.mean(self.config.speed_bounds)))
        return waypoints, speeds

    def _analysis_metrics(self, own_trajectory: Trajectory, risk: RiskBreakdown) -> dict[str, float]:
        heading_variation = float(np.sum(np.abs(np.diff(own_trajectory.headings)))) if len(own_trajectory.headings) > 1 else 0.0
        control_effort = float(np.sum(np.abs(np.diff(own_trajectory.commanded_yaw_rates)))) if len(own_trajectory.commanded_yaw_rates) > 1 else 0.0
        smoothness = float(np.sum(np.diff(own_trajectory.positions, n=2, axis=0) ** 2)) if len(own_trajectory.positions) > 2 else 0.0
        return {
            "minimum_clearance": float(risk.min_clearance),
            "minimum_static_clearance": float(risk.min_static_clearance),
            "minimum_ship_distance": float(risk.min_ship_distance),
            "minimum_dcpa": float(risk.min_dcpa),
            "minimum_tcpa": float(risk.min_tcpa),
            "smoothness": smoothness,
            "control_effort": control_effort,
            "heading_variation": heading_variation,
            "max_yaw_rate": float(np.max(np.abs(own_trajectory.yaw_rates))) if own_trajectory.yaw_rates.size else 0.0,
            "max_commanded_yaw_rate": float(np.max(np.abs(own_trajectory.commanded_yaw_rates))) if own_trajectory.commanded_yaw_rates.size else 0.0,
            "terminal_error": float(own_trajectory.terminal_distance),
            "mean_environment_risk": float(np.mean(risk.environment_risk_series)) if risk.environment_risk_series.size else 0.0,
        }

    def score_trajectory_bundle(
        self,
        own_trajectory: Trajectory,
        target_trajectories: Sequence[Trajectory],
        *,
        bounds_penalty: float = 0.0,
    ) -> EvaluationResult:
        """对一组轨迹做统一打分，供单次仿真和 episode 汇总复用。"""

        risk = self.risk_model.evaluate(
            own_trajectory,
            target_trajectories,
            static_obstacles=self.scenario.static_obstacles,
            environment=self.environment,
            colreg_roles=self.scenario.metadata.colreg_roles,
            target_names=[target.name for target in self.scenario.target_ships],
        )
        terminal_distance = float(own_trajectory.terminal_distance)
        clearance = float(risk.min_clearance)
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
        fuel = self.fuel_model.integrate(own_trajectory) + terminal_fuel_penalty + bounds_penalty + 2.5 * safety_penalty
        total_time = float(own_trajectory.times[-1] - own_trajectory.times[0]) + terminal_time_penalty + bounds_penalty + 0.25 * safety_penalty
        collision_risk = (
            self.config.domain_risk_weight * float(risk.max_risk)
            + (1.0 - self.config.domain_risk_weight) * float(risk.mean_risk)
            + terminal_risk_penalty
            + bounds_penalty
            + 0.025 * safety_penalty
            + self.config.intrusion_risk_penalty_per_second * float(risk.intrusion_time)
        )
        objectives = np.array([fuel, total_time, collision_risk], dtype=float)
        metrics = self._analysis_metrics(own_trajectory, risk)
        metrics["clearance_shortfall"] = clearance_shortfall
        metrics["hard_intrusion"] = hard_intrusion
        return EvaluationResult(
            objectives=objectives,
            own_trajectory=own_trajectory,
            target_trajectories=list(target_trajectories),
            risk=risk,
            reached_goal=own_trajectory.reached_goal,
            terminal_distance=terminal_distance,
            analysis_metrics=metrics,
        )

    def simulate(self, decision_vector: Sequence[float]) -> EvaluationResult:
        vector = np.asarray(decision_vector, dtype=float)
        clipped = np.clip(vector, self.var_bounds[:, 0], self.var_bounds[:, 1])
        bounds_penalty = float(np.linalg.norm(vector - clipped, ord=1) * self.config.penalty_out_of_bounds)
        waypoints, speeds = self.decode(clipped)
        own_trajectory = self.own_ship_model.simulate_route(
            initial_state=self.scenario.own_ship.initial_state,
            waypoints=waypoints,
            segment_speeds=speeds,
        )
        target_trajectories = [
            self.target_ship_model.simulate_constant_velocity(target.initial_state)
            for target in self.scenario.target_ships
        ]
        return self.score_trajectory_bundle(own_trajectory, target_trajectories, bounds_penalty=bounds_penalty)

    def _cache_key(self, decision_vector: Sequence[float]) -> bytes:
        clipped = np.clip(np.asarray(decision_vector, dtype=float), self.var_bounds[:, 0], self.var_bounds[:, 1])
        if self.config.population_cache_decimals >= 0:
            clipped = np.round(clipped, decimals=int(self.config.population_cache_decimals))
        return np.ascontiguousarray(clipped, dtype=np.float64).tobytes()

    def evaluate(self, decision_vector: Sequence[float]) -> np.ndarray:
        if self.config.population_evaluation_cache:
            key = self._cache_key(decision_vector)
            cached = self._objective_cache.get(key)
            if cached is not None:
                return cached.copy()
        objectives = self.simulate(decision_vector).objectives
        if self.config.population_evaluation_cache:
            self._objective_cache[key] = objectives.copy()
        return objectives

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        pop = np.atleast_2d(np.asarray(population, dtype=float))
        if not self.config.population_evaluation_cache:
            return np.vstack([self.evaluate(individual) for individual in pop])

        inverse = np.zeros(len(pop), dtype=int)
        unique_vectors: list[np.ndarray] = []
        key_to_index: dict[bytes, int] = {}
        for idx, individual in enumerate(pop):
            key = self._cache_key(individual)
            mapped = key_to_index.get(key)
            if mapped is None:
                mapped = len(unique_vectors)
                key_to_index[key] = mapped
                unique_vectors.append(np.asarray(individual, dtype=float))
            inverse[idx] = mapped
        unique_objectives = np.vstack([self.evaluate(individual) for individual in unique_vectors])
        return unique_objectives[inverse]

    def describe(self) -> Dict[str, object]:
        return {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "var_bounds": self.var_bounds.copy(),
            "initial_guess": self.initial_guess(),
            "scenario_name": self.scenario.name,
            "static_obstacles": len(self.scenario.static_obstacles),
            "traffic_agents": len(self.scenario.target_ships),
            "environment_layers": self.environment.describe_layers(),
        }
