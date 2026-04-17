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


def _copy_trajectory(traj: Trajectory) -> Trajectory:
    return Trajectory(
        times=traj.times.copy(),
        positions=traj.positions.copy(),
        headings=traj.headings.copy(),
        speeds=traj.speeds.copy(),
        yaw_rates=traj.yaw_rates.copy(),
        commanded_yaw_rates=traj.commanded_yaw_rates.copy(),
        drift_vectors=traj.drift_vectors.copy(),
        waypoint_indices=traj.waypoint_indices.copy(),
        reached_goal=bool(traj.reached_goal),
        terminal_distance=float(traj.terminal_distance),
    )


def _copy_risk_breakdown(risk: RiskBreakdown) -> RiskBreakdown:
    return RiskBreakdown(
        max_risk=float(risk.max_risk),
        mean_risk=float(risk.mean_risk),
        intrusion_time=float(risk.intrusion_time),
        min_clearance=float(risk.min_clearance),
        min_dcpa=float(risk.min_dcpa),
        min_tcpa=float(risk.min_tcpa),
        min_static_clearance=float(risk.min_static_clearance),
        min_ship_distance=float(risk.min_ship_distance),
        risk_series=risk.risk_series.copy(),
        domain_risk_series=risk.domain_risk_series.copy(),
        dcpa_risk_series=risk.dcpa_risk_series.copy(),
        obstacle_risk_series=risk.obstacle_risk_series.copy(),
        environment_risk_series=risk.environment_risk_series.copy(),
        clearance_series=risk.clearance_series.copy(),
        static_clearance_series=risk.static_clearance_series.copy(),
        ship_distance_series=risk.ship_distance_series.copy(),
        dcpa_series=risk.dcpa_series.copy(),
        tcpa_series=risk.tcpa_series.copy(),
        colreg_scale_series=risk.colreg_scale_series.copy(),
    )


def _copy_evaluation_result(result: EvaluationResult) -> EvaluationResult:
    return EvaluationResult(
        objectives=result.objectives.copy(),
        own_trajectory=_copy_trajectory(result.own_trajectory),
        target_trajectories=[_copy_trajectory(traj) for traj in result.target_trajectories],
        risk=_copy_risk_breakdown(result.risk),
        reached_goal=bool(result.reached_goal),
        terminal_distance=float(result.terminal_distance),
        analysis_metrics=dict(result.analysis_metrics),
    )


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
        self._target_trajectories = [
            self.target_ship_model.simulate_constant_velocity(target.initial_state)
            for target in self.scenario.target_ships
        ]

        self.n_waypoints = config.num_intermediate_waypoints
        self.n_var = self.n_waypoints * 3
        self.n_obj = 3
        self.var_bounds = self._build_bounds()
        self._objective_cache: dict[bytes, np.ndarray] = {}
        self._evaluation_cache: dict[bytes, EvaluationResult] = {}

    def _cached_evaluation(self, cache_key: bytes, *, copy_result: bool) -> EvaluationResult | None:
        cached = self._evaluation_cache.get(cache_key)
        if cached is None:
            return None
        return _copy_evaluation_result(cached) if copy_result else cached

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

    def _route_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        start = self.scenario.own_ship.initial_state.position()
        goal = np.asarray(self.scenario.own_ship.goal, dtype=float)
        route = goal - start
        length = float(np.linalg.norm(route))
        if length <= 1e-9:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = route / length
        normal = np.array([-direction[1], direction[0]], dtype=float)
        fractions = np.linspace(0.0, 1.0, self.n_waypoints + 2)[1:-1]
        base_points = np.vstack([start + frac * route for frac in fractions]).astype(float)
        return start, goal, direction, normal, length, fractions, base_points

    def _clip_waypoint(self, point: Sequence[float]) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.scenario.area
        margin = max(self.config.ship.length * 0.35, 30.0)
        clipped = np.asarray(point, dtype=float).copy()
        clipped[0] = float(np.clip(clipped[0], xmin + margin, xmax - margin))
        clipped[1] = float(np.clip(clipped[1], ymin + margin, ymax - margin))
        return clipped

    def _encode_seed_vector(self, intermediate_waypoints: np.ndarray, segment_speeds: Sequence[float] | None = None) -> np.ndarray:
        vector = np.zeros(self.n_var, dtype=float)
        if segment_speeds is None:
            segment_speeds = [float(np.mean(self.config.speed_bounds))] * self.n_waypoints
        for idx in range(self.n_waypoints):
            base = idx * 3
            vector[base : base + 2] = self._clip_waypoint(intermediate_waypoints[idx])
            vector[base + 2] = float(np.clip(segment_speeds[idx], *self.config.speed_bounds))
        return vector

    def _static_hazard_descriptors(self) -> list[tuple[np.ndarray, float]]:
        hazards: list[tuple[np.ndarray, float]] = []
        for obstacle in self.scenario.static_obstacles:
            if hasattr(obstacle, "radius"):
                center = np.asarray(obstacle.center, dtype=float)
                radius = float(obstacle.radius)
            else:
                vertices = np.asarray(obstacle.vertices, dtype=float)
                center = np.mean(vertices, axis=0)
                radius = float(np.max(np.linalg.norm(vertices - center, axis=1))) if len(vertices) else 0.0
            hazards.append((center, radius))
        return hazards

    def _dynamic_hazard_descriptors(self) -> list[tuple[np.ndarray, float]]:
        return [
            (target.initial_state.position(), 0.6 * self.config.safety_clearance)
            for target in self.scenario.target_ships
        ]

    def heuristic_seed_vectors(
        self,
        max_count: int,
        *,
        offset_scale: float = 1.0,
    ) -> np.ndarray:
        """Generate scenario-aware warm-start routes that explicitly include safe detours."""

        if max_count <= 0:
            return np.zeros((0, self.n_var), dtype=float)

        start, goal, direction, normal, length, fractions, base_points = self._route_geometry()
        if self.n_waypoints == 0:
            return np.zeros((0, self.n_var), dtype=float)

        route = goal - start
        cruise_speed = float(np.mean(self.config.speed_bounds))
        safety = float(self.config.safety_clearance) * max(float(offset_scale), 0.1)
        envelope = np.sin(np.pi * fractions)

        static_hazards = self._static_hazard_descriptors()
        dynamic_hazards = self._dynamic_hazard_descriptors()
        hazard_side_balance = 0.0
        for center, _radius in static_hazards + dynamic_hazards:
            hazard_side_balance += float(np.dot(np.asarray(center, dtype=float) - start, normal))
        preferred_signs = (-1.0, 1.0) if hazard_side_balance > 0.0 else (1.0, -1.0)

        seeds: list[np.ndarray] = []
        seen: set[bytes] = set()

        def add_seed(points: np.ndarray, speed_scale: float = 1.0) -> None:
            speeds = np.full(self.n_waypoints, cruise_speed * speed_scale, dtype=float)
            vector = self._encode_seed_vector(points, speeds)
            key = self._cache_key(vector)
            if key not in seen:
                seen.add(key)
                seeds.append(vector)

        add_seed(base_points, speed_scale=1.0)

        corridor_amplitudes = safety * np.array([0.85, 1.35, 1.9], dtype=float)
        for sign in preferred_signs:
            for amplitude in corridor_amplitudes:
                shifted = base_points + np.outer(envelope, normal * sign * amplitude)
                speed_scale = 0.96 if amplitude < corridor_amplitudes[-1] else 0.92
                add_seed(shifted, speed_scale=speed_scale)
                if len(seeds) >= max_count:
                    return np.asarray(seeds[:max_count], dtype=float)

        def hazard_order_key(item: tuple[np.ndarray, float]) -> tuple[float, float]:
            center, _radius = item
            rel = np.asarray(center, dtype=float) - start
            progress = float(np.dot(rel, direction) / max(length, 1e-9))
            projected = start + np.clip(progress, 0.0, 1.0) * route
            lateral = float(np.linalg.norm(np.asarray(center, dtype=float) - projected))
            return lateral, abs(progress - 0.5)

        for center, radius in sorted(static_hazards + dynamic_hazards, key=hazard_order_key)[:3]:
            rel = np.asarray(center, dtype=float) - start
            progress = float(np.clip(np.dot(rel, direction) / max(length, 1e-9), 0.05, 0.95))
            local_envelope = np.exp(-((fractions - progress) ** 2) / 0.03)
            local_offset = max(radius + 0.9 * safety, 1.1 * safety)
            nearest_idx = int(np.argmin(np.abs(fractions - progress)))
            for sign in preferred_signs:
                guided = base_points + np.outer(local_envelope, normal * sign * local_offset)
                guided[nearest_idx] = np.asarray(center, dtype=float) + normal * sign * local_offset
                add_seed(guided, speed_scale=0.9)
                if len(seeds) >= max_count:
                    return np.asarray(seeds[:max_count], dtype=float)

        return np.asarray(seeds[:max_count], dtype=float)

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

    def penalty_terms(
        self,
        risk: RiskBreakdown,
        terminal_distance: float,
        *,
        bounds_penalty: float = 0.0,
        terminal_penalty_scale: float = 1.0,
    ) -> dict[str, float]:
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
        terminal_fuel_penalty = terminal_penalty_scale * terminal_distance * self.config.terminal_fuel_penalty_per_meter
        terminal_time_penalty = terminal_penalty_scale * terminal_distance * self.config.terminal_time_penalty_per_meter
        terminal_risk_penalty = terminal_penalty_scale * terminal_distance * self.config.terminal_risk_penalty_per_meter
        intrusion_penalty = self.config.intrusion_risk_penalty_per_second * float(risk.intrusion_time)
        cv = (
            bounds_penalty
            + safety_penalty
            + intrusion_penalty
        )
        return {
            "clearance_shortfall": float(clearance_shortfall),
            "hard_intrusion": float(hard_intrusion),
            "safety_penalty": float(safety_penalty),
            "terminal_fuel_penalty": float(terminal_fuel_penalty),
            "terminal_time_penalty": float(terminal_time_penalty),
            "terminal_risk_penalty": float(terminal_risk_penalty),
            "intrusion_penalty": float(intrusion_penalty),
            "cv": float(cv),
        }

    def compose_objectives(
        self,
        *,
        fuel: float,
        total_time: float,
        risk: RiskBreakdown,
        terminal_distance: float,
        bounds_penalty: float = 0.0,
        terminal_penalty_scale: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        penalties = self.penalty_terms(
            risk,
            terminal_distance,
            bounds_penalty=bounds_penalty,
            terminal_penalty_scale=terminal_penalty_scale,
        )
        collision_risk = float(
            self.config.domain_risk_weight * float(risk.max_risk)
            + (1.0 - self.config.domain_risk_weight) * float(risk.mean_risk)
            + self.config.risk_safety_penalty_weight * penalties["safety_penalty"]
            + penalties["intrusion_penalty"]
        )
        objectives = np.array(
            [
                float(fuel) + penalties["terminal_fuel_penalty"] + self.config.fuel_safety_penalty_weight * penalties["safety_penalty"],
                float(total_time) + penalties["terminal_time_penalty"] + self.config.time_safety_penalty_weight * penalties["safety_penalty"],
                collision_risk,
            ],
            dtype=float,
        )
        return objectives, penalties

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
        fuel = float(self.fuel_model.integrate(own_trajectory))
        total_time = float(own_trajectory.times[-1] - own_trajectory.times[0])
        objectives, penalties = self.compose_objectives(
            fuel=fuel,
            total_time=total_time,
            risk=risk,
            terminal_distance=terminal_distance,
            bounds_penalty=bounds_penalty,
            terminal_penalty_scale=self.config.local_terminal_penalty_scale,
        )
        metrics = self._analysis_metrics(own_trajectory, risk)
        metrics["clearance_shortfall"] = penalties["clearance_shortfall"]
        metrics["hard_intrusion"] = penalties["hard_intrusion"]
        metrics["cv"] = penalties["cv"]
        return EvaluationResult(
            objectives=objectives,
            own_trajectory=own_trajectory,
            target_trajectories=list(target_trajectories),
            risk=risk,
            reached_goal=own_trajectory.reached_goal,
            terminal_distance=terminal_distance,
            analysis_metrics=metrics,
        )

    def simulate(self, decision_vector: Sequence[float], *, copy_result: bool = True) -> EvaluationResult:
        vector = np.asarray(decision_vector, dtype=float)
        clipped = np.clip(vector, self.var_bounds[:, 0], self.var_bounds[:, 1])
        bounds_penalty = float(np.linalg.norm(vector - clipped, ord=1) * self.config.penalty_out_of_bounds)
        cache_key = self._cache_key(clipped)
        if self.config.population_evaluation_cache:
            cached = self._cached_evaluation(cache_key, copy_result=copy_result)
            if cached is not None:
                return cached
        waypoints, speeds = self.decode(clipped)
        own_trajectory = self.own_ship_model.simulate_route(
            initial_state=self.scenario.own_ship.initial_state,
            waypoints=waypoints,
            segment_speeds=speeds,
        )
        target_trajectories = self._target_trajectories
        result = self.score_trajectory_bundle(own_trajectory, target_trajectories, bounds_penalty=bounds_penalty)
        if self.config.population_evaluation_cache:
            self._evaluation_cache[cache_key] = _copy_evaluation_result(result)
            self._objective_cache[cache_key] = np.append(result.objectives, result.analysis_metrics.get("cv", 0.0)).copy()
            return self._cached_evaluation(cache_key, copy_result=copy_result) or result
        return _copy_evaluation_result(result) if copy_result else result

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
        result = self.simulate(decision_vector)
        # 为EA底层提供 [Objectives ..., CV]
        eval_array = np.append(result.objectives, result.analysis_metrics.get("cv", 0.0))
        if self.config.population_evaluation_cache:
            self._objective_cache[key] = eval_array.copy()
        return eval_array

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

    def simulate_population(
        self,
        population: np.ndarray,
        *,
        copy_results: bool = True,
    ) -> list[EvaluationResult]:
        pop = np.atleast_2d(np.asarray(population, dtype=float))
        if len(pop) == 0:
            return []
        if not self.config.population_evaluation_cache:
            return [self.simulate(individual, copy_result=copy_results) for individual in pop]

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
        unique_results = [self.simulate(individual, copy_result=copy_results) for individual in unique_vectors]
        return [unique_results[idx] for idx in inverse]

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
