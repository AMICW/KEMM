"""组合碰撞风险模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, exp, sin
from typing import Iterable, Sequence

import numpy as np

from ship_simulation.config import DomainConfig, ProblemConfig, ShipConfig
from ship_simulation.core.environment import EnvironmentField
from ship_simulation.core.ship_model import Trajectory
from ship_simulation.scenario.encounter import CircularObstacle, PolygonObstacle


@dataclass
class RiskBreakdown:
    """碰撞风险分解结果。"""

    max_risk: float
    mean_risk: float
    intrusion_time: float
    min_clearance: float
    min_dcpa: float
    min_tcpa: float
    min_static_clearance: float
    min_ship_distance: float
    risk_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    domain_risk_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    dcpa_risk_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    obstacle_risk_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    environment_risk_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    clearance_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    static_clearance_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    ship_distance_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    dcpa_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    tcpa_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    colreg_scale_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


class ShipDomainRiskModel:
    """船舶域 + DCPA/TCPA + 障碍/环境惩罚的组合风险模型。"""

    def __init__(self, ship_config: ShipConfig, domain_config: DomainConfig, problem_config: ProblemConfig):
        self.ship_config = ship_config
        self.domain_config = domain_config
        self.problem_config = problem_config

    def _domain_axes(self, rel_body: np.ndarray) -> np.ndarray:
        longitudinal = self.domain_config.forward_factor if rel_body[0] >= 0.0 else self.domain_config.aft_factor
        lateral = self.domain_config.starboard_factor if rel_body[1] <= 0.0 else self.domain_config.port_factor
        return np.array(
            [longitudinal * self.ship_config.length, lateral * self.ship_config.beam],
            dtype=float,
        )

    @staticmethod
    def _rotate_to_body(relative_xy: np.ndarray, heading: float) -> np.ndarray:
        c = cos(heading)
        s = sin(heading)
        rotation = np.array([[c, s], [-s, c]], dtype=float)
        return rotation @ relative_xy

    def instantaneous_domain_risk(self, own_position: np.ndarray, own_heading: float, target_position: np.ndarray) -> float:
        rel_world = np.asarray(target_position, dtype=float) - np.asarray(own_position, dtype=float)
        rel_body = self._rotate_to_body(rel_world, own_heading)
        axes = self._domain_axes(rel_body)
        normalized_distance = np.sqrt(np.sum((rel_body / (axes * self.domain_config.soft_margin)) ** 2))
        if normalized_distance <= 1.0:
            return 1.0 + (1.0 - normalized_distance)
        return float(np.exp(-2.2 * (normalized_distance - 1.0)))

    @staticmethod
    def _trajectory_velocity(traj: Trajectory, idx: int) -> np.ndarray:
        if len(traj.positions) < 2:
            return np.zeros(2, dtype=float)
        if idx <= 0:
            delta = traj.positions[1] - traj.positions[0]
            dt = traj.times[1] - traj.times[0]
        else:
            prev_idx = min(idx, len(traj.positions) - 1)
            delta = traj.positions[prev_idx] - traj.positions[prev_idx - 1]
            dt = traj.times[prev_idx] - traj.times[prev_idx - 1]
        return delta / max(float(dt), 1e-6)

    def _dcpa_tcpa(self, own_traj: Trajectory, target_traj: Trajectory, idx: int) -> tuple[float, float]:
        own_pos = own_traj.positions[idx]
        target_pos = target_traj.positions[idx]
        rel_pos = target_pos - own_pos
        rel_vel = self._trajectory_velocity(target_traj, idx) - self._trajectory_velocity(own_traj, idx)
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        if rel_speed_sq < 1e-9:
            return float(np.linalg.norm(rel_pos)), 0.0
        tcpa = float(-np.dot(rel_pos, rel_vel) / rel_speed_sq)
        tcpa = max(tcpa, 0.0)
        dcpa_vec = rel_pos + tcpa * rel_vel
        dcpa = float(np.linalg.norm(dcpa_vec))
        return dcpa, tcpa

    @staticmethod
    def _distance_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        if np.allclose(ab, 0.0):
            return float(np.linalg.norm(point - a))
        t = float(np.dot(point - a, ab) / np.dot(ab, ab))
        t = min(1.0, max(0.0, t))
        projection = a + t * ab
        return float(np.linalg.norm(point - projection))

    def _point_in_polygon(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        x, y = point
        inside = False
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / max(y2 - y1, 1e-9) + x1)
            if intersects:
                inside = not inside
        return inside

    def _obstacle_clearance(
        self,
        own_position: np.ndarray,
        obstacles: Sequence[CircularObstacle | PolygonObstacle],
    ) -> float:
        if not obstacles:
            return float("inf")
        clearance = float("inf")
        for obstacle in obstacles:
            if isinstance(obstacle, CircularObstacle):
                distance = float(np.linalg.norm(own_position - np.asarray(obstacle.center, dtype=float)) - obstacle.radius)
            else:
                vertices = np.asarray(obstacle.vertices, dtype=float)
                edge_distances = [
                    self._distance_to_segment(own_position, vertices[i], vertices[(i + 1) % len(vertices)])
                    for i in range(len(vertices))
                ]
                distance = min(edge_distances)
                if self._point_in_polygon(own_position, vertices):
                    distance = -distance
            clearance = min(clearance, distance)
        return clearance

    def _clearance_risk(self, clearance: float) -> float:
        if not np.isfinite(clearance):
            return 0.0
        if clearance <= 0.0:
            return 2.0
        return float(np.exp(-clearance / max(self.problem_config.safety_clearance, 1.0)))

    def _dcpa_risk(self, dcpa: float, tcpa: float) -> float:
        safety = max(self.problem_config.safety_clearance, 1.0)
        spatial = np.exp(-dcpa / safety)
        temporal = np.exp(-tcpa / max(self.problem_config.tcpa_decay_seconds, 1.0))
        return float(spatial * temporal)

    @staticmethod
    def _colreg_scale(role: str) -> float:
        mapping = {
            "head_on": 1.18,
            "crossing_stand_on": 1.12,
            "crossing_give_way": 1.22,
            "overtaking": 1.06,
        }
        return float(mapping.get(role, 1.0))

    def evaluate(
        self,
        own_traj: Trajectory,
        target_trajectories: Iterable[Trajectory],
        *,
        static_obstacles: Sequence[CircularObstacle | PolygonObstacle] = (),
        environment: EnvironmentField | None = None,
        colreg_roles: dict[str, str] | None = None,
        target_names: Sequence[str] | None = None,
    ) -> RiskBreakdown:
        targets = list(target_trajectories)
        if not targets and not static_obstacles:
            zero = np.zeros(len(own_traj.times), dtype=float)
            return RiskBreakdown(
                max_risk=0.0,
                mean_risk=0.0,
                intrusion_time=0.0,
                min_clearance=float("inf"),
                min_dcpa=float("inf"),
                min_tcpa=float("inf"),
                min_static_clearance=float("inf"),
                min_ship_distance=float("inf"),
                risk_series=zero,
                domain_risk_series=zero,
                dcpa_risk_series=zero,
                obstacle_risk_series=zero,
                environment_risk_series=zero,
                clearance_series=zero,
                static_clearance_series=zero,
                ship_distance_series=zero,
                dcpa_series=zero,
                tcpa_series=zero,
                colreg_scale_series=zero,
            )

        colreg_roles = colreg_roles or {}
        target_names = list(target_names or [f"target_{idx}" for idx in range(len(targets))])
        count = len(own_traj.times)
        risk = np.zeros(count, dtype=float)
        domain_risk = np.zeros(count, dtype=float)
        dcpa_risk = np.zeros(count, dtype=float)
        obstacle_risk = np.zeros(count, dtype=float)
        environment_risk = np.zeros(count, dtype=float)
        clearance_series = np.full(count, np.inf, dtype=float)
        ship_distance_series = np.full(count, np.inf, dtype=float)
        static_clearance_series = np.full(count, np.inf, dtype=float)
        dcpa_series = np.full(count, np.inf, dtype=float)
        tcpa_series = np.full(count, np.inf, dtype=float)
        colreg_scale_series = np.ones(count, dtype=float)
        dt = own_traj.times[1] - own_traj.times[0] if len(own_traj.times) > 1 else 0.0

        for idx in range(count):
            obstacle_clearance = self._obstacle_clearance(own_traj.positions[idx], static_obstacles)
            static_clearance_series[idx] = obstacle_clearance
            obstacle_risk[idx] = self._clearance_risk(obstacle_clearance)
            if environment is not None:
                environment_risk[idx] = min(2.0, environment.scalar_risk_at(own_traj.positions[idx], own_traj.times[idx]))

            step_domain = 0.0
            step_dcpa_risk = 0.0
            step_ship_distance = float("inf")
            step_dcpa = float("inf")
            step_tcpa = float("inf")
            step_scale = 1.0
            for target_idx, target in enumerate(targets):
                sample_idx = min(idx, len(target.times) - 1)
                ship_distance = float(np.linalg.norm(target.positions[sample_idx] - own_traj.positions[idx]))
                step_ship_distance = min(step_ship_distance, ship_distance)
                domain_value = self.instantaneous_domain_risk(
                    own_position=own_traj.positions[idx],
                    own_heading=own_traj.headings[idx],
                    target_position=target.positions[sample_idx],
                )
                dcpa, tcpa = self._dcpa_tcpa(own_traj, target, sample_idx)
                dcpa_value = self._dcpa_risk(dcpa, tcpa)
                role = colreg_roles.get(target_names[target_idx], "")
                scale = self._colreg_scale(role)
                if domain_value * scale > step_domain * step_scale:
                    step_domain = domain_value
                    step_scale = scale
                if dcpa_value > step_dcpa_risk:
                    step_dcpa_risk = dcpa_value
                step_dcpa = min(step_dcpa, dcpa)
                step_tcpa = min(step_tcpa, tcpa)

            domain_risk[idx] = step_domain
            dcpa_risk[idx] = step_dcpa_risk
            ship_distance_series[idx] = step_ship_distance
            clearance_series[idx] = min(obstacle_clearance, step_ship_distance)
            dcpa_series[idx] = step_dcpa
            tcpa_series[idx] = step_tcpa
            colreg_scale_series[idx] = step_scale
            combined = (
                self.problem_config.domain_risk_weight * step_domain
                + self.problem_config.dcpa_risk_weight * step_dcpa_risk
                + self.problem_config.obstacle_risk_weight * obstacle_risk[idx]
                + self.problem_config.environment_risk_weight * environment_risk[idx]
            ) * step_scale
            risk[idx] = combined

        intrusion_time = float(np.sum(risk >= 1.0) * dt)
        min_clearance = float(np.min(clearance_series)) if clearance_series.size else float("inf")
        min_static_clearance = float(np.min(static_clearance_series)) if static_clearance_series.size else float("inf")
        min_ship_distance = float(np.min(ship_distance_series)) if np.isfinite(ship_distance_series).any() else float("inf")
        min_dcpa = float(np.min(dcpa_series)) if np.isfinite(dcpa_series).any() else float("inf")
        min_tcpa = float(np.min(tcpa_series)) if np.isfinite(tcpa_series).any() else float("inf")
        return RiskBreakdown(
            max_risk=float(np.max(risk)) if risk.size else 0.0,
            mean_risk=float(np.mean(risk)) if risk.size else 0.0,
            intrusion_time=intrusion_time,
            min_clearance=min_clearance,
            min_dcpa=min_dcpa,
            min_tcpa=min_tcpa,
            min_static_clearance=min_static_clearance,
            min_ship_distance=min_ship_distance,
            risk_series=risk,
            domain_risk_series=domain_risk,
            dcpa_risk_series=dcpa_risk,
            obstacle_risk_series=obstacle_risk,
            environment_risk_series=environment_risk,
            clearance_series=clearance_series,
            static_clearance_series=static_clearance_series,
            ship_distance_series=ship_distance_series,
            dcpa_series=dcpa_series,
            tcpa_series=tcpa_series,
            colreg_scale_series=colreg_scale_series,
        )
