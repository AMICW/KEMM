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


@dataclass(frozen=True)
class CircularObstacleDescriptor:
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class PolygonObstacleDescriptor:
    vertices: np.ndarray
    edge_starts: np.ndarray
    edge_ends: np.ndarray
    bounding_box: np.ndarray


StaticObstacleDescriptor = CircularObstacleDescriptor | PolygonObstacleDescriptor


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

    def _domain_axes_series(self, rel_body: np.ndarray) -> np.ndarray:
        rel = np.asarray(rel_body, dtype=float)
        longitudinal = np.where(rel[:, 0] >= 0.0, self.domain_config.forward_factor, self.domain_config.aft_factor)
        lateral = np.where(rel[:, 1] <= 0.0, self.domain_config.starboard_factor, self.domain_config.port_factor)
        return np.column_stack(
            [
                longitudinal * self.ship_config.length,
                lateral * self.ship_config.beam,
            ]
        ).astype(float)

    @staticmethod
    def _rotate_to_body(relative_xy: np.ndarray, heading: float) -> np.ndarray:
        c = cos(heading)
        s = sin(heading)
        rotation = np.array([[c, s], [-s, c]], dtype=float)
        return rotation @ relative_xy

    @staticmethod
    def _rotate_to_body_series(relative_xy: np.ndarray, headings: np.ndarray) -> np.ndarray:
        rel = np.asarray(relative_xy, dtype=float)
        heading_arr = np.asarray(headings, dtype=float)
        c = np.cos(heading_arr)
        s = np.sin(heading_arr)
        x_body = c * rel[:, 0] + s * rel[:, 1]
        y_body = -s * rel[:, 0] + c * rel[:, 1]
        return np.column_stack([x_body, y_body]).astype(float)

    def instantaneous_domain_risk(self, own_position: np.ndarray, own_heading: float, target_position: np.ndarray) -> float:
        rel_world = np.asarray(target_position, dtype=float) - np.asarray(own_position, dtype=float)
        rel_body = self._rotate_to_body(rel_world, own_heading)
        axes = self._domain_axes(rel_body)
        normalized_distance = np.sqrt(np.sum((rel_body / (axes * self.domain_config.soft_margin)) ** 2))
        if normalized_distance <= 1.0:
            return 1.0 + (1.0 - normalized_distance)
        return float(np.exp(-2.2 * (normalized_distance - 1.0)))

    def instantaneous_domain_risk_series(
        self,
        own_positions: np.ndarray,
        own_headings: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        rel_world = np.asarray(target_positions, dtype=float) - np.asarray(own_positions, dtype=float)
        rel_body = self._rotate_to_body_series(rel_world, np.asarray(own_headings, dtype=float))
        axes = self._domain_axes_series(rel_body)
        normalized_distance = np.sqrt(
            np.sum((rel_body / np.maximum(axes * self.domain_config.soft_margin, 1e-9)) ** 2, axis=1)
        )
        inside = normalized_distance <= 1.0
        values = np.exp(-2.2 * (normalized_distance - 1.0))
        values[inside] = 1.0 + (1.0 - normalized_distance[inside])
        return values.astype(float)

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

    @staticmethod
    def _trajectory_velocities(traj: Trajectory) -> np.ndarray:
        if len(traj.positions) < 2:
            return np.zeros((len(traj.positions), 2), dtype=float)
        deltas = np.diff(traj.positions, axis=0)
        dts = np.diff(traj.times)
        safe_dts = np.maximum(dts, 1e-6)
        velocities = np.zeros((len(traj.positions), 2), dtype=float)
        velocities[0] = deltas[0] / safe_dts[0]
        velocities[1:] = deltas / safe_dts[:, None]
        return velocities

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

    def _dcpa_tcpa_series(
        self,
        own_positions: np.ndarray,
        own_velocities: np.ndarray,
        target_positions: np.ndarray,
        target_velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel_pos = np.asarray(target_positions, dtype=float) - np.asarray(own_positions, dtype=float)
        rel_vel = np.asarray(target_velocities, dtype=float) - np.asarray(own_velocities, dtype=float)
        rel_speed_sq = np.sum(rel_vel * rel_vel, axis=1)
        tcpa = np.zeros(len(rel_pos), dtype=float)
        moving = rel_speed_sq >= 1e-9
        tcpa[moving] = np.maximum(-np.sum(rel_pos[moving] * rel_vel[moving], axis=1) / rel_speed_sq[moving], 0.0)
        dcpa_vec = rel_pos + tcpa[:, None] * rel_vel
        dcpa = np.linalg.norm(dcpa_vec, axis=1)
        dcpa[~moving] = np.linalg.norm(rel_pos[~moving], axis=1)
        return dcpa.astype(float), tcpa.astype(float)

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

    def _point_in_polygon_series(self, points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        verts = np.asarray(vertices, dtype=float)
        if len(pts) == 0 or len(verts) == 0:
            return np.zeros(len(pts), dtype=bool)
        x = pts[:, 0][:, None]
        y = pts[:, 1][:, None]
        x1 = verts[:, 0][None, :]
        y1 = verts[:, 1][None, :]
        rolled = np.roll(verts, -1, axis=0)
        x2 = rolled[:, 0][None, :]
        y2 = rolled[:, 1][None, :]
        denom = np.where(np.abs(y2 - y1) <= 1e-9, 1e-9, y2 - y1)
        intersects = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / denom + x1)
        return (np.sum(intersects, axis=1) % 2 == 1)

    def _obstacle_clearance(
        self,
        own_position: np.ndarray,
        obstacles: Sequence[CircularObstacle | PolygonObstacle] = (),
        descriptors: Sequence[StaticObstacleDescriptor] | None = None,
    ) -> float:
        clearance = self._obstacle_clearance_series(
            np.asarray(own_position, dtype=float).reshape(1, 2),
            obstacles=obstacles,
            descriptors=descriptors,
        )
        return float(clearance[0]) if clearance.size else float("inf")

    @staticmethod
    def _distance_to_segments_series(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        seg_starts = np.asarray(starts, dtype=float)
        seg_ends = np.asarray(ends, dtype=float)
        if len(pts) == 0 or len(seg_starts) == 0:
            return np.full(len(pts), np.inf, dtype=float)
        ab = seg_ends - seg_starts
        denom = np.sum(ab * ab, axis=1)
        safe_denom = np.where(denom <= 1e-12, 1.0, denom)
        ap = pts[:, None, :] - seg_starts[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / safe_denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        projection = seg_starts[None, :, :] + t[:, :, None] * ab[None, :, :]
        distances = np.linalg.norm(pts[:, None, :] - projection, axis=2)
        zero_segments = denom <= 1e-12
        if np.any(zero_segments):
            distances[:, zero_segments] = np.linalg.norm(
                pts[:, None, :] - seg_starts[None, zero_segments, :],
                axis=2,
            )
        return np.min(distances, axis=1)

    def _obstacle_clearance_series(
        self,
        own_positions: np.ndarray,
        *,
        obstacles: Sequence[CircularObstacle | PolygonObstacle] = (),
        descriptors: Sequence[StaticObstacleDescriptor] | None = None,
    ) -> np.ndarray:
        positions = np.asarray(own_positions, dtype=float)
        if len(positions) == 0:
            return np.zeros(0, dtype=float)
        descriptor_list = list(descriptors or [])
        if not descriptor_list and obstacles:
            for obstacle in obstacles:
                if isinstance(obstacle, CircularObstacle):
                    descriptor_list.append(
                        CircularObstacleDescriptor(center=np.asarray(obstacle.center, dtype=float), radius=float(obstacle.radius))
                    )
                else:
                    vertices = np.asarray(obstacle.vertices, dtype=float)
                    descriptor_list.append(
                        PolygonObstacleDescriptor(
                            vertices=vertices,
                            edge_starts=vertices,
                            edge_ends=np.roll(vertices, -1, axis=0),
                            bounding_box=np.array(
                                [
                                    float(np.min(vertices[:, 0])),
                                    float(np.max(vertices[:, 0])),
                                    float(np.min(vertices[:, 1])),
                                    float(np.max(vertices[:, 1])),
                                ],
                                dtype=float,
                            ),
                        )
                    )
        if not descriptor_list:
            return np.full(len(positions), np.inf, dtype=float)
        clearance = np.full(len(positions), np.inf, dtype=float)
        for obstacle in descriptor_list:
            if isinstance(obstacle, CircularObstacleDescriptor):
                distance = np.linalg.norm(positions - np.asarray(obstacle.center, dtype=float), axis=1) - float(obstacle.radius)
            else:
                edge_distances = self._distance_to_segments_series(positions, obstacle.edge_starts, obstacle.edge_ends)
                bbox = np.asarray(obstacle.bounding_box, dtype=float)
                candidate_mask = (
                    (positions[:, 0] >= bbox[0])
                    & (positions[:, 0] <= bbox[1])
                    & (positions[:, 1] >= bbox[2])
                    & (positions[:, 1] <= bbox[3])
                )
                inside = np.zeros(len(positions), dtype=bool)
                if np.any(candidate_mask):
                    inside[candidate_mask] = self._point_in_polygon_series(positions[candidate_mask], obstacle.vertices)
                distance = edge_distances
                distance[inside] = -distance[inside]
            clearance = np.minimum(clearance, distance)
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

    def _dcpa_risk_series(self, dcpa: np.ndarray, tcpa: np.ndarray) -> np.ndarray:
        safety = max(self.problem_config.safety_clearance, 1.0)
        spatial = np.exp(-np.asarray(dcpa, dtype=float) / safety)
        temporal = np.exp(-np.asarray(tcpa, dtype=float) / max(self.problem_config.tcpa_decay_seconds, 1.0))
        return (spatial * temporal).astype(float)

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
        static_obstacle_descriptors: Sequence[StaticObstacleDescriptor] | None = None,
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
        own_positions = np.asarray(own_traj.positions, dtype=float)
        own_headings = np.asarray(own_traj.headings, dtype=float)
        own_velocities = self._trajectory_velocities(own_traj)

        static_clearance_series = self._obstacle_clearance_series(
            own_positions,
            obstacles=static_obstacles,
            descriptors=static_obstacle_descriptors,
        )
        obstacle_risk = np.asarray([self._clearance_risk(float(clearance)) for clearance in static_clearance_series], dtype=float)
        if environment is not None:
            environment_risk = np.minimum(2.0, environment.scalar_risk_series(own_positions, own_traj.times))

        best_scaled_domain = np.zeros(count, dtype=float)
        step_domain = np.zeros(count, dtype=float)
        step_dcpa_risk = np.zeros(count, dtype=float)
        step_ship_distance = np.full(count, np.inf, dtype=float)
        step_dcpa = np.full(count, np.inf, dtype=float)
        step_tcpa = np.full(count, np.inf, dtype=float)
        step_scale = np.ones(count, dtype=float)

        for target_idx, target in enumerate(targets):
            sample_indices = np.minimum(np.arange(count, dtype=int), max(len(target.times) - 1, 0))
            target_positions = np.asarray(target.positions[sample_indices], dtype=float)
            target_velocities = self._trajectory_velocities(target)[sample_indices]
            ship_distance = np.linalg.norm(target_positions - own_positions, axis=1)
            domain_value = self.instantaneous_domain_risk_series(
                own_positions=own_positions,
                own_headings=own_headings,
                target_positions=target_positions,
            )
            dcpa, tcpa = self._dcpa_tcpa_series(
                own_positions=own_positions,
                own_velocities=own_velocities,
                target_positions=target_positions,
                target_velocities=target_velocities,
            )
            dcpa_value = self._dcpa_risk_series(dcpa, tcpa)
            role = colreg_roles.get(target_names[target_idx], "")
            scale = self._colreg_scale(role)
            scaled_domain = domain_value * scale
            better_domain = scaled_domain > best_scaled_domain
            best_scaled_domain = np.where(better_domain, scaled_domain, best_scaled_domain)
            step_domain = np.where(better_domain, domain_value, step_domain)
            step_scale = np.where(better_domain, scale, step_scale)
            step_dcpa_risk = np.maximum(step_dcpa_risk, dcpa_value)
            step_ship_distance = np.minimum(step_ship_distance, ship_distance)
            step_dcpa = np.minimum(step_dcpa, dcpa)
            step_tcpa = np.minimum(step_tcpa, tcpa)

        domain_risk = step_domain
        dcpa_risk = step_dcpa_risk
        ship_distance_series = step_ship_distance
        clearance_series = np.minimum(static_clearance_series, ship_distance_series)
        dcpa_series = step_dcpa
        tcpa_series = step_tcpa
        colreg_scale_series = step_scale
        risk = (
            self.problem_config.domain_risk_weight * domain_risk
            + self.problem_config.dcpa_risk_weight * dcpa_risk
            + self.problem_config.obstacle_risk_weight * obstacle_risk
            + self.problem_config.environment_risk_weight * environment_risk
        ) * colreg_scale_series

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
