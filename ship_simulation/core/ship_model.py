"""船舶运动学/动力学模型。"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import List, Sequence

import numpy as np

from ship_simulation.config import ShipConfig, SimulationConfig
from ship_simulation.core.environment import EnvironmentField


def wrap_to_pi(angle: float) -> float:
    """将角度约束到 [-pi, pi]。"""

    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class ShipState:
    """二维平面下的船舶连续状态。"""

    x: float
    y: float
    heading: float
    speed: float
    yaw_rate: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Trajectory:
    """轨迹时序数据。"""

    times: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    speeds: np.ndarray
    yaw_rates: np.ndarray
    commanded_yaw_rates: np.ndarray
    drift_vectors: np.ndarray
    waypoint_indices: np.ndarray
    reached_goal: bool = False
    terminal_distance: float = 0.0

    def final_state(self) -> ShipState:
        """返回轨迹末端状态。"""

        return ShipState(
            x=float(self.positions[-1, 0]),
            y=float(self.positions[-1, 1]),
            heading=float(self.headings[-1]),
            speed=float(self.speeds[-1]),
            yaw_rate=float(self.yaw_rates[-1]),
        )


class NomotoShip:
    """带速度滞后和航路点跟踪能力的增强版 Nomoto 一阶船模。"""

    def __init__(self, config: ShipConfig, sim_config: SimulationConfig, environment: EnvironmentField):
        self.config = config
        self.sim_config = sim_config
        self.environment = environment

    def simulate_route(
        self,
        initial_state: ShipState,
        waypoints: Sequence[np.ndarray],
        segment_speeds: Sequence[float],
        horizon: float | None = None,
        *,
        start_time: float = 0.0,
    ) -> Trajectory:
        """仿真本船沿航路点序列航行。"""

        if len(waypoints) == 0:
            raise ValueError("At least one waypoint is required.")
        if len(segment_speeds) != len(waypoints):
            raise ValueError("segment_speeds must match the number of waypoints.")

        dt = self.sim_config.dt
        horizon = horizon if horizon is not None else self.sim_config.horizon
        state_x = float(initial_state.x)
        state_y = float(initial_state.y)
        state_heading = float(initial_state.heading)
        state_speed = float(initial_state.speed)
        state_yaw_rate = float(initial_state.yaw_rate)

        waypoint_targets = [np.asarray(point, dtype=float) for point in waypoints]
        clipped_speeds = [float(np.clip(speed, self.config.min_speed, self.config.max_speed)) for speed in segment_speeds]
        arrival_radius = max(self.sim_config.arrival_tolerance, self.config.length * 0.4)
        steps = int(horizon / dt)
        max_samples = steps + 1
        times = np.empty(max_samples, dtype=float)
        positions = np.empty((max_samples, 2), dtype=float)
        headings = np.empty(max_samples, dtype=float)
        speeds = np.empty(max_samples, dtype=float)
        yaw_rates = np.empty(max_samples, dtype=float)
        commanded_yaw_rates = np.empty(max_samples, dtype=float)
        drift_vectors = np.empty((max_samples, 2), dtype=float)
        waypoint_indices = np.empty(max_samples, dtype=int)

        times[0] = float(start_time)
        positions[0] = np.array([state_x, state_y], dtype=float)
        headings[0] = state_heading
        speeds[0] = state_speed
        yaw_rates[0] = state_yaw_rate
        commanded_yaw_rates[0] = state_yaw_rate
        drift_vectors[0] = self.environment.drift_velocity(positions[0], start_time)
        waypoint_indices[0] = 0

        waypoint_index = 0
        count = 1
        reached_goal = False
        terminal_distance = float("inf")

        for step in range(1, steps + 1):
            time_s = start_time + step * dt
            target = waypoint_targets[waypoint_index]
            target_speed = clipped_speeds[waypoint_index]

            rel_x = float(target[0] - state_x)
            rel_y = float(target[1] - state_y)
            distance = float(np.hypot(rel_x, rel_y))
            if distance < arrival_radius and waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
                target = waypoint_targets[waypoint_index]
                target_speed = clipped_speeds[waypoint_index]
                rel_x = float(target[0] - state_x)
                rel_y = float(target[1] - state_y)
                distance = float(np.hypot(rel_x, rel_y))

            desired_heading = atan2(rel_y, rel_x)
            heading_error = wrap_to_pi(desired_heading - state_heading)
            commanded_yaw_rate = np.clip(
                self.config.heading_gain * heading_error,
                -np.deg2rad(self.config.max_commanded_yaw_rate_deg),
                np.deg2rad(self.config.max_commanded_yaw_rate_deg),
            )
            yaw_rate_dot = (
                self.config.nomoto_k * commanded_yaw_rate - state_yaw_rate
            ) / max(self.config.nomoto_t, 1e-6)
            speed_dot = (target_speed - state_speed) / max(self.config.speed_time_constant, 1e-6)
            state_heading = wrap_to_pi(state_heading + (state_yaw_rate + yaw_rate_dot * dt) * dt)
            state_speed = float(np.clip(state_speed + speed_dot * dt, self.config.min_speed, self.config.max_speed))
            state_yaw_rate = float(
                np.clip(
                    state_yaw_rate + yaw_rate_dot * dt,
                    -np.deg2rad(self.config.max_turn_rate_deg),
                    np.deg2rad(self.config.max_turn_rate_deg),
                )
            )

            body_velocity = np.array([state_speed * cos(state_heading), state_speed * sin(state_heading)], dtype=float)
            drift = self.environment.drift_velocity(np.array([state_x, state_y], dtype=float), time_s)
            ground_velocity = body_velocity + drift
            state_x = float(state_x + ground_velocity[0] * dt)
            state_y = float(state_y + ground_velocity[1] * dt)

            current_distance = float(np.linalg.norm(waypoint_targets[-1] - np.array([state_x, state_y], dtype=float)))
            times[count] = time_s
            positions[count] = np.array([state_x, state_y], dtype=float)
            headings[count] = state_heading
            speeds[count] = state_speed
            yaw_rates[count] = state_yaw_rate
            commanded_yaw_rates[count] = float(commanded_yaw_rate)
            drift_vectors[count] = drift
            waypoint_indices[count] = int(waypoint_index)
            count += 1

            if waypoint_index == len(waypoints) - 1 and current_distance < arrival_radius:
                reached_goal = True
                terminal_distance = current_distance
                break

            if waypoint_index == len(waypoints) - 1:
                terminal_distance = current_distance

        if not np.isfinite(terminal_distance):
            terminal_distance = float(np.linalg.norm(waypoint_targets[-1] - np.array([state_x, state_y], dtype=float)))

        return Trajectory(
            times=times[:count].copy(),
            positions=positions[:count].copy(),
            headings=headings[:count].copy(),
            speeds=speeds[:count].copy(),
            yaw_rates=yaw_rates[:count].copy(),
            commanded_yaw_rates=commanded_yaw_rates[:count].copy(),
            drift_vectors=drift_vectors[:count].copy(),
            waypoint_indices=waypoint_indices[:count].copy(),
            reached_goal=reached_goal,
            terminal_distance=float(terminal_distance),
        )

    def simulate_constant_velocity(
        self,
        initial_state: ShipState,
        horizon: float | None = None,
        *,
        start_time: float = 0.0,
    ) -> Trajectory:
        """仿真目标船恒向恒速航行。"""

        dt = self.sim_config.dt
        horizon = horizon if horizon is not None else self.sim_config.horizon
        steps = int(horizon / dt)

        times = np.linspace(start_time, start_time + steps * dt, steps + 1)
        positions = np.zeros((steps + 1, 2), dtype=float)
        headings = np.full(steps + 1, initial_state.heading, dtype=float)
        speeds = np.full(steps + 1, initial_state.speed, dtype=float)
        yaw_rates = np.zeros(steps + 1, dtype=float)
        commanded = np.zeros(steps + 1, dtype=float)
        drifts = np.zeros((steps + 1, 2), dtype=float)
        waypoint_indices = np.zeros(steps + 1, dtype=int)
        positions[0] = initial_state.position()
        drifts[0] = self.environment.drift_velocity(positions[0], start_time)

        velocity_body = np.array(
            [initial_state.speed * cos(initial_state.heading), initial_state.speed * sin(initial_state.heading)],
            dtype=float,
        )
        pos_x = float(positions[0, 0])
        pos_y = float(positions[0, 1])
        for idx in range(1, steps + 1):
            drift = self.environment.drift_velocity(np.array([pos_x, pos_y], dtype=float), times[idx])
            drifts[idx] = drift
            pos_x = float(pos_x + (velocity_body[0] + drift[0]) * dt)
            pos_y = float(pos_y + (velocity_body[1] + drift[1]) * dt)
            positions[idx] = np.array([pos_x, pos_y], dtype=float)

        return Trajectory(
            times=times,
            positions=positions,
            headings=headings,
            speeds=speeds,
            yaw_rates=yaw_rates,
            commanded_yaw_rates=commanded,
            drift_vectors=drifts,
            waypoint_indices=waypoint_indices,
            reached_goal=False,
            terminal_distance=float("inf"),
        )
