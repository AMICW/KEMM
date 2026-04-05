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
        state = ShipState(
            x=initial_state.x,
            y=initial_state.y,
            heading=initial_state.heading,
            speed=initial_state.speed,
            yaw_rate=initial_state.yaw_rate,
        )

        times: List[float] = [start_time]
        positions: List[np.ndarray] = [state.position()]
        headings: List[float] = [state.heading]
        speeds: List[float] = [state.speed]
        yaw_rates: List[float] = [state.yaw_rate]
        commanded_yaw_rates: List[float] = [state.yaw_rate]
        drift_vectors: List[np.ndarray] = [self.environment.drift_velocity(state.position(), start_time)]
        waypoint_indices: List[int] = [0]

        waypoint_index = 0
        arrival_radius = max(self.sim_config.arrival_tolerance, self.config.length * 0.4)
        steps = int(horizon / dt)
        reached_goal = False
        terminal_distance = float("inf")

        for step in range(1, steps + 1):
            time_s = start_time + step * dt
            target = np.asarray(waypoints[waypoint_index], dtype=float)
            target_speed = float(np.clip(segment_speeds[waypoint_index], self.config.min_speed, self.config.max_speed))

            rel = target - state.position()
            distance = float(np.linalg.norm(rel))
            if distance < arrival_radius and waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
                target = np.asarray(waypoints[waypoint_index], dtype=float)
                target_speed = float(np.clip(segment_speeds[waypoint_index], self.config.min_speed, self.config.max_speed))
                rel = target - state.position()
                distance = float(np.linalg.norm(rel))

            desired_heading = atan2(rel[1], rel[0])
            heading_error = wrap_to_pi(desired_heading - state.heading)
            commanded_yaw_rate = np.clip(
                self.config.heading_gain * heading_error,
                -np.deg2rad(self.config.max_commanded_yaw_rate_deg),
                np.deg2rad(self.config.max_commanded_yaw_rate_deg),
            )
            yaw_rate_dot = (
                self.config.nomoto_k * commanded_yaw_rate - state.yaw_rate
            ) / max(self.config.nomoto_t, 1e-6)
            speed_dot = (target_speed - state.speed) / max(self.config.speed_time_constant, 1e-6)

            state = ShipState(
                x=state.x,
                y=state.y,
                heading=wrap_to_pi(state.heading + (state.yaw_rate + yaw_rate_dot * dt) * dt),
                speed=float(np.clip(state.speed + speed_dot * dt, self.config.min_speed, self.config.max_speed)),
                yaw_rate=float(np.clip(
                    state.yaw_rate + yaw_rate_dot * dt,
                    -np.deg2rad(self.config.max_turn_rate_deg),
                    np.deg2rad(self.config.max_turn_rate_deg),
                )),
            )

            body_velocity = np.array([state.speed * cos(state.heading), state.speed * sin(state.heading)], dtype=float)
            drift = self.environment.drift_velocity(state.position(), time_s)
            ground_velocity = body_velocity + drift
            state = ShipState(
                x=float(state.x + ground_velocity[0] * dt),
                y=float(state.y + ground_velocity[1] * dt),
                heading=state.heading,
                speed=state.speed,
                yaw_rate=state.yaw_rate,
            )

            current_distance = float(np.linalg.norm(np.asarray(waypoints[-1], dtype=float) - state.position()))
            times.append(time_s)
            positions.append(state.position())
            headings.append(state.heading)
            speeds.append(state.speed)
            yaw_rates.append(state.yaw_rate)
            commanded_yaw_rates.append(float(commanded_yaw_rate))
            drift_vectors.append(drift)
            waypoint_indices.append(int(waypoint_index))

            if waypoint_index == len(waypoints) - 1 and current_distance < arrival_radius:
                reached_goal = True
                terminal_distance = current_distance
                break

            if waypoint_index == len(waypoints) - 1:
                terminal_distance = current_distance

        if not np.isfinite(terminal_distance):
            terminal_distance = float(np.linalg.norm(np.asarray(waypoints[-1], dtype=float) - state.position()))

        return Trajectory(
            times=np.asarray(times, dtype=float),
            positions=np.asarray(positions, dtype=float),
            headings=np.asarray(headings, dtype=float),
            speeds=np.asarray(speeds, dtype=float),
            yaw_rates=np.asarray(yaw_rates, dtype=float),
            commanded_yaw_rates=np.asarray(commanded_yaw_rates, dtype=float),
            drift_vectors=np.asarray(drift_vectors, dtype=float),
            waypoint_indices=np.asarray(waypoint_indices, dtype=int),
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
        for idx in range(1, steps + 1):
            drift = self.environment.drift_velocity(positions[idx - 1], times[idx])
            drifts[idx] = drift
            positions[idx] = positions[idx - 1] + (velocity_body + drift) * dt

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
