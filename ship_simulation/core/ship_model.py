"""ship_simulation/core/ship_model.py

本文件负责船舶运动学/动力学仿真的核心实现。

当前 MVP 的建模策略：
1. 本船采用 Nomoto 一阶模型近似航向响应
2. 速度采用一阶滞后逼近到目标速度
3. 路径表示为“航路点 + 每段目标速度”
4. 目标船采用恒向恒速模型，便于快速生成多船会遇场景

这是一个面向算法验证的简化版本，重点是：
- 接口清晰
- 计算稳定
- 易于替换成 3 自由度 MMG
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import List, Sequence

import numpy as np

from ship_simulation.config import ShipConfig, SimulationConfig
from ship_simulation.core.environment import EnvironmentField


def wrap_to_pi(angle: float) -> float:
    """将角度约束到 [-pi, pi]，避免航向误差跳变。"""

    return (angle + pi) % (2.0 * pi) - pi


@dataclass
class ShipState:
    """船舶连续状态。

    用于表示某一时刻的二维位置、航向、速度和角速度。
    """

    x: float
    y: float
    heading: float
    speed: float
    yaw_rate: float = 0.0

    def position(self) -> np.ndarray:
        """以 numpy 向量形式返回当前位置。"""

        return np.array([self.x, self.y], dtype=float)


@dataclass
class Trajectory:
    """轨迹时序数据。

    一个轨迹对象保存整段仿真过程中每个采样时刻的状态历史。
    后续风险评估、燃油积分、动画展示都基于它完成。
    """

    times: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    speeds: np.ndarray
    yaw_rates: np.ndarray
    reached_goal: bool = False
    terminal_distance: float = 0.0


class NomotoShip:
    """带航路点跟踪能力的 Nomoto 一阶船模。"""

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
    ) -> Trajectory:
        """仿真本船沿给定航路点序列航行。

        输入：
        - initial_state: 初始状态
        - waypoints: 航路点序列，最后一个点通常是终点
        - segment_speeds: 每一段对应的目标速度

        输出：
        - 整段航行的轨迹时间序列
        """

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

        times: List[float] = [0.0]
        positions: List[np.ndarray] = [state.position()]
        headings: List[float] = [state.heading]
        speeds: List[float] = [state.speed]
        yaw_rates: List[float] = [state.yaw_rate]

        # waypoint_index 指向当前正在追踪的目标航路点
        waypoint_index = 0

        # 到达半径用于判定“当前航路点是否完成”
        # 这里用船长相关量来定义，避免尺度过小导致在点附近振荡
        arrival_radius = max(self.sim_config.arrival_tolerance, self.config.length * 0.4)
        steps = int(horizon / dt)
        reached_goal = False
        terminal_distance = float("inf")

        for step in range(1, steps + 1):
            target = np.asarray(waypoints[waypoint_index], dtype=float)
            target_speed = float(np.clip(segment_speeds[waypoint_index], self.config.min_speed, self.config.max_speed))

            rel = target - state.position()
            distance = float(np.linalg.norm(rel))
            # 如果到达当前航路点，就切换到下一个航路点
            if distance < arrival_radius and waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
                target = np.asarray(waypoints[waypoint_index], dtype=float)
                target_speed = float(np.clip(segment_speeds[waypoint_index], self.config.min_speed, self.config.max_speed))
                rel = target - state.position()
                distance = float(np.linalg.norm(rel))

            # 航迹跟踪：根据当前位置和目标航路点计算期望航向
            desired_heading = atan2(rel[1], rel[0])
            heading_error = wrap_to_pi(desired_heading - state.heading)

            # 先用比例控制器给出期望角速度，再用一阶响应逼近。
            # 旧实现把 heading_error 和 nomoto_t 同时用作两次缩放，转向响应过弱，
            # 导致即使是直线基线也很难稳定到达终点。
            commanded_yaw_rate = np.clip(
                self.config.heading_gain * heading_error,
                -np.deg2rad(self.config.max_turn_rate_deg),
                np.deg2rad(self.config.max_turn_rate_deg),
            )
            yaw_rate_dot = (
                self.config.nomoto_k * commanded_yaw_rate - state.yaw_rate
            ) / max(self.config.nomoto_t, 1e-6)
            speed_dot = (target_speed - state.speed) / max(self.config.speed_time_constant, 1e-6)

            state.yaw_rate += yaw_rate_dot * dt
            state.heading = wrap_to_pi(state.heading + state.yaw_rate * dt)
            state.speed += speed_dot * dt
            state.speed = float(np.clip(state.speed, self.config.min_speed, self.config.max_speed))

            # 船体坐标下的推进速度转换到世界坐标系
            body_velocity = np.array([state.speed * cos(state.heading), state.speed * sin(state.heading)], dtype=float)
            drift = self.environment.drift_velocity(state.position(), step * dt)
            ground_velocity = body_velocity + drift
            state.x += ground_velocity[0] * dt
            state.y += ground_velocity[1] * dt

            current_distance = float(np.linalg.norm(np.asarray(waypoints[-1], dtype=float) - state.position()))

            times.append(step * dt)
            positions.append(state.position())
            headings.append(state.heading)
            speeds.append(state.speed)
            yaw_rates.append(state.yaw_rate)

            # 已经到达最后终点时提前结束仿真
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
            reached_goal=reached_goal,
            terminal_distance=float(terminal_distance),
        )

    def simulate_constant_velocity(self, initial_state: ShipState, horizon: float | None = None) -> Trajectory:
        """仿真目标船恒向恒速航行。

        这是经典会遇场景中最常用的基线模型，便于分析本船规避行为。
        """

        dt = self.sim_config.dt
        horizon = horizon if horizon is not None else self.sim_config.horizon
        steps = int(horizon / dt)

        times = np.linspace(0.0, steps * dt, steps + 1)
        positions = np.zeros((steps + 1, 2), dtype=float)
        headings = np.full(steps + 1, initial_state.heading, dtype=float)
        speeds = np.full(steps + 1, initial_state.speed, dtype=float)
        yaw_rates = np.zeros(steps + 1, dtype=float)
        positions[0] = initial_state.position()

        # 目标船保持固定船首向和固定推进速度
        velocity_body = np.array(
            [initial_state.speed * cos(initial_state.heading), initial_state.speed * sin(initial_state.heading)],
            dtype=float,
        )
        for idx in range(1, steps + 1):
            drift = self.environment.drift_velocity(positions[idx - 1], times[idx])
            positions[idx] = positions[idx - 1] + (velocity_body + drift) * dt

        return Trajectory(
            times=times,
            positions=positions,
            headings=headings,
            speeds=speeds,
            yaw_rates=yaw_rates,
            reached_goal=False,
            terminal_distance=float("inf"),
        )
