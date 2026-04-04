"""ship_simulation/optimizer/problem.py

本文件负责把“船舶轨迹规划仿真”封装成一个标准多目标优化问题。

这是连接“物理仿真”和“优化算法”的关键文件：
- 上游接收场景、配置、决策变量
- 中间调用船模、风险模型、燃油模型完成评估
- 下游输出多目标值，供 Pareto 优化算法搜索

当前目标函数为：
1. 燃油消耗
2. 总航行时间
3. 碰撞风险
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """单个候选解的完整评估结果。

    优化器通常只关心 objectives，
    但做调试、分析和动画展示时，经常还需要轨迹和风险细节。
    """

    objectives: np.ndarray
    own_trajectory: Trajectory
    target_trajectories: List[Trajectory]
    risk: RiskBreakdown
    reached_goal: bool
    terminal_distance: float


class ShipTrajectoryProblem:
    """基于决策向量的船舶多目标轨迹规划问题。"""

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig):
        self.scenario = scenario
        self.config = config

        # 在问题对象内部组装完整仿真链路
        self.environment = EnvironmentField(config.environment)
        self.own_ship_model = NomotoShip(config.ship, config.simulation, self.environment)
        self.target_ship_model = NomotoShip(config.ship, config.simulation, self.environment)
        self.risk_model = ShipDomainRiskModel(config.ship, config.domain)
        self.fuel_model = FuelConsumptionModel(config.environment)

        # 决策变量采用 [x1, y1, v1, x2, y2, v2, ...] 编码
        self.n_waypoints = config.num_intermediate_waypoints
        self.n_var = self.n_waypoints * 3
        self.n_obj = 3
        self.var_bounds = self._build_bounds()

    def _build_bounds(self) -> np.ndarray:
        """构造每个决策变量的上下界。"""

        xmin, xmax, ymin, ymax = self.scenario.area
        vmin, vmax = self.config.speed_bounds
        bounds = []
        for _ in range(self.n_waypoints):
            # 每个中间航路点包含 x, y 和该段目标速度 v
            bounds.extend([(xmin, xmax), (ymin, ymax), (vmin, vmax)])
        return np.asarray(bounds, dtype=float)

    def initial_guess(self) -> np.ndarray:
        """构造一条直线初始猜测轨迹。

        这对进化算法并不是必需的，但通常有利于：
        - 提供一个可行的起点参考
        - 做 baseline 对比
        - 用于调试和可视化
        """

        start = self.scenario.own_ship.initial_state.position()
        goal = np.asarray(self.scenario.own_ship.goal, dtype=float)
        fractions = np.linspace(0.0, 1.0, self.n_waypoints + 2)[1:-1]
        vector = np.zeros(self.n_var, dtype=float)
        cruise_speed = float(np.mean(self.config.speed_bounds))
        for idx, frac in enumerate(fractions):
            wp = start + frac * (goal - start)
            base = idx * 3
            vector[base : base + 2] = wp
            vector[base + 2] = cruise_speed
        return vector

    def decode(self, decision_vector: Sequence[float]) -> Tuple[List[np.ndarray], List[float]]:
        """把扁平决策向量解码为航路点序列和速度序列。"""

        vector = np.asarray(decision_vector, dtype=float)
        if vector.size != self.n_var:
            raise ValueError(f"Expected decision vector of length {self.n_var}, got {vector.size}.")

        waypoints: List[np.ndarray] = []
        speeds: List[float] = []
        for idx in range(self.n_waypoints):
            base = idx * 3
            waypoints.append(np.array([vector[base], vector[base + 1]], dtype=float))
            speeds.append(float(vector[base + 2]))

        # 终点不是优化变量，而是由场景直接给定的固定目标点
        waypoints.append(np.asarray(self.scenario.own_ship.goal, dtype=float))

        # 最后一段终点速度沿用最后一个中间段速度
        speeds.append(speeds[-1] if speeds else float(np.mean(self.config.speed_bounds)))
        return waypoints, speeds

    def simulate(self, decision_vector: Sequence[float]) -> EvaluationResult:
        """执行一次完整仿真并计算三目标值。"""

        vector = np.asarray(decision_vector, dtype=float)

        # 为了方便和通用优化器直接对接，这里不硬性报错越界，
        # 而是先裁剪到合法范围，再把越界量转成惩罚项加入目标值
        clipped = np.clip(vector, self.var_bounds[:, 0], self.var_bounds[:, 1])
        bounds_penalty = float(np.linalg.norm(vector - clipped, ord=1) * self.config.penalty_out_of_bounds)
        waypoints, speeds = self.decode(clipped)

        # 仿真本船轨迹
        own_trajectory = self.own_ship_model.simulate_route(
            initial_state=self.scenario.own_ship.initial_state,
            waypoints=waypoints,
            segment_speeds=speeds,
        )

        # 目标船默认采用恒向恒速预测
        target_trajectories = [
            self.target_ship_model.simulate_constant_velocity(target.initial_state)
            for target in self.scenario.target_ships
        ]

        # 计算三个真实物理语义目标
        risk = self.risk_model.evaluate(own_trajectory, target_trajectories)
        terminal_distance = float(own_trajectory.terminal_distance)
        terminal_fuel_penalty = terminal_distance * self.config.terminal_fuel_penalty_per_meter
        terminal_time_penalty = terminal_distance * self.config.terminal_time_penalty_per_meter
        terminal_risk_penalty = terminal_distance * self.config.terminal_risk_penalty_per_meter

        fuel = self.fuel_model.integrate(own_trajectory) + terminal_fuel_penalty + bounds_penalty
        total_time = float(own_trajectory.times[-1]) + terminal_time_penalty + bounds_penalty
        collision_risk = (
            float(0.7 * risk.max_risk + 0.3 * risk.mean_risk)
            + terminal_risk_penalty
            + bounds_penalty
        )

        objectives = np.array([fuel, total_time, collision_risk], dtype=float)
        return EvaluationResult(
            objectives=objectives,
            own_trajectory=own_trajectory,
            target_trajectories=target_trajectories,
            risk=risk,
            reached_goal=own_trajectory.reached_goal,
            terminal_distance=terminal_distance,
        )

    def evaluate(self, decision_vector: Sequence[float]) -> np.ndarray:
        """返回单个候选解的目标值向量。"""

        return self.simulate(decision_vector).objectives

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """批量评估多个候选解。

        这是和进化算法对接时最常用的入口。
        """

        pop = np.atleast_2d(np.asarray(population, dtype=float))
        return np.vstack([self.evaluate(individual) for individual in pop])

    def describe(self) -> Dict[str, object]:
        """返回优化器接入时常用的元数据。"""

        return {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "var_bounds": self.var_bounds.copy(),
            "initial_guess": self.initial_guess(),
            "scenario_name": self.scenario.name,
        }
