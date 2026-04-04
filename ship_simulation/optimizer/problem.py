"""将船舶仿真场景封装为标准多目标优化问题。

这个文件是 `ship_simulation` 主线里最重要的连接层之一。上游的场景生成器负责
构造会遇场景、起终点和目标船初始状态；下游的优化算法只关心一个黑盒问题：

1. 给定决策向量 `[x1, y1, v1, x2, y2, v2, ...]`
2. 返回三目标值 `[fuel, time, risk]`

本模块负责把这两者连接起来，并把运动学仿真、燃油计算、碰撞风险评估和终点
惩罚统一组织成一个可重复调用的接口。
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
    """保存单个候选解的完整评估结果。

    优化器通常只需要 `objectives`，但在调试、画图、写报告时，还需要轨迹、风险
    分解、终点到达情况等额外信息，因此这里统一返回完整结果对象。
    """

    objectives: np.ndarray
    own_trajectory: Trajectory
    target_trajectories: List[Trajectory]
    risk: RiskBreakdown
    reached_goal: bool
    terminal_distance: float


class ShipTrajectoryProblem:
    """船舶多目标轨迹规划问题。

    该类面向优化算法提供统一接口：

    - `evaluate(x)`：评估单个解
    - `evaluate_population(X)`：批量评估
    - `describe()`：返回变量维度、边界、初始猜测等元数据

    目标函数当前固定为三目标最小化：

    1. 燃油消耗
    2. 航行总时间
    3. 综合碰撞风险
    """

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig):
        self.scenario = scenario
        self.config = config

        # 在问题对象内部一次性组装完整仿真链路，避免外部调用者了解底层细节。
        self.environment = EnvironmentField(config.environment)
        self.own_ship_model = NomotoShip(config.ship, config.simulation, self.environment)
        self.target_ship_model = NomotoShip(config.ship, config.simulation, self.environment)
        self.risk_model = ShipDomainRiskModel(config.ship, config.domain)
        self.fuel_model = FuelConsumptionModel(config.environment)

        # 决策变量采用固定长度编码，每个中间航路点包含三项：
        # x 坐标、y 坐标、该段目标航速。
        self.n_waypoints = config.num_intermediate_waypoints
        self.n_var = self.n_waypoints * 3
        self.n_obj = 3
        self.var_bounds = self._build_bounds()

    def _build_bounds(self) -> np.ndarray:
        """构造决策变量上下界矩阵。

        返回形状为 `(n_var, 2)` 的数组。每一行对应一个变量的 `[lower, upper]`。
        """

        xmin, xmax, ymin, ymax = self.scenario.area
        vmin, vmax = self.config.speed_bounds
        bounds: List[Tuple[float, float]] = []

        for _ in range(self.n_waypoints):
            bounds.extend([(xmin, xmax), (ymin, ymax), (vmin, vmax)])

        return np.asarray(bounds, dtype=float)

    def initial_guess(self) -> np.ndarray:
        """构造一条直线初始轨迹。

        该初始解不是必须的，但在多数进化优化场景下有三个用途：

        1. 给算法一个稳定的可行起点
        2. 作为随机搜索之外的 baseline
        3. 作为调试和可视化时的参考轨迹
        """

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
        """把扁平决策向量解码为航路点序列和速度序列。

        输入向量只包含中间航路点，终点不是优化变量，而是由场景直接给定。为方便
        仿真，本函数会自动把场景终点追加到航路点序列尾部。
        """

        vector = np.asarray(decision_vector, dtype=float)
        if vector.size != self.n_var:
            raise ValueError(f"Expected decision vector of length {self.n_var}, got {vector.size}.")

        waypoints: List[np.ndarray] = []
        speeds: List[float] = []
        for idx in range(self.n_waypoints):
            base = idx * 3
            waypoints.append(np.array([vector[base], vector[base + 1]], dtype=float))
            speeds.append(float(vector[base + 2]))

        # 终点固定由场景定义，不参与优化编码。
        waypoints.append(np.asarray(self.scenario.own_ship.goal, dtype=float))

        # 最后一段沿用上一段的目标航速；如果没有中间点，则回退到平均速度。
        speeds.append(speeds[-1] if speeds else float(np.mean(self.config.speed_bounds)))
        return waypoints, speeds

    def simulate(self, decision_vector: Sequence[float]) -> EvaluationResult:
        """执行一次完整仿真并返回评估结果。

        这里的处理策略是“软边界 + 惩罚”而不是“越界即报错”：

        - 先把决策向量裁剪到合法范围内
        - 再把越界量转换为惩罚项加到目标函数中

        这样做对进化算法更友好，因为候选解在早期阶段经常会落到边界外。
        """

        vector = np.asarray(decision_vector, dtype=float)
        clipped = np.clip(vector, self.var_bounds[:, 0], self.var_bounds[:, 1])
        bounds_penalty = float(np.linalg.norm(vector - clipped, ord=1) * self.config.penalty_out_of_bounds)
        waypoints, speeds = self.decode(clipped)

        # 本船走“跟踪航路点 + Nomoto 一阶转向响应”的仿真。
        own_trajectory = self.own_ship_model.simulate_route(
            initial_state=self.scenario.own_ship.initial_state,
            waypoints=waypoints,
            segment_speeds=speeds,
        )

        # 目标船当前采用恒向恒速预测，这是 MVP 阶段的简化建模。
        target_trajectories = [
            self.target_ship_model.simulate_constant_velocity(target.initial_state)
            for target in self.scenario.target_ships
        ]

        # 风险评估与终点惩罚共同构成第三目标。
        # 其中终点惩罚的作用是避免“没到终点却看起来时间更短”的伪优解。
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
        """返回单个候选解的三目标值。"""

        return self.simulate(decision_vector).objectives

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """批量评估候选解种群。

        当前实现仍是串行循环，这是 MVP 阶段为了保持逻辑直观而做的取舍。
        后续如果接大规模实验，可以在这里替换为并行评估或向量化积分接口。
        """

        pop = np.atleast_2d(np.asarray(population, dtype=float))
        return np.vstack([self.evaluate(individual) for individual in pop])

    def describe(self) -> Dict[str, object]:
        """返回优化器最常用的问题元数据。"""

        return {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "var_bounds": self.var_bounds.copy(),
            "initial_guess": self.initial_guess(),
            "scenario_name": self.scenario.name,
        }
