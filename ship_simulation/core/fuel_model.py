"""ship_simulation/core/fuel_model.py

本文件负责燃油消耗目标的近似建模。

这里没有使用真实主机工况图或高精度阻力模型，而是采用一个物理意义清晰的代理模型：
- 燃油消耗随速度立方增长
- 转向越剧烈，附加操纵损耗越大
- 海流和风越强，等效推进负担略有增加

这种做法的目标不是得到真实吨油值，而是构造一个合理、连续、可比较的优化目标。
"""

from __future__ import annotations

import numpy as np

from ship_simulation.config import EnvironmentConfig
from ship_simulation.core.ship_model import Trajectory


class FuelConsumptionModel:
    """简化推进功率/燃油代理模型。"""

    def __init__(self, environment_config: EnvironmentConfig):
        self.environment_config = environment_config

        # 下面三个系数决定燃油模型的标度
        # 后续如果有真实船型数据，可以替换成更接近实船的经验参数
        self.base_coeff = 0.045
        self.turn_coeff = 18.0
        self.env_coeff = 0.04

    def fuel_rate(self, speed_mps: float, yaw_rate: float) -> float:
        """计算瞬时燃油消耗率。"""

        # 环境越强，等效推进负荷越高
        env_penalty = 1.0 + self.env_coeff * (
            self.environment_config.current_speed
            + self.environment_config.wind_drift_coefficient * self.environment_config.wind_speed
        )

        # 转向越激烈，舵效损失和操纵负担越大
        maneuver_penalty = 1.0 + self.turn_coeff * abs(yaw_rate)

        # 速度立方规律是很多推进功率近似模型中的常见简化
        return self.base_coeff * (max(speed_mps, 0.1) ** 3) * env_penalty * maneuver_penalty

    def integrate(self, trajectory: Trajectory) -> float:
        """对整段轨迹做数值积分，得到总燃油消耗。"""

        if len(trajectory.times) < 2:
            return 0.0
        dt = np.diff(trajectory.times)
        # 使用分段常值近似，在每个时间步上累积燃油
        rates = np.array(
            [self.fuel_rate(speed, yaw_rate) for speed, yaw_rate in zip(trajectory.speeds[:-1], trajectory.yaw_rates[:-1])],
            dtype=float,
        )
        return float(np.sum(rates * dt))
