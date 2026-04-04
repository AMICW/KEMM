"""ship_simulation/core/collision_risk.py

本文件负责碰撞风险评估。

当前实现基于“船舶域（Ship Domain）”思想：
- 把本船周围定义成一个非对称椭圆安全域
- 目标船越靠近安全域中心，风险越高
- 风险不是简单 0/1，而是连续值，便于优化算法搜索

这样做的优点：
1. 比纯最近距离阈值更符合船舶避碰逻辑
2. 连续风险函数更适合多目标优化
3. 容易扩展为更复杂的藤野模型或 COLREG 规则增强模型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Iterable

import numpy as np

from ship_simulation.config import DomainConfig, ShipConfig
from ship_simulation.core.ship_model import Trajectory


@dataclass
class RiskBreakdown:
    """碰撞风险汇总结果。

    - max_risk: 全航程中最大瞬时风险
    - mean_risk: 全航程平均风险
    - intrusion_time: 进入船舶域核心区域的累计时间
    """

    max_risk: float
    mean_risk: float
    intrusion_time: float
    risk_series: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))


class ShipDomainRiskModel:
    """基于椭圆船舶域的连续碰撞风险模型。"""

    def __init__(self, ship_config: ShipConfig, domain_config: DomainConfig):
        self.ship_config = ship_config
        self.domain_config = domain_config

    def _domain_axes(self, rel_body: np.ndarray) -> np.ndarray:
        """根据目标相对方位确定船舶域长短轴。

        这里体现船舶域的方向性：
        - 前方需要更大安全余度
        - 后方安全余度较小
        - 左右舷可以设置为不同尺度
        """

        longitudinal = self.domain_config.forward_factor if rel_body[0] >= 0.0 else self.domain_config.aft_factor
        lateral = self.domain_config.starboard_factor if rel_body[1] <= 0.0 else self.domain_config.port_factor
        return np.array(
            [longitudinal * self.ship_config.length, lateral * self.ship_config.beam],
            dtype=float,
        )

    @staticmethod
    def _rotate_to_body(relative_xy: np.ndarray, heading: float) -> np.ndarray:
        """将世界坐标系中的相对位置旋转到本船体坐标系。"""

        c = cos(heading)
        s = sin(heading)
        rotation = np.array([[c, s], [-s, c]], dtype=float)
        return rotation @ relative_xy

    def instantaneous_risk(self, own_position: np.ndarray, own_heading: float, target_position: np.ndarray) -> float:
        """计算某一时刻的瞬时风险。

        计算逻辑：
        1. 先将目标船相对位置变换到本船体坐标系
        2. 再按本船当前朝向下的椭圆船舶域做归一化
        3. 若进入域内，则风险大于等于 1
        4. 若在域外，则风险按指数形式衰减
        """

        rel_world = np.asarray(target_position, dtype=float) - np.asarray(own_position, dtype=float)
        rel_body = self._rotate_to_body(rel_world, own_heading)
        axes = self._domain_axes(rel_body)
        normalized_distance = np.sqrt(np.sum((rel_body / (axes * self.domain_config.soft_margin)) ** 2))
        # 域内返回高风险；越靠近中心，风险越高
        if normalized_distance <= 1.0:
            return 1.0 + (1.0 - normalized_distance)

        # 域外采用连续衰减，避免目标函数出现大面积平坦区
        return float(np.exp(-2.2 * (normalized_distance - 1.0)))

    def evaluate(self, own_traj: Trajectory, target_trajectories: Iterable[Trajectory]) -> RiskBreakdown:
        """汇总整段航迹上的碰撞风险。"""

        targets = list(target_trajectories)
        if not targets:
            return RiskBreakdown(max_risk=0.0, mean_risk=0.0, intrusion_time=0.0)

        # 对每个仿真时刻保存“所有目标船中的最大风险”
        per_step_risk = np.zeros(len(own_traj.times), dtype=float)
        dt = own_traj.times[1] - own_traj.times[0] if len(own_traj.times) > 1 else 0.0
        for target in targets:
            sample_count = min(len(own_traj.times), len(target.times))
            for idx in range(sample_count):
                per_step_risk[idx] = max(
                    per_step_risk[idx],
                    self.instantaneous_risk(
                        own_position=own_traj.positions[idx],
                        own_heading=own_traj.headings[idx],
                        target_position=target.positions[idx],
                    ),
                )

        # intrusion_time 表示“本船进入高风险域”的累计时长
        intrusion_time = float(np.sum(per_step_risk >= 1.0) * dt)
        return RiskBreakdown(
            max_risk=float(np.max(per_step_risk)),
            mean_risk=float(np.mean(per_step_risk)),
            intrusion_time=intrusion_time,
            risk_series=per_step_risk.copy(),
        )
