"""ship_simulation/visualization/animator.py

本文件负责轨迹可视化与动画展示。

可视化目标：
1. 直观看到本船和目标船的运动过程
2. 显示本船和目标船船舶域
3. 用于验证优化结果是否真的绕开了高风险区域

当前基于 matplotlib 实现，优先满足“易用”和“可调试”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

from ship_simulation.config import ProblemConfig
from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.scenario.encounter import EncounterScenario


@dataclass
class AnimationBundle:
    """动画对象返回容器。"""

    figure: plt.Figure
    animation: FuncAnimation


class TrajectoryAnimator:
    """基于 matplotlib 的轨迹动画器。"""

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig):
        self.scenario = scenario
        self.config = config
        self._last_bundle: AnimationBundle | None = None

    def _domain_patch(self, position: np.ndarray, heading: float, color: str) -> Ellipse:
        """根据船舶域配置生成一个椭圆补丁对象。"""

        width = self.config.domain.starboard_factor * self.config.ship.beam * 2.0
        height = self.config.domain.forward_factor * self.config.ship.length * 2.0
        return Ellipse(
            xy=position,
            width=height,
            height=width,
            angle=np.rad2deg(heading),
            fill=False,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.5,
        )

    def create_animation(self, result: EvaluationResult) -> AnimationBundle:
        """根据仿真结果创建动画对象。"""

        fig, ax = plt.subplots(figsize=(10, 6))
        xmin, xmax, ymin, ymax = self.scenario.area
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{self.scenario.name} Scenario")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.25)

        # 本船轨迹线与当前位置标记
        own_line, = ax.plot([], [], color=self.scenario.own_ship.color, linewidth=2.0, label=self.scenario.own_ship.name)
        own_marker, = ax.plot([], [], "o", color=self.scenario.own_ship.color)
        target_lines = []
        target_markers = []
        for vessel in self.scenario.target_ships:
            # 每艘目标船都有独立轨迹线和当前位置标记
            line, = ax.plot([], [], linewidth=1.8, label=vessel.name, color=vessel.color)
            marker, = ax.plot([], [], "s", color=vessel.color)
            target_lines.append(line)
            target_markers.append(marker)

        # 船舶域以椭圆形式显示，随船一起移动和旋转
        own_domain = self._domain_patch(result.own_trajectory.positions[0], result.own_trajectory.headings[0], self.scenario.own_ship.color)
        ax.add_patch(own_domain)
        target_domains: List[Ellipse] = []
        for vessel, traj in zip(self.scenario.target_ships, result.target_trajectories):
            patch = self._domain_patch(traj.positions[0], traj.headings[0], vessel.color)
            target_domains.append(patch)
            ax.add_patch(patch)

        ax.scatter(
            self.scenario.own_ship.initial_state.x,
            self.scenario.own_ship.initial_state.y,
            marker="^",
            color=self.scenario.own_ship.color,
            s=80,
            label="Own Start",
        )
        ax.scatter(
            self.scenario.own_ship.goal[0],
            self.scenario.own_ship.goal[1],
            marker="*",
            color="gold",
            s=140,
            label="Own Goal",
        )
        ax.legend(loc="upper right")

        def update(frame: int):
            """动画逐帧刷新函数。"""

            own_positions = result.own_trajectory.positions[: frame + 1]
            own_line.set_data(own_positions[:, 0], own_positions[:, 1])
            own_marker.set_data([own_positions[-1, 0]], [own_positions[-1, 1]])
            own_domain.center = own_positions[-1]
            own_domain.angle = np.rad2deg(result.own_trajectory.headings[frame])

            artists = [own_line, own_marker, own_domain]
            for idx, traj in enumerate(result.target_trajectories):
                sample_count = min(frame + 1, len(traj.positions))
                positions = traj.positions[:sample_count]
                target_lines[idx].set_data(positions[:, 0], positions[:, 1])
                target_markers[idx].set_data([positions[-1, 0]], [positions[-1, 1]])
                target_domains[idx].center = positions[-1]
                target_domains[idx].angle = np.rad2deg(traj.headings[sample_count - 1])
                artists.extend([target_lines[idx], target_markers[idx], target_domains[idx]])
            return artists

        animation = FuncAnimation(
            fig,
            update,
            frames=len(result.own_trajectory.times),
            interval=60,
            blit=False,
            repeat=False,
        )
        return AnimationBundle(figure=fig, animation=animation)

    def show(self, result: EvaluationResult) -> None:
        """直接显示动画窗口。"""

        self._last_bundle = self.create_animation(result)
        plt.show()
