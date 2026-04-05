"""ship 轨迹动画。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from ship_simulation.config import ProblemConfig
from ship_simulation.optimizer.episode import PlanningEpisodeResult
from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.scenario.encounter import EncounterScenario


@dataclass
class AnimationBundle:
    figure: plt.Figure
    animation: FuncAnimation


class TrajectoryAnimator:
    """基于 matplotlib 的轨迹动画器。"""

    def __init__(self, scenario: EncounterScenario, config: ProblemConfig):
        self.scenario = scenario
        self.config = config
        self._last_bundle: AnimationBundle | None = None

    def _domain_patch(self, position: np.ndarray, heading: float, color: str) -> Polygon:
        forward = self.config.domain.forward_factor * self.config.ship.length
        aft = self.config.domain.aft_factor * self.config.ship.length
        starboard = self.config.domain.starboard_factor * self.config.ship.beam
        port = self.config.domain.port_factor * self.config.ship.beam
        angles = np.linspace(0.0, 2.0 * np.pi, 120)
        points = []
        for angle in angles:
            longitudinal = forward if np.cos(angle) >= 0.0 else aft
            lateral = starboard if np.sin(angle) <= 0.0 else port
            body = np.array([longitudinal * np.cos(angle), lateral * np.sin(angle)], dtype=float)
            rot = np.array(
                [
                    [np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)],
                ],
                dtype=float,
            )
            points.append(position + rot @ body)
        return Polygon(np.asarray(points), closed=True, fill=False, linestyle="--", linewidth=1.0, edgecolor=color, alpha=0.5)

    def create_animation(self, result: EvaluationResult) -> AnimationBundle:
        fig, ax = plt.subplots(figsize=(10, 6))
        xmin, xmax, ymin, ymax = self.scenario.area
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{self.scenario.name} Scenario")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.25)

        own_line, = ax.plot([], [], color=self.scenario.own_ship.color, linewidth=2.0, label=self.scenario.own_ship.name)
        own_marker, = ax.plot([], [], "o", color=self.scenario.own_ship.color)
        target_lines = []
        target_markers = []
        for vessel in self.scenario.target_ships:
            line, = ax.plot([], [], linewidth=1.8, label=vessel.name, color=vessel.color)
            marker, = ax.plot([], [], "s", color=vessel.color)
            target_lines.append(line)
            target_markers.append(marker)

        own_domain = self._domain_patch(result.own_trajectory.positions[0], result.own_trajectory.headings[0], self.scenario.own_ship.color)
        ax.add_patch(own_domain)
        target_domains: List[Polygon] = []
        for vessel, traj in zip(self.scenario.target_ships, result.target_trajectories):
            patch = self._domain_patch(traj.positions[0], traj.headings[0], vessel.color)
            target_domains.append(patch)
            ax.add_patch(patch)

        ax.scatter(self.scenario.own_ship.initial_state.x, self.scenario.own_ship.initial_state.y, marker="^", color=self.scenario.own_ship.color, s=80)
        if self.scenario.own_ship.goal is not None:
            ax.scatter(self.scenario.own_ship.goal[0], self.scenario.own_ship.goal[1], marker="*", color="gold", s=140)
        ax.legend(loc="upper right")

        def update(frame: int):
            own_positions = result.own_trajectory.positions[: frame + 1]
            own_line.set_data(own_positions[:, 0], own_positions[:, 1])
            own_marker.set_data([own_positions[-1, 0]], [own_positions[-1, 1]])
            own_domain.set_xy(self._domain_patch(own_positions[-1], result.own_trajectory.headings[frame], self.scenario.own_ship.color).get_xy())
            artists = [own_line, own_marker, own_domain]
            for idx, traj in enumerate(result.target_trajectories):
                sample_count = min(frame + 1, len(traj.positions))
                positions = traj.positions[:sample_count]
                target_lines[idx].set_data(positions[:, 0], positions[:, 1])
                target_markers[idx].set_data([positions[-1, 0]], [positions[-1, 1]])
                target_domains[idx].set_xy(self._domain_patch(positions[-1], traj.headings[sample_count - 1], self.scenario.target_ships[idx].color).get_xy())
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
        self._last_bundle = self.create_animation(result)
        plt.show()

    def show_episode(self, episode: PlanningEpisodeResult) -> None:
        self.show(episode.final_evaluation)
