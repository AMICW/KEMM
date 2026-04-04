"""ship_simulation/visualization/report_plots.py

本文件负责生成完整实验报告所需的静态图表。

和 `animator.py` 的区别：
- `animator.py` 面向单次交互式演示
- `report_plots.py` 面向实验归档，输出可直接插入论文或汇报材料的 PNG 图
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.scenario.encounter import EncounterScenario


@dataclass
class ExperimentSeries:
    """单个场景下某个优化器的结果封装。"""

    label: str
    result: EvaluationResult
    color: str
    history: Sequence[dict[str, float]] | None = None
    pareto_objectives: np.ndarray | None = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_trajectory_comparison(
    output_path: Path,
    scenario: EncounterScenario,
    series_list: Iterable[ExperimentSeries],
) -> None:
    """保存同一场景下不同优化器的轨迹对比图。"""

    _ensure_parent(output_path)
    series_list = list(series_list)
    if not series_list:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    xmin, xmax, ymin, ymax = scenario.area
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{scenario.name}: trajectory comparison")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.25)

    ax.scatter(
        scenario.own_ship.initial_state.x,
        scenario.own_ship.initial_state.y,
        marker="^",
        color="tab:blue",
        s=80,
        label="Own Start",
    )
    if scenario.own_ship.goal is not None:
        ax.scatter(
            scenario.own_ship.goal[0],
            scenario.own_ship.goal[1],
            marker="*",
            color="gold",
            s=150,
            label="Own Goal",
        )

    for target in scenario.target_ships:
        ax.scatter(
            target.initial_state.x,
            target.initial_state.y,
            marker="s",
            color=target.color,
            s=60,
            label=f"{target.name} Start",
        )

    for idx, target in enumerate(scenario.target_ships):
        style = "--" if idx == 0 else ":"
        # 用恒速目标船的实际仿真轨迹，而不是只画起点。
        # 第一条结果里就已经包含了所有目标船轨迹，可直接复用。
        first_series = series_list[0]
        target_traj = first_series.result.target_trajectories[idx]
        ax.plot(
            target_traj.positions[:, 0],
            target_traj.positions[:, 1],
            linestyle=style,
            linewidth=1.6,
            color=target.color,
            label=f"{target.name} path",
        )
        ax.scatter(
            target_traj.positions[-1, 0],
            target_traj.positions[-1, 1],
            marker="x",
            color=target.color,
            s=70,
        )

    for series in series_list:
        positions = series.result.own_trajectory.positions
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            linewidth=2.2,
            color=series.color,
            label=f"{series.label} own path",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            marker="o",
            color=series.color,
            s=65,
        )

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_normalized_objective_bars(
    output_path: Path,
    scenario_names: Sequence[str],
    kemm_objectives: np.ndarray,
    random_objectives: np.ndarray,
) -> None:
    """保存 KEMM 与随机基线的三目标归一化对比柱状图。"""

    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metric_names = ["Fuel", "Time", "Risk"]
    colors = {"KEMM": "tab:blue", "Random": "tab:orange"}

    for metric_idx, ax in enumerate(axes):
        pair = np.vstack([kemm_objectives[:, metric_idx], random_objectives[:, metric_idx]])
        denom = np.max(pair, axis=0)
        normalized_kemm = kemm_objectives[:, metric_idx] / (denom + 1e-9)
        normalized_random = random_objectives[:, metric_idx] / (denom + 1e-9)
        x = np.arange(len(scenario_names))
        width = 0.36
        ax.bar(x - width / 2, normalized_kemm, width=width, color=colors["KEMM"], label="KEMM")
        ax.bar(x + width / 2, normalized_random, width=width, color=colors["Random"], label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=10)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(f"Normalized {metric_names[metric_idx]}")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_risk_bars(
    output_path: Path,
    scenario_names: Sequence[str],
    kemm_risk_triplets: np.ndarray,
    random_risk_triplets: np.ndarray,
) -> None:
    """保存风险细分指标对比图。"""

    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metric_names = ["Max Risk", "Mean Risk", "Intrusion Time [s]"]
    colors = {"KEMM": "tab:blue", "Random": "tab:orange"}

    for metric_idx, ax in enumerate(axes):
        x = np.arange(len(scenario_names))
        width = 0.36
        ax.bar(x - width / 2, kemm_risk_triplets[:, metric_idx], width=width, color=colors["KEMM"], label="KEMM")
        ax.bar(x + width / 2, random_risk_triplets[:, metric_idx], width=width, color=colors["Random"], label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=10)
        ax.set_title(metric_names[metric_idx])
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_convergence_plot(output_path: Path, scenario_name: str, history: Sequence[dict[str, float]]) -> None:
    """保存 KEMM 收敛曲线。"""

    _ensure_parent(output_path)
    if not history:
        return

    generations = np.array([row["generation"] for row in history], dtype=float)
    best_fuel = np.array([row["best_fuel"] for row in history], dtype=float)
    best_time = np.array([row["best_time"] for row in history], dtype=float)
    best_risk = np.array([row["best_risk"] for row in history], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(generations, best_fuel, label="Best Fuel", color="tab:blue")
    ax.plot(generations, best_time, label="Best Time", color="tab:green")
    ax.plot(generations, best_risk, label="Best Risk", color="tab:red")
    ax.set_title(f"{scenario_name}: KEMM convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pareto_scatter(output_path: Path, scenario_name: str, pareto_objectives: np.ndarray) -> None:
    """保存 Fuel-Time 平面上的 Pareto 散点图，颜色映射 Risk。"""

    _ensure_parent(output_path)
    if pareto_objectives is None or len(pareto_objectives) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter = ax.scatter(
        pareto_objectives[:, 0],
        pareto_objectives[:, 1],
        c=pareto_objectives[:, 2],
        cmap="viridis",
        s=28,
        alpha=0.85,
    )
    ax.set_title(f"{scenario_name}: KEMM Pareto front")
    ax.set_xlabel("Fuel")
    ax.set_ylabel("Time")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Risk")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
