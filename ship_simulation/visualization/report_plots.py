"""生成 ship 主线报告所需的静态图表。

这个文件和 `animator.py` 的定位不同：

- `animator.py` 面向单次演示，强调动态播放
- `report_plots.py` 面向实验归档，强调稳定输出可直接插入报告的 PNG 图表

所有函数都尽量接收结构化结果对象，而不是直接依赖某个求解器的内部字段，目的是
让 KEMM、随机搜索以及未来的新算法都能复用同一套可视化接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from reporting_config import ShipPlotConfig, plot_style_context
from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.scenario.encounter import EncounterScenario


@dataclass
class ExperimentSeries:
    """封装单个算法在单个场景下的展示数据。

    这个 dataclass 是可视化层和求解器之间的最小接口约定：

    - `label`：图例名称
    - `result`：最佳解或代表解的完整评估结果
    - `color`：绘图颜色
    - `history`：可选的收敛历史
    - `pareto_objectives`：可选的 Pareto 前沿目标值
    """

    label: str
    result: EvaluationResult
    color: str
    history: Sequence[dict[str, float]] | None = None
    pareto_objectives: np.ndarray | None = None



def _ensure_parent(path: Path) -> None:
    """确保输出目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_plot_config(plot_config: ShipPlotConfig | None) -> ShipPlotConfig:
    return plot_config or ShipPlotConfig()


def _styled_plot(func):
    """为 ship 图表统一注入论文风格上下文。"""

    @wraps(func)
    def wrapper(*args, plot_config: ShipPlotConfig | None = None, **kwargs):
        cfg = _resolve_plot_config(plot_config)
        with plot_style_context(cfg.style):
            return func(*args, plot_config=cfg, **kwargs)

    return wrapper



@_styled_plot
def save_trajectory_comparison(
    output_path: Path,
    scenario: EncounterScenario,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存同一场景下不同算法的轨迹对比图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return

    fig, ax = plt.subplots(figsize=cfg.trajectory_figsize)
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
        color=cfg.own_ship_color,
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

    # 目标船轨迹对不同优化器是共享的，直接复用第一条结果中保存的目标船仿真轨迹。
    for idx, target in enumerate(scenario.target_ships):
        style = "--" if idx == 0 else ":"
        first_series = series_list[0]
        target_traj = first_series.result.target_trajectories[idx]
        ax.plot(
            target_traj.positions[:, 0],
            target_traj.positions[:, 1],
            linestyle=style,
            linewidth=cfg.target_width,
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
            linewidth=cfg.trajectory_width,
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
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_normalized_objective_bars(
    output_path: Path,
    scenario_names: Sequence[str],
    kemm_objectives: np.ndarray,
    random_objectives: np.ndarray,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存 KEMM 与随机 baseline 的三目标归一化柱状对比图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    fig, axes = plt.subplots(1, 3, figsize=cfg.comparison_figsize)
    metric_names = ["Fuel", "Time", "Risk"]
    colors = {"KEMM": cfg.own_ship_color, "Random": cfg.baseline_color}

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
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_risk_bars(
    output_path: Path,
    scenario_names: Sequence[str],
    kemm_risk_triplets: np.ndarray,
    random_risk_triplets: np.ndarray,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存风险细分指标对比图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    fig, axes = plt.subplots(1, 3, figsize=cfg.comparison_figsize)
    metric_names = ["Max Risk", "Mean Risk", "Intrusion Time [s]"]
    colors = {"KEMM": cfg.own_ship_color, "Random": cfg.baseline_color}

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
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_convergence_plot(
    output_path: Path,
    scenario_name: str,
    history: Sequence[dict[str, float]],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存 KEMM 在单个场景下的收敛曲线。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    if not history:
        return

    generations = np.array([row["generation"] for row in history], dtype=float)
    best_fuel = np.array([row["best_fuel"] for row in history], dtype=float)
    best_time = np.array([row["best_time"] for row in history], dtype=float)
    best_risk = np.array([row["best_risk"] for row in history], dtype=float)

    fig, ax = plt.subplots(figsize=cfg.convergence_figsize)
    ax.plot(generations, best_fuel, label="Best Fuel", color=cfg.own_ship_color)
    ax.plot(generations, best_time, label="Best Time", color="tab:green")
    ax.plot(generations, best_risk, label="Best Risk", color=cfg.risk_threshold_color)
    ax.set_title(f"{scenario_name}: KEMM convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_pareto_scatter(
    output_path: Path,
    scenario_name: str,
    pareto_objectives: np.ndarray,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存 Fuel-Time 平面上的 Pareto 散点图，并用颜色映射风险值。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    if pareto_objectives is None or len(pareto_objectives) == 0:
        return

    fig, ax = plt.subplots(figsize=cfg.pareto_figsize)
    scatter = ax.scatter(
        pareto_objectives[:, 0],
        pareto_objectives[:, 1],
        c=pareto_objectives[:, 2],
        cmap=cfg.pareto_cmap,
        s=cfg.style.scatter_size,
        alpha=0.85,
    )
    ax.set_title(f"{scenario_name}: KEMM Pareto front")
    ax.set_xlabel("Fuel")
    ax.set_ylabel("Time")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Risk")
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_risk_time_series(
    output_path: Path,
    scenario_name: str,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存逐时风险曲线。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return

    fig, ax = plt.subplots(figsize=cfg.time_series_figsize)
    for series in series_list:
        risk_series = np.asarray(series.result.risk.risk_series, dtype=float)
        if risk_series.size == 0:
            continue
        ax.plot(
            series.result.own_trajectory.times[: risk_series.size],
            risk_series,
            linewidth=cfg.trajectory_width,
            color=series.color,
            label=series.label,
        )

    # 风险值大于 1 通常表示已明显侵入高风险域，因此这里额外画出参考线。
    ax.axhline(
        cfg.risk_threshold,
        linestyle="--",
        color=cfg.risk_threshold_color,
        linewidth=1.2,
        alpha=0.8,
        label="Domain Boundary",
    )
    ax.set_title(f"{scenario_name}: risk time series")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Risk")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_speed_profiles(
    output_path: Path,
    scenario_name: str,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """保存本船速度剖面对比图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return

    fig, ax = plt.subplots(figsize=cfg.time_series_figsize)
    for series in series_list:
        ax.plot(
            series.result.own_trajectory.times,
            series.result.own_trajectory.speeds,
            linewidth=cfg.trajectory_width,
            color=series.color,
            label=series.label,
        )

    ax.set_title(f"{scenario_name}: speed profile")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)



@_styled_plot
def save_summary_dashboard(
    output_path: Path,
    scenario: EncounterScenario,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """生成一页式汇总图板。

    这张图会把最常用的四类信息放在一页内：

    1. 空间轨迹对比
    2. 风险时间历程
    3. 速度剖面
    4. 三目标归一化柱状图
    """

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return

    fig, axes = plt.subplots(2, 2, figsize=cfg.dashboard_figsize)
    ax_traj, ax_risk, ax_speed, ax_obj = axes.flatten()

    xmin, xmax, ymin, ymax = scenario.area
    ax_traj.set_xlim(xmin, xmax)
    ax_traj.set_ylim(ymin, ymax)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_title(f"{scenario.name}: trajectory comparison")
    ax_traj.set_xlabel("X [m]")
    ax_traj.set_ylabel("Y [m]")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.scatter(
        scenario.own_ship.initial_state.x,
        scenario.own_ship.initial_state.y,
        marker="^",
        color=cfg.own_ship_color,
        s=80,
        label="Own Start",
    )
    if scenario.own_ship.goal is not None:
        ax_traj.scatter(
            scenario.own_ship.goal[0],
            scenario.own_ship.goal[1],
            marker="*",
            color="gold",
            s=150,
            label="Own Goal",
        )
    for idx, target in enumerate(scenario.target_ships):
        first_series = series_list[0]
        target_traj = first_series.result.target_trajectories[idx]
        ax_traj.plot(
            target_traj.positions[:, 0],
            target_traj.positions[:, 1],
            linestyle="--",
            linewidth=cfg.target_width,
            color=target.color,
            label=f"{target.name} path",
        )
    for series in series_list:
        positions = series.result.own_trajectory.positions
        ax_traj.plot(
            positions[:, 0],
            positions[:, 1],
            linewidth=cfg.trajectory_width,
            color=series.color,
            label=series.label,
        )
    ax_traj.legend(loc="best")

    for series in series_list:
        risk_series = np.asarray(series.result.risk.risk_series, dtype=float)
        if risk_series.size == 0:
            continue
        ax_risk.plot(
            series.result.own_trajectory.times[: risk_series.size],
            risk_series,
            linewidth=cfg.trajectory_width,
            color=series.color,
            label=series.label,
        )
    ax_risk.axhline(
        cfg.risk_threshold,
        linestyle="--",
        color=cfg.risk_threshold_color,
        linewidth=1.2,
        alpha=0.8,
    )
    ax_risk.set_title("Risk Time Series")
    ax_risk.set_xlabel("Time [s]")
    ax_risk.set_ylabel("Risk")
    ax_risk.grid(True, alpha=0.25)
    ax_risk.legend(loc="best")

    for series in series_list:
        ax_speed.plot(
            series.result.own_trajectory.times,
            series.result.own_trajectory.speeds,
            linewidth=cfg.trajectory_width,
            color=series.color,
            label=series.label,
        )
    ax_speed.set_title("Speed Profile")
    ax_speed.set_xlabel("Time [s]")
    ax_speed.set_ylabel("Speed [m/s]")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(loc="best")

    objective_labels = ["Fuel", "Time", "Risk"]
    objective_matrix = np.vstack([series.result.objectives for series in series_list])
    denom = np.max(objective_matrix, axis=0) + 1e-9
    x = np.arange(len(objective_labels))
    width = 0.8 / max(len(series_list), 1)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(series_list))
    for offset, series in zip(offsets, series_list):
        normalized = series.result.objectives / denom
        ax_obj.bar(x + offset, normalized, width=width, color=series.color, alpha=cfg.style.bar_alpha, label=series.label)
    ax_obj.set_xticks(x)
    ax_obj.set_xticklabels(objective_labels)
    ax_obj.set_ylim(0.0, 1.15)
    ax_obj.set_title("Normalized Objectives")
    ax_obj.set_ylabel("Relative value")
    ax_obj.grid(True, axis="y", alpha=0.25)
    ax_obj.legend(loc="best")

    fig.suptitle(f"{scenario.name}: simulation summary dashboard", fontsize=cfg.style.title_size)
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.style.dpi)
    plt.close(fig)
