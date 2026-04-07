"""ship 主线的论文风格图表输出。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from reporting_config import ShipPlotConfig, plot_style_context, save_figure_bundle
from ship_simulation.optimizer.episode import PlanningEpisodeResult
from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.scenario.encounter import CircularObstacle, EncounterScenario, PolygonObstacle

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False


@dataclass
class ExperimentSeries:
    """可视化层统一消费的数据容器。"""

    label: str
    color: str
    episode: PlanningEpisodeResult | None = None
    result: EvaluationResult | None = None
    pareto_objectives: np.ndarray | None = None
    pareto_decisions: np.ndarray | None = None
    snapshots: list[object] = field(default_factory=list)
    histories: list[list[dict[str, float]]] = field(default_factory=list)
    distribution_metrics: list[dict[str, float]] = field(default_factory=list)
    repeated_statistics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.result is None and self.episode is not None:
            self.result = self.episode.final_evaluation
        if self.episode is not None:
            if self.pareto_objectives is None:
                self.pareto_objectives = self.episode.pareto_objectives
            if self.pareto_decisions is None:
                self.pareto_decisions = self.episode.pareto_decisions
            if not self.snapshots:
                self.snapshots = list(self.episode.snapshots)
        if self.result is None:
            raise ValueError("ExperimentSeries requires either `episode` or `result`.")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _interactive_html_path(output_path: Path) -> Path:
    return output_path.with_suffix(".html")


def _plotly_html_enabled(cfg: ShipPlotConfig) -> bool:
    return bool(cfg.interactive_html and HAS_PLOTLY)


def _plotly_color(color: str) -> str:
    try:
        return str(to_hex(color))
    except ValueError:
        return color


def _finalize_figure(fig, output_path: Path, cfg: ShipPlotConfig, *, allow_interactive: bool = False) -> None:
    save_figure_bundle(
        fig,
        output_path,
        dpi=cfg.style.dpi,
        interactive_figures=bool(cfg.interactive_figures and allow_interactive),
    )
    plt.close(fig)


def _save_plotly_html(fig, output_path: Path, cfg: ShipPlotConfig) -> None:
    if not _plotly_html_enabled(cfg):
        return
    html_path = _interactive_html_path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(html_path),
        include_plotlyjs=cfg.interactive_html_include_plotlyjs,
        full_html=True,
    )


def _resolve_plot_config(plot_config: ShipPlotConfig | None) -> ShipPlotConfig:
    return plot_config or ShipPlotConfig()


def _styled_plot(func):
    @wraps(func)
    def wrapper(*args, plot_config: ShipPlotConfig | None = None, **kwargs):
        cfg = _resolve_plot_config(plot_config)
        with plot_style_context(cfg.style):
            return func(*args, plot_config=cfg, **kwargs)

    return wrapper


def _apply_plotly_scene_layout(
    fig,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    zaxis_title: str,
) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=56),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.82)",
        ),
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.92)),
        ),
    )


def _obstacle_style(obstacle, cfg: ShipPlotConfig) -> dict[str, object]:
    face = getattr(obstacle, "color", cfg.obstacle_facecolor)
    if isinstance(obstacle, CircularObstacle):
        alpha = cfg.obstacle_alpha
    elif getattr(obstacle, "kind", "") == "channel_boundary":
        alpha = cfg.boundary_alpha
    else:
        alpha = cfg.obstacle_alpha
    return {
        "facecolor": face,
        "edgecolor": cfg.obstacle_edgecolor,
        "alpha": alpha,
    }


def _draw_start_goal(ax, scenario: EncounterScenario) -> None:
    start = scenario.own_ship.initial_state.position()
    goal = np.asarray(scenario.own_ship.goal, dtype=float) if scenario.own_ship.goal is not None else None
    ax.scatter(start[0], start[1], marker="o", color="#1d4ed8", s=48, zorder=6)
    if goal is not None:
        ax.scatter(goal[0], goal[1], marker="o", color="#16a34a", s=48, zorder=6)


def _draw_traffic_initials(ax, scenario: EncounterScenario, cfg: ShipPlotConfig) -> None:
    arrow_scale = max(cfg.velocity_arrow_scale * 0.55, 25.0)
    for target in scenario.target_ships:
        state = target.initial_state
        start = np.array([state.x, state.y], dtype=float)
        ax.scatter(start[0], start[1], marker="o", color=target.color, s=46, zorder=6)
        heading_vec = np.array([np.cos(state.heading), np.sin(state.heading)], dtype=float) * arrow_scale
        ax.annotate(
            "",
            xy=(start[0] + heading_vec[0], start[1] + heading_vec[1]),
            xytext=(start[0], start[1]),
            arrowprops=dict(arrowstyle="->", lw=1.4, color=target.color, alpha=0.9),
            zorder=6,
        )


def _scenario_legend_handles(scenario: EncounterScenario) -> list[object]:
    handles: list[object] = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1d4ed8", markersize=6, label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", markersize=6, label="Goal"),
    ]
    if any(isinstance(obstacle, CircularObstacle) for obstacle in scenario.static_obstacles):
        handles.append(Patch(facecolor="#111827", edgecolor="#111827", alpha=0.9, label="Circular obstacle"))
    if any(isinstance(obstacle, PolygonObstacle) and getattr(obstacle, "kind", "") == "keep_out" for obstacle in scenario.static_obstacles):
        handles.append(Patch(facecolor="#86efac", edgecolor="#475569", alpha=0.9, label="Keep-out polygon"))
    if any(isinstance(obstacle, PolygonObstacle) and getattr(obstacle, "kind", "") == "channel_boundary" for obstacle in scenario.static_obstacles):
        handles.append(Patch(facecolor="#d1d5db", edgecolor="#475569", alpha=0.7, label="Channel boundary"))
    if scenario.target_ships:
        handles.append(Line2D([0], [0], marker="o", color="#dc2626", markerfacecolor="#dc2626", markersize=6, label="Traffic ship"))
    return handles


def _series_legend_handles(series_list: Sequence[ExperimentSeries]) -> list[object]:
    return [
        Line2D([0], [0], color=series.color, lw=2.8, label=series.label)
        for series in series_list
    ]


def _panel_caption(index: int, title: str) -> str:
    return f"({chr(ord('a') + index)}) {title}"


def _add_obstacles(ax, scenario: EncounterScenario, cfg: ShipPlotConfig) -> None:
    for obstacle in scenario.static_obstacles:
        style = _obstacle_style(obstacle, cfg)
        if isinstance(obstacle, CircularObstacle):
            patch = Circle(
                xy=np.asarray(obstacle.center, dtype=float),
                radius=float(obstacle.radius),
                **style,
            )
        else:
            patch = Polygon(
                np.asarray(obstacle.vertices, dtype=float),
                closed=True,
                **style,
            )
        ax.add_patch(patch)


def _plot_environment(ax, scenario: EncounterScenario, series: ExperimentSeries, cfg: ShipPlotConfig) -> None:
    xmin, xmax, ymin, ymax = scenario.area
    env = series.episode.steps[0].selected_evaluation if series.episode.steps else series.result
    _ = env
    field = series.episode.steps[0].scenario if series.episode.steps else scenario
    _ = field
    problem = None
    if series.episode.steps:
        problem = series.episode.steps[0].selected_evaluation
    xx, yy, zz = series.episode.steps[0].scenario.scalar_fields, None, None
    _ = xx, yy, zz


def _plot_scalar_vector_background(ax, scenario: EncounterScenario, series: ExperimentSeries, cfg: ShipPlotConfig) -> None:
    from ship_simulation.optimizer.problem import ShipTrajectoryProblem
    from ship_simulation.config import build_default_config

    problem = ShipTrajectoryProblem(scenario, build_default_config())
    xx, yy, zz = problem.environment.sample_scalar_grid(scenario.area, resolution=cfg.scalar_grid_resolution)
    ax.contourf(xx, yy, zz, levels=16, cmap=cfg.scalar_cmap, alpha=0.42)
    vx, vy, uu, vv = problem.environment.sample_vector_grid(scenario.area, resolution=cfg.vector_grid_resolution)
    ax.quiver(vx, vy, uu, vv, color=cfg.vector_color, alpha=0.45, width=0.002)


def _draw_ship_trajectories(
    ax,
    scenario: EncounterScenario,
    series_list: Sequence[ExperimentSeries],
    cfg: ShipPlotConfig,
    *,
    show_velocity_arrows: bool = False,
) -> None:
    if not series_list:
        return
    base_result = series_list[0].result
    for target_idx, target in enumerate(scenario.target_ships):
        traj = base_result.target_trajectories[target_idx]
        ax.plot(
            traj.positions[:, 0],
            traj.positions[:, 1],
            linestyle="--",
            linewidth=cfg.target_width,
            color=target.color,
            label=target.name,
        )
        if show_velocity_arrows and len(traj.positions) > 2:
            step = max(1, len(traj.positions) // 7)
            sample_idx = np.arange(0, len(traj.positions) - 1, step, dtype=int)
            delta = traj.positions[sample_idx + 1] - traj.positions[sample_idx]
            norms = np.linalg.norm(delta, axis=1, keepdims=True)
            arrows = delta / np.maximum(norms, 1e-9) * cfg.velocity_arrow_scale
            ax.quiver(
                traj.positions[sample_idx, 0],
                traj.positions[sample_idx, 1],
                arrows[:, 0],
                arrows[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=target.color,
                alpha=0.8,
                width=0.0028,
            )
    for series in series_list:
        positions = series.result.own_trajectory.positions
        ax.plot(positions[:, 0], positions[:, 1], linewidth=cfg.trajectory_width, color=series.color, label=series.label)
        ax.scatter(positions[0, 0], positions[0, 1], marker="^", color=series.color, s=90)
        ax.scatter(positions[-1, 0], positions[-1, 1], marker="o", color=series.color, s=60)
    if scenario.own_ship.goal is not None:
        ax.scatter(scenario.own_ship.goal[0], scenario.own_ship.goal[1], marker="*", color="gold", s=180, label="Goal")


def _format_spatial_axis(ax, scenario: EncounterScenario, title: str, cfg: ShipPlotConfig) -> None:
    xmin, xmax, ymin, ymax = scenario.area
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=cfg.style.grid_alpha)


def _format_gallery_axis(ax, scenario: EncounterScenario, title: str, cfg: ShipPlotConfig) -> None:
    xmin, xmax, ymin, ymax = scenario.area
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_alpha(0.75)


def _projection_front_order(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=int)
    keep: list[int] = []
    for i in range(len(points)):
        dominated = False
        for j in range(len(points)):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    if not keep:
        return np.zeros(0, dtype=int)
    keep_array = np.asarray(keep, dtype=int)
    return keep_array[np.lexsort((points[keep_array, 1], points[keep_array, 0]))]


def _bottleneck_indices(result: EvaluationResult) -> list[int]:
    risk = result.risk
    candidates: list[int] = []
    if risk.clearance_series.size:
        candidates.append(int(np.argmin(risk.clearance_series)))
    if risk.ship_distance_series.size and np.isfinite(risk.ship_distance_series).any():
        ship_idx = int(np.nanargmin(np.where(np.isfinite(risk.ship_distance_series), risk.ship_distance_series, np.nan)))
        if all(abs(ship_idx - existing) > 4 for existing in candidates):
            candidates.append(ship_idx)
    return candidates[:2]


def _draw_route_insets(fig, ax, scenario: EncounterScenario, series_list: Sequence[ExperimentSeries], cfg: ShipPlotConfig) -> None:
    if not series_list:
        return
    reference = series_list[0].result
    indices = _bottleneck_indices(reference)
    if not indices:
        return
    inset_boxes = ([0.66, 0.60, 0.28, 0.28], [0.66, 0.20, 0.28, 0.28])
    half_window = max((scenario.area[1] - scenario.area[0]) * 0.08, 420.0)
    for idx, bounds in zip(indices, inset_boxes):
        center = reference.own_trajectory.positions[min(idx, len(reference.own_trajectory.positions) - 1)]
        inset = ax.inset_axes(bounds)
        inset.patch.set_alpha(cfg.inset_zoom_alpha)
        _plot_scalar_vector_background(inset, scenario, series_list[0], cfg)
        _add_obstacles(inset, scenario, cfg)
        _draw_ship_trajectories(inset, scenario, series_list, cfg, show_velocity_arrows=False)
        inset.set_xlim(center[0] - half_window, center[0] + half_window)
        inset.set_ylim(center[1] - half_window, center[1] + half_window)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.grid(True, alpha=cfg.style.grid_alpha)
        clearance = reference.risk.clearance_series[min(idx, len(reference.risk.clearance_series) - 1)]
        inset.set_title(f"t={reference.own_trajectory.times[min(idx, len(reference.own_trajectory.times) - 1)]:.0f}s\nclr={clearance:.1f} m", fontsize=max(cfg.style.tick_size - 1, 7))
        ax.indicate_inset_zoom(inset, edgecolor=cfg.knee_color, alpha=0.65)


def _step_change_events(episode: PlanningEpisodeResult) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    for step in episode.steps:
        if not step.applied_changes:
            continue
        labels = [str(change.get("label", "Change")) for change in step.applied_changes]
        events.append((float(step.start_time), " + ".join(labels)))
    return events


def _annotate_change_events(ax, episode: PlanningEpisodeResult, cfg: ShipPlotConfig, *, with_labels: bool = False) -> None:
    events = _step_change_events(episode)
    if not events:
        return
    ymin, ymax = ax.get_ylim()
    label_height = ymax - 0.06 * (ymax - ymin)
    for idx, (time_s, label) in enumerate(events):
        ax.axvline(time_s, color=cfg.event_line_color, linestyle="--", linewidth=1.2, alpha=0.82)
        ax.axvspan(time_s, time_s + max(episode.result.own_trajectory.times[-1] * 0.01, 8.0), color=cfg.event_fill_color, alpha=0.22)
        if with_labels:
            ax.text(
                time_s,
                label_height - idx * 0.05 * (ymax - ymin),
                label,
                rotation=90,
                va="top",
                ha="left",
                color=cfg.event_line_color,
                fontsize=max(cfg.style.tick_size - 1, 8),
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.82, edgecolor=cfg.event_fill_color),
            )


def _plotly_pareto_hover_payload(episode: PlanningEpisodeResult) -> tuple[np.ndarray, str]:
    objectives = np.asarray(episode.pareto_objectives, dtype=float)
    decisions = np.asarray(episode.pareto_decisions, dtype=float) if episode.pareto_decisions is not None else np.zeros((len(objectives), 0), dtype=float)
    if objectives.size == 0:
        return np.zeros((0, 0), dtype=float), ""
    weighted = 0.4 * objectives[:, 0] + 0.25 * objectives[:, 1] + 0.35 * objectives[:, 2]
    speed_mean = np.zeros(len(objectives), dtype=float)
    waypoint_xy = np.full((len(objectives), 2), np.nan, dtype=float)
    if decisions.ndim == 2 and decisions.shape[1] >= 3:
        reshaped = decisions.reshape(len(decisions), -1, 3)
        speed_mean = np.mean(reshaped[:, :, 2], axis=1)
        waypoint_xy = reshaped[:, 0, :2]
    is_knee = np.zeros(len(objectives), dtype=int)
    if episode.knee_index is not None and 0 <= int(episode.knee_index) < len(objectives):
        is_knee[int(episode.knee_index)] = 1
    custom = np.column_stack(
        [
            np.arange(len(objectives), dtype=float),
            weighted,
            speed_mean,
            waypoint_xy[:, 0],
            waypoint_xy[:, 1],
            is_knee.astype(float),
        ]
    )
    hovertemplate = (
        "Point %{customdata[0]:.0f}<br>"
        "Fuel=%{x:.2f}<br>"
        "Time=%{y:.2f}<br>"
        "Risk=%{z:.3f}<br>"
        "Weighted=%{customdata[1]:.2f}<br>"
        "Mean speed=%{customdata[2]:.2f}<br>"
        "WP1=(%{customdata[3]:.1f}, %{customdata[4]:.1f})<br>"
        "Knee=%{customdata[5]:.0f}<extra></extra>"
    )
    return custom, hovertemplate


def _plotly_change_event_trace(episode: PlanningEpisodeResult) -> tuple[np.ndarray, list[str]]:
    if not episode.steps:
        return np.zeros((0, 4), dtype=float), []
    own = episode.result.own_trajectory
    rows: list[list[float]] = []
    labels: list[str] = []
    for step in episode.steps:
        if not step.applied_changes:
            continue
        idx = min(np.searchsorted(own.times, step.start_time), len(own.times) - 1)
        position = own.positions[idx]
        rows.append([position[0], position[1], own.times[idx], float(step.step_index)])
        labels.append(" + ".join(str(change.get("label", "Change")) for change in step.applied_changes))
    if not rows:
        return np.zeros((0, 4), dtype=float), []
    return np.asarray(rows, dtype=float), labels


def _stat_value(series: ExperimentSeries, key: str) -> tuple[float, float]:
    payload = series.repeated_statistics or {}
    return float(payload.get(key, np.nan)), float(payload.get(f"{key}_std", 0.0))


def _select_metrics(series_list: Sequence[ExperimentSeries]) -> tuple[list[str], np.ndarray]:
    metric_names = ["Fuel", "Time", "Risk", "Clearance", "Smoothness", "Control"]
    values = []
    for series in series_list:
        metrics = series.episode.analysis_metrics if series.episode is not None else series.result.analysis_metrics
        values.append(
            [
                float(series.result.objectives[0]),
                float(series.result.objectives[1]),
                float(series.result.objectives[2]),
                max(1e-6, float(metrics.get("minimum_clearance", 0.0))),
                float(metrics.get("smoothness", 0.0)),
                float(metrics.get("control_effort", 0.0)),
            ]
        )
    return metric_names, np.asarray(values, dtype=float)


def _normalized(values: np.ndarray, minimize_mask: Sequence[bool]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalized = np.zeros_like(values)
    for col, minimize in enumerate(minimize_mask):
        column = values[:, col]
        span = np.max(column) - np.min(column)
        if span <= 1e-9:
            normalized[:, col] = 1.0
        elif minimize:
            normalized[:, col] = (np.max(column) - column) / span
        else:
            normalized[:, col] = (column - np.min(column)) / span
    return normalized


def _save_spatiotemporal_html(output_path: Path, scenario: EncounterScenario, episode: PlanningEpisodeResult, cfg: ShipPlotConfig) -> None:
    if not _plotly_html_enabled(cfg):
        return
    own = episode.result.own_trajectory
    fig = go.Figure()
    own_risk = episode.result.risk.risk_series
    own_clearance = episode.result.risk.clearance_series
    own_custom = np.column_stack(
        [
            own.times,
            own.speeds,
            np.rad2deg(own.headings),
            own_risk[: len(own.times)] if own_risk.size else np.zeros(len(own.times), dtype=float),
            own_clearance[: len(own.times)] if own_clearance.size else np.zeros(len(own.times), dtype=float),
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=own.positions[:, 0],
            y=own.positions[:, 1],
            z=own.times,
            mode="lines",
            name=episode.optimizer_name,
            line=dict(color=_plotly_color(cfg.own_ship_color), width=6),
            customdata=own_custom,
            hovertemplate=(
                "Own ship<br>"
                "x=%{x:.1f} m<br>"
                "y=%{y:.1f} m<br>"
                "t=%{customdata[0]:.1f} s<br>"
                "speed=%{customdata[1]:.2f} m/s<br>"
                "heading=%{customdata[2]:.1f} deg<br>"
                "risk=%{customdata[3]:.3f}<br>"
                "clearance=%{customdata[4]:.1f} m<extra></extra>"
            ),
        )
    )
    for target_idx, target in enumerate(scenario.target_ships):
        traj = episode.result.target_trajectories[target_idx]
        target_custom = np.column_stack(
            [
                traj.times,
                traj.speeds,
                np.rad2deg(traj.headings),
            ]
        )
        fig.add_trace(
            go.Scatter3d(
                x=traj.positions[:, 0],
                y=traj.positions[:, 1],
                z=traj.times,
                mode="lines",
                name=target.name,
                line=dict(color=_plotly_color(target.color), width=4, dash="dash"),
                customdata=target_custom,
                hovertemplate=(
                    f"{target.name}<br>"
                    "x=%{x:.1f} m<br>"
                    "y=%{y:.1f} m<br>"
                    "t=%{customdata[0]:.1f} s<br>"
                    "speed=%{customdata[1]:.2f} m/s<br>"
                    "heading=%{customdata[2]:.1f} deg<extra></extra>"
                ),
            )
        )
    event_points, labels = _plotly_change_event_trace(episode)
    if event_points.size:
        fig.add_trace(
            go.Scatter3d(
                x=event_points[:, 0],
                y=event_points[:, 1],
                z=event_points[:, 2],
                mode="markers+text",
                name="Scheduled change",
                text=labels,
                textposition="top center",
                marker=dict(size=5, color=_plotly_color(cfg.event_line_color), symbol="diamond"),
                hovertemplate="Step %{customdata[0]:.0f}<br>%{text}<extra></extra>",
                customdata=event_points[:, 3:4],
            )
        )
    _apply_plotly_scene_layout(
        fig,
        title=f"{scenario.name}: spatiotemporal planning",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        zaxis_title="Time [s]",
    )
    _save_plotly_html(fig, output_path, cfg)


def _save_pareto_html(output_path: Path, scenario_name: str, episode: PlanningEpisodeResult, cfg: ShipPlotConfig) -> None:
    if not _plotly_html_enabled(cfg):
        return
    objectives = np.asarray(episode.pareto_objectives, dtype=float)
    if objectives.size == 0:
        return
    custom, hovertemplate = _plotly_pareto_hover_payload(episode)
    marker = dict(
        size=6,
        color=np.linalg.norm(objectives, axis=1),
        colorscale=cfg.pareto_cmap,
        opacity=0.85,
        colorbar=dict(title="Objective-space radius"),
        line=dict(width=0.8, color="#334155"),
    )
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=objectives[:, 0],
                y=objectives[:, 1],
                z=objectives[:, 2],
                mode="markers",
                name="Pareto Front",
                marker=marker,
                customdata=custom,
                hovertemplate=hovertemplate,
            )
        ]
    )
    if episode.knee_index is not None:
        knee = objectives[episode.knee_index]
        fig.add_trace(
            go.Scatter3d(
                x=[knee[0]],
                y=[knee[1]],
                z=[knee[2]],
                mode="markers+text",
                name="Knee Point",
                text=["Knee"],
                textposition="top center",
                marker=dict(size=9, color=_plotly_color(cfg.knee_color), symbol="diamond"),
                hovertemplate="Knee point<br>Fuel=%{x:.2f}<br>Time=%{y:.2f}<br>Risk=%{z:.3f}<extra></extra>",
            )
        )
    _apply_plotly_scene_layout(
        fig,
        title=f"{scenario_name}: 3D Pareto front",
        xaxis_title="Fuel",
        yaxis_title="Time",
        zaxis_title="Risk",
    )
    _save_plotly_html(fig, output_path, cfg)


@_styled_plot
def save_environment_overlay(
    output_path: Path,
    scenario: EncounterScenario,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """环境标量/矢量场叠加轨迹图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return
    fig, ax = plt.subplots(figsize=cfg.overlay_figsize)
    _plot_scalar_vector_background(ax, scenario, series_list[0], cfg)
    _add_obstacles(ax, scenario, cfg)
    _draw_ship_trajectories(ax, scenario, series_list, cfg, show_velocity_arrows=False)
    _draw_start_goal(ax, scenario)
    _format_spatial_axis(ax, scenario, f"{scenario.name}: trajectory overview on environment field", cfg)
    legend_handles = _series_legend_handles(series_list) + _scenario_legend_handles(scenario)
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, frameon=True, fancybox=False)
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_route_planning_panel(
    output_path: Path,
    scenario: EncounterScenario,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """复杂海域二维路线规划主图，包含密障碍环境与 bottleneck inset。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return
    fig, ax = plt.subplots(figsize=cfg.route_panel_figsize)
    _plot_scalar_vector_background(ax, scenario, series_list[0], cfg)
    _add_obstacles(ax, scenario, cfg)
    _draw_ship_trajectories(ax, scenario, series_list, cfg, show_velocity_arrows=True)
    _draw_start_goal(ax, scenario)
    _format_spatial_axis(ax, scenario, f"{scenario.name}: constrained-water route planning", cfg)
    _draw_route_insets(fig, ax, scenario, series_list, cfg)
    legend_handles = _series_legend_handles(series_list) + _scenario_legend_handles(scenario)
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, frameon=True, fancybox=False)
    reference_episode = next((series.episode for series in series_list if series.episode is not None), None)
    profile_name = reference_episode.experiment_profile if reference_episode is not None else "baseline"
    ax.text(
        0.985,
        0.02,
        f"profile={profile_name}\ntraffic={len(scenario.target_ships)}\nobstacles={len(scenario.static_obstacles)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(cfg.style.tick_size - 1, 8),
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", alpha=0.9, edgecolor="#cbd5e1"),
    )
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_scenario_gallery(
    output_path: Path,
    scenario_map: Mapping[str, EncounterScenario],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """多场景环境总览图，突出起终点、障碍类型与动态交通体。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    items = list(scenario_map.items())
    if not items:
        return
    ncols = 2
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=cfg.scenario_gallery_figsize)
    axes = np.atleast_1d(axes).ravel()
    for index, ((scenario_key, scenario), ax) in enumerate(zip(items, axes)):
        _add_obstacles(ax, scenario, cfg)
        _draw_start_goal(ax, scenario)
        _draw_traffic_initials(ax, scenario, cfg)
        _format_gallery_axis(ax, scenario, _panel_caption(index, scenario.name), cfg)
        ax.text(
            0.02,
            0.02,
            f"{scenario_key}\n{len(scenario.static_obstacles)} obstacles\n{len(scenario.target_ships)} traffic ships",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=max(cfg.style.tick_size - 1, 8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="#cbd5e1"),
        )
    for ax in axes[len(items) :]:
        ax.set_visible(False)
    handles = _scenario_legend_handles(items[0][1])
    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Generated ship-encounter environments")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_route_bundle_gallery(
    output_path: Path,
    scenario_map: Mapping[str, EncounterScenario],
    episode_groups: Mapping[str, Mapping[str, Sequence[PlanningEpisodeResult]]],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """多场景最终轨迹带图，展示跨重复运行的路线带与代表路径。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    items = list(scenario_map.items())
    if not items:
        return
    algo_styles = [
        ("kemm", "KEMM bundle", cfg.own_ship_color),
        ("nsga_style", "NSGA-style bundle", cfg.baseline_color),
    ]
    ncols = 2
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=cfg.route_bundle_figsize)
    axes = np.atleast_1d(axes).ravel()
    for index, ((scenario_key, scenario), ax) in enumerate(zip(items, axes)):
        _add_obstacles(ax, scenario, cfg)
        _draw_start_goal(ax, scenario)
        _draw_traffic_initials(ax, scenario, cfg)
        groups = episode_groups.get(scenario_key, {})
        for algo_key, _, color in algo_styles:
            episodes = list(groups.get(algo_key, []))
            for episode in episodes:
                positions = episode.final_evaluation.own_trajectory.positions
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    color=color,
                    linewidth=max(cfg.trajectory_width - 0.4, 1.2),
                    alpha=cfg.route_bundle_alpha,
                )
            if episodes:
                representative = min(
                    episodes,
                    key=lambda ep: float(np.dot(ep.final_evaluation.objectives, np.array([0.4, 0.25, 0.35], dtype=float))),
                )
                positions = representative.final_evaluation.own_trajectory.positions
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    color=color,
                    linewidth=cfg.style.emphasis_line_width,
                    alpha=cfg.representative_route_alpha,
                )
        _format_gallery_axis(ax, scenario, _panel_caption(index, f"{scenario.name} route bundles"), cfg)
    for ax in axes[len(items) :]:
        ax.set_visible(False)
    legend_handles = _scenario_legend_handles(items[0][1]) + [
        Line2D([0], [0], color=cfg.own_ship_color, lw=2.8, label="KEMM bundle"),
        Line2D([0], [0], color=cfg.baseline_color, lw=2.8, label="NSGA-style bundle"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(legend_handles), 6), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Final route bundles across repeated ship simulations")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_dynamic_avoidance_snapshots(
    output_path: Path,
    scenario: EncounterScenario,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """多时刻动态避碰快照图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    snapshots = episode.snapshots
    if not snapshots:
        return
    fig, axes = plt.subplots(2, 2, figsize=cfg.snapshot_figsize)
    axes = axes.ravel()
    for ax, snapshot in zip(axes, snapshots):
        _plot_scalar_vector_background(
            ax,
            scenario,
            ExperimentSeries(label="tmp", color=cfg.own_ship_color, episode=episode),
            cfg,
        )
        _add_obstacles(ax, scenario, cfg)
        idx = min(np.searchsorted(episode.result.own_trajectory.times, snapshot.time_s), len(episode.result.own_trajectory.times) - 1)
        ax.plot(episode.result.own_trajectory.positions[:, 0], episode.result.own_trajectory.positions[:, 1], color=cfg.own_ship_color, alpha=0.25)
        ax.scatter(snapshot.own_position[0], snapshot.own_position[1], marker="^", color=cfg.own_ship_color, s=120)
        for target_idx, target in enumerate(scenario.target_ships):
            traj = episode.result.target_trajectories[target_idx]
            target_pos = snapshot.target_positions[target_idx]
            ax.plot(traj.positions[:, 0], traj.positions[:, 1], linestyle="--", color=target.color, alpha=0.25)
            ax.scatter(target_pos[0], target_pos[1], marker="s", color=target.color, s=90)
            ax.plot([snapshot.own_position[0], target_pos[0]], [snapshot.own_position[1], target_pos[1]], linestyle=":", color="#111827", alpha=0.65)
        ax.set_title(f"t = {snapshot.time_s:.0f}s | risk = {snapshot.risk:.2f}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=cfg.style.grid_alpha)
    for ax in axes[len(snapshots) :]:
        ax.set_visible(False)
    fig.suptitle(f"{scenario.name}: time-lapse avoidance snapshots")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_spatiotemporal_plot(
    output_path: Path,
    scenario: EncounterScenario,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """三维时空轨迹图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    fig = plt.figure(figsize=cfg.spatiotemporal_figsize)
    ax = fig.add_subplot(111, projection="3d")
    own = episode.result.own_trajectory
    ax.plot(own.positions[:, 0], own.positions[:, 1], own.times, color=cfg.own_ship_color, linewidth=cfg.trajectory_width, label=episode.optimizer_name)
    for target_idx, target in enumerate(scenario.target_ships):
        traj = episode.result.target_trajectories[target_idx]
        ax.plot(traj.positions[:, 0], traj.positions[:, 1], traj.times, linestyle="--", color=target.color, linewidth=cfg.target_width, label=target.name)
    for obstacle in scenario.static_obstacles:
        if isinstance(obstacle, CircularObstacle):
            theta = np.linspace(0.0, 2.0 * np.pi, 60)
            x = obstacle.center[0] + obstacle.radius * np.cos(theta)
            y = obstacle.center[1] + obstacle.radius * np.sin(theta)
            z0 = np.full_like(theta, own.times[0])
            z1 = np.full_like(theta, own.times[-1])
            ax.plot(x, y, z0, color=cfg.obstacle_edgecolor, alpha=0.4)
            ax.plot(x, y, z1, color=cfg.obstacle_edgecolor, alpha=0.4)
        else:
            vertices = np.asarray(obstacle.vertices, dtype=float)
            ax.plot(vertices[:, 0], vertices[:, 1], np.full(len(vertices), own.times[0]), color=cfg.obstacle_edgecolor, alpha=0.4)
            ax.plot(vertices[:, 0], vertices[:, 1], np.full(len(vertices), own.times[-1]), color=cfg.obstacle_edgecolor, alpha=0.4)
    ax.set_title(f"{scenario.name}: spatiotemporal planning")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Time [s]")
    ax.legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg, allow_interactive=True)
    _save_spatiotemporal_html(output_path, scenario, episode, cfg)


@_styled_plot
def save_control_time_series(
    output_path: Path,
    scenario_name: str,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """多子图动力学/控制时序图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    traj = episode.result.own_trajectory
    fig, axes = plt.subplots(4, 1, figsize=cfg.control_figsize, sharex=True)
    axes[0].plot(traj.times, np.rad2deg(traj.headings), color=cfg.own_ship_color)
    axes[0].set_ylabel("Heading [deg]")
    axes[1].plot(traj.times, np.rad2deg(traj.commanded_yaw_rates), color="#7c3aed")
    axes[1].axhline(np.rad2deg(episode.result.analysis_metrics.get("max_commanded_yaw_rate", 0.0)), linestyle="--", color="#ef4444", alpha=0.6)
    axes[1].set_ylabel("Cmd yaw [deg/s]")
    axes[2].plot(traj.times, np.rad2deg(traj.yaw_rates), color="#10b981")
    axes[2].axhline(np.rad2deg(episode.result.analysis_metrics.get("max_yaw_rate", 0.0)), linestyle="--", color="#ef4444", alpha=0.6)
    axes[2].set_ylabel("Yaw rate [deg/s]")
    axes[3].plot(traj.times, traj.speeds, color="#f59e0b")
    axes[3].set_ylabel("Speed [m/s]")
    axes[3].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=cfg.style.grid_alpha)
        _annotate_change_events(ax, episode, cfg, with_labels=ax is axes[0])
    fig.suptitle(f"{scenario_name}: dynamic feasibility and control response")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_pareto_3d_with_knee(
    output_path: Path,
    scenario_name: str,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """最终一次局部重规划的 3D Pareto 前沿图，并高亮 knee point。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    objectives = episode.pareto_objectives
    if objectives.size == 0:
        return
    fig = plt.figure(figsize=cfg.pareto3d_figsize)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        objectives[:, 0],
        objectives[:, 1],
        objectives[:, 2],
        c=np.linalg.norm(objectives, axis=1),
        cmap=cfg.pareto_cmap,
        s=cfg.style.scatter_size,
        alpha=0.82,
    )
    if episode.knee_index is not None:
        knee = objectives[episode.knee_index]
        ax.scatter([knee[0]], [knee[1]], [knee[2]], c=cfg.knee_color, marker=cfg.knee_marker, s=cfg.knee_size, label="Knee Point")
        ax.text(knee[0], knee[1], knee[2], "  Knee", color=cfg.knee_color)
    ax.set_title(f"{scenario_name}: final local Pareto front")
    ax.set_xlabel("Fuel")
    ax.set_ylabel("Time")
    ax.set_zlabel("Risk")
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Objective-space radius")
    fig.tight_layout(rect=(0.0, 0.0, 0.95, 1.0))
    if episode.knee_index is not None:
        ax.legend(loc="best")
        knee = objectives[episode.knee_index]
        x_span = max(np.ptp(objectives[:, 0]) * 0.22, 1e-6)
        y_span = max(np.ptp(objectives[:, 1]) * 0.22, 1e-6)
        zoom_ax = fig.add_axes([0.64, 0.58, 0.28, 0.28])
        zoom_ax.scatter(
            objectives[:, 0],
            objectives[:, 1],
            c=objectives[:, 2],
            cmap=cfg.pareto_cmap,
            s=max(cfg.style.scatter_size * 0.7, 10.0),
            alpha=0.7,
        )
        zoom_ax.scatter([knee[0]], [knee[1]], c=cfg.knee_color, marker=cfg.knee_marker, s=cfg.knee_size * 0.45)
        zoom_ax.set_xlim(knee[0] - x_span, knee[0] + x_span)
        zoom_ax.set_ylim(knee[1] - y_span, knee[1] + y_span)
        zoom_ax.set_title("Knee Zoom", fontsize=max(cfg.style.tick_size, 8))
        zoom_ax.set_xlabel("Fuel", fontsize=cfg.style.tick_size)
        zoom_ax.set_ylabel("Time", fontsize=cfg.style.tick_size)
        zoom_ax.grid(True, alpha=cfg.style.grid_alpha)
    _finalize_figure(fig, output_path, cfg, allow_interactive=True)
    _save_pareto_html(output_path, scenario_name, episode, cfg)


@_styled_plot
def save_pareto_projection_panel(
    output_path: Path,
    scenario_name: str,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """最终局部 Pareto 前沿的二维投影视图与 projected frontier curve。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    objectives = np.asarray(episode.pareto_objectives, dtype=float)
    if objectives.size == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=cfg.pareto_projection_figsize)
    projections = (
        (0, 1, "Fuel", "Time"),
        (0, 2, "Fuel", "Risk"),
        (1, 2, "Time", "Risk"),
    )
    knee = objectives[episode.knee_index] if episode.knee_index is not None else None
    for ax, (x_idx, y_idx, x_label, y_label) in zip(np.atleast_1d(axes), projections):
        points = objectives[:, [x_idx, y_idx]]
        front_order = _projection_front_order(points)
        ax.scatter(points[:, 0], points[:, 1], color="#94a3b8", alpha=0.45, s=cfg.style.scatter_size * 0.9, label="Projected points")
        if front_order.size:
            front = points[front_order]
            ax.plot(front[:, 0], front[:, 1], color=cfg.own_ship_color, linewidth=cfg.style.emphasis_line_width, label="Projected frontier")
        if knee is not None:
            ax.scatter([knee[x_idx]], [knee[y_idx]], c=cfg.knee_color, marker=cfg.knee_marker, s=cfg.knee_size * 0.5, label="Knee")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label}")
        ax.grid(True, alpha=cfg.style.grid_alpha)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{scenario_name}: pairwise Pareto projections of final local front")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_parallel_coordinates(
    output_path: Path,
    scenario_name: str,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if len(series_list) < 1:
        return
    metric_names, values = _select_metrics(series_list)
    normalized = _normalized(values, minimize_mask=[True, True, True, False, True, True])
    fig, ax = plt.subplots(figsize=cfg.parallel_figsize)
    x = np.arange(len(metric_names))
    for idx, series in enumerate(series_list):
        linewidth = cfg.style.emphasis_line_width if idx == 0 else cfg.style.line_width
        alpha = 1.0 if idx < 3 else 0.4
        ax.plot(x, normalized[idx], color=series.color, linewidth=linewidth, alpha=alpha, label=series.label)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_title(f"{scenario_name}: parallel coordinates")
    ax.grid(True, alpha=cfg.style.grid_alpha)
    ax.legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_radar_chart(
    output_path: Path,
    scenario_name: str,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return
    metric_names, values = _select_metrics(series_list)
    normalized = _normalized(values, minimize_mask=[True, True, True, False, True, True])
    angles = np.linspace(0.0, 2.0 * np.pi, len(metric_names), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=cfg.radar_figsize)
    ax = fig.add_subplot(111, polar=True)
    for idx, series in enumerate(series_list):
        stats = np.concatenate([normalized[idx], [normalized[idx, 0]]])
        ax.plot(angles, stats, color=series.color, linewidth=cfg.style.line_width, label=series.label)
        ax.fill(angles, stats, color=series.color, alpha=cfg.radar_fill_alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_title(f"{scenario_name}: radar comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_convergence_statistics(
    output_path: Path,
    scenario_name: str,
    histories_by_label: Mapping[str, Sequence[Sequence[dict[str, float]]]],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    fig, ax = plt.subplots(figsize=cfg.convergence_figsize)
    for idx, (label, histories) in enumerate(histories_by_label.items()):
        if not histories:
            continue
        curves = []
        for history in histories:
            curve = []
            for row in history:
                score = row.get("best_weighted_score")
                if score is None:
                    score = row.get("best_fuel", 0.0) * 0.4 + row.get("best_time", 0.0) * 0.25 + row.get("best_risk", 0.0) * 0.35
                curve.append(float(score))
            if curve:
                curves.append(curve)
        if not curves:
            continue
        max_len = max(len(curve) for curve in curves)
        padded = np.asarray([curve + [curve[-1]] * (max_len - len(curve)) for curve in curves], dtype=float)
        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)
        x = np.arange(len(mean_curve))
        color = [cfg.own_ship_color, cfg.baseline_color, cfg.third_algo_color][idx % 3]
        ax.plot(x, mean_curve, color=color, linewidth=cfg.style.line_width, label=label)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=cfg.style.band_alpha)
    ax.set_title(f"{scenario_name}: convergence with confidence bands")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best actual weighted score")
    ax.grid(True, alpha=cfg.style.grid_alpha)
    ax.legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_distribution_violin(
    output_path: Path,
    scenario_name: str,
    metrics_by_label: Mapping[str, Sequence[dict[str, float]]],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    labels = list(metrics_by_label.keys())
    if not labels:
        return
    metric_keys = ["fuel", "time", "risk"]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=cfg.violin_figsize)
    for ax, metric in zip(np.atleast_1d(axes), metric_keys):
        data = [[row.get(metric, np.nan) for row in metrics_by_label[label]] for label in labels]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(cfg.violin_alpha)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(metric.capitalize())
        ax.grid(True, axis="y", alpha=cfg.style.grid_alpha)
    fig.suptitle(f"{scenario_name}: distribution comparison")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_risk_breakdown_time_series(
    output_path: Path,
    scenario_name: str,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """风险总量与分解项的时间历程图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    risk = episode.result.risk
    times = episode.result.own_trajectory.times
    fig, axes = plt.subplots(2, 1, figsize=cfg.risk_breakdown_figsize, sharex=True)
    axes[0].plot(times[: len(risk.risk_series)], risk.risk_series, color=cfg.own_ship_color, label="Total risk")
    axes[0].axhline(cfg.risk_threshold, linestyle="--", color=cfg.risk_threshold_color, alpha=0.7, label="Risk threshold")
    axes[0].set_ylabel("Total risk")
    axes[0].set_title("Integrated collision-risk response")
    axes[0].grid(True, alpha=cfg.style.grid_alpha)
    axes[0].legend(loc="best")
    _annotate_change_events(axes[0], episode, cfg, with_labels=True)

    components = [
        ("Domain", risk.domain_risk_series, "#0f766e"),
        ("DCPA/TCPA", risk.dcpa_risk_series, "#7c3aed"),
        ("Obstacle", risk.obstacle_risk_series, "#dc2626"),
        ("Environment", risk.environment_risk_series, "#f59e0b"),
    ]
    for label, values, color in components:
        axes[1].plot(times[: len(values)], values, color=color, linewidth=cfg.style.line_width, label=label)
    axes[1].set_title("Risk component breakdown")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Component intensity")
    axes[1].grid(True, alpha=cfg.style.grid_alpha)
    axes[1].legend(loc="best", ncol=2)
    _annotate_change_events(axes[1], episode, cfg)
    fig.suptitle(f"{scenario_name}: risk breakdown time series")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_safety_envelope_plot(
    output_path: Path,
    scenario_name: str,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """净空、避船距离与 DCPA/TCPA/COLREG 安全包络图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    risk = episode.result.risk
    times = episode.result.own_trajectory.times
    fig, axes = plt.subplots(2, 1, figsize=cfg.safety_envelope_figsize, sharex=True)

    axes[0].plot(times[: len(risk.clearance_series)], risk.clearance_series, color=cfg.own_ship_color, label="Overall clearance")
    axes[0].plot(times[: len(risk.static_clearance_series)], risk.static_clearance_series, color="#7c3aed", linestyle="--", label="Static clearance")
    axes[0].plot(times[: len(risk.ship_distance_series)], risk.ship_distance_series, color="#16a34a", label="Ship distance")
    axes[0].axhline(0.0, color="#991b1b", linestyle=":", alpha=0.75)
    axes[0].set_ylabel("Distance [m]")
    axes[0].set_title("Clearance and encounter separation")
    axes[0].grid(True, alpha=cfg.style.grid_alpha)
    axes[0].legend(loc="best")
    _annotate_change_events(axes[0], episode, cfg, with_labels=True)

    dcpa = np.where(np.isfinite(risk.dcpa_series), risk.dcpa_series, np.nan)
    tcpa = np.where(np.isfinite(risk.tcpa_series), risk.tcpa_series, np.nan)
    axes[1].plot(times[: len(dcpa)], dcpa, color="#0f766e", label="DCPA")
    axes[1].plot(times[: len(tcpa)], tcpa, color="#f97316", label="TCPA")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("CPA metric")
    axes[1].set_title("Encounter envelope with COLREG scaling")
    axes[1].grid(True, alpha=cfg.style.grid_alpha)
    twin = axes[1].twinx()
    twin.plot(times[: len(risk.colreg_scale_series)], risk.colreg_scale_series, color="#b42318", linestyle="--", label="COLREG scale")
    twin.set_ylabel("COLREG scale")
    handles, labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, labels + twin_labels, loc="best")
    _annotate_change_events(axes[1], episode, cfg)
    fig.suptitle(f"{scenario_name}: safety envelope")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_change_timeline_panel(
    output_path: Path,
    scenario_name: str,
    episode: PlanningEpisodeResult,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """按重规划 step 展示实验变化、风险恢复和规划开销。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    steps = episode.steps
    if not steps:
        return
    step_index = np.arange(len(steps), dtype=float)
    weighted = np.asarray(
        [
            0.4 * float(step.selected_evaluation.objectives[0])
            + 0.25 * float(step.selected_evaluation.objectives[1])
            + 0.35 * float(step.selected_evaluation.objectives[2])
            for step in steps
        ],
        dtype=float,
    )
    risk = np.asarray([float(step.selected_evaluation.risk.max_risk) for step in steps], dtype=float)
    runtime = np.asarray([float(step.runtime_s) for step in steps], dtype=float)
    labels = ["None"] * len(steps)
    for step in steps:
        if step.applied_changes:
            labels[step.step_index] = " + ".join(str(change.get("label", "Change")) for change in step.applied_changes)

    fig, axes = plt.subplots(3, 1, figsize=cfg.change_timeline_figsize, sharex=True)
    axes[0].plot(step_index, weighted, color=cfg.own_ship_color, marker="o")
    axes[0].set_ylabel("Weighted objective")
    axes[0].set_title("Local-front quality across replanning steps")
    axes[0].grid(True, alpha=cfg.style.grid_alpha)

    axes[1].plot(step_index, risk, color="#dc2626", marker="o")
    axes[1].axhline(cfg.risk_threshold, linestyle="--", color=cfg.risk_threshold_color, alpha=0.7)
    axes[1].set_ylabel("Max risk")
    axes[1].set_title("Risk response after scheduled changes")
    axes[1].grid(True, alpha=cfg.style.grid_alpha)

    bars = axes[2].bar(step_index, runtime, color="#64748b", alpha=0.88)
    axes[2].set_ylabel("Runtime [s]")
    axes[2].set_xlabel("Planning step")
    axes[2].set_title("Solver burden and applied scenario events")
    axes[2].grid(True, axis="y", alpha=cfg.style.grid_alpha)
    axes[2].set_xticks(step_index)
    axes[2].set_xticklabels([str(int(idx)) for idx in step_index])

    for idx, label in enumerate(labels):
        if label == "None":
            continue
        for ax in axes:
            ax.axvline(idx, color=cfg.event_line_color, linestyle="--", linewidth=1.1, alpha=0.78)
        axes[2].text(
            idx,
            runtime[idx] + max(np.max(runtime) * 0.05, 0.5),
            label,
            rotation=20,
            ha="left",
            va="bottom",
            color=cfg.event_line_color,
            fontsize=max(cfg.style.tick_size - 1, 8),
        )
        bars[int(idx)].set_color(cfg.event_line_color)
    fig.suptitle(f"{scenario_name}: replanning change timeline")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_run_statistics_panel(
    output_path: Path,
    scenario_name: str,
    series_list: Iterable[ExperimentSeries],
    plot_config: ShipPlotConfig | None = None,
) -> None:
    """跨重复运行的安全与效率统计图。"""

    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return
    metrics = [
        ("success_rate", "Success rate", 100.0),
        ("minimum_clearance", "Min clearance [m]", 1.0),
        ("minimum_ship_distance", "Min ship distance [m]", 1.0),
        ("runtime", "Runtime [s]", 1.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=cfg.statistics_figsize)
    x = np.arange(len(series_list))
    for ax, (key, title, scale) in zip(np.atleast_1d(axes), metrics):
        means = []
        stds = []
        colors = []
        labels = []
        for series in series_list:
            mean, std = _stat_value(series, key)
            means.append(mean * scale)
            stds.append(std * scale)
            colors.append(series.color)
            labels.append(series.label)
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.88, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=cfg.style.grid_alpha)
        if key == "success_rate":
            ax.set_ylim(0.0, 105.0)
            ax.set_ylabel("Percent [%]")
    fig.suptitle(f"{scenario_name}: repeated-run statistics")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_summary_dashboard(
    output_path: Path,
    scenario: EncounterScenario,
    series_list: Iterable[ExperimentSeries],
    histories_by_label: Mapping[str, Sequence[Sequence[dict[str, float]]]] | None = None,
    metrics_by_label: Mapping[str, Sequence[dict[str, float]]] | None = None,
    plot_config: ShipPlotConfig | None = None,
) -> None:
    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    series_list = list(series_list)
    if not series_list:
        return
    fig = plt.figure(figsize=cfg.dashboard_figsize)
    grid = GridSpec(2, 2, figure=fig)
    ax_overlay = fig.add_subplot(grid[0, 0])
    ax_radar = fig.add_subplot(grid[0, 1], polar=True)
    ax_conv = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    _plot_scalar_vector_background(ax_overlay, scenario, series_list[0], cfg)
    _add_obstacles(ax_overlay, scenario, cfg)
    _draw_start_goal(ax_overlay, scenario)
    for target_idx, target in enumerate(scenario.target_ships):
        traj = series_list[0].result.target_trajectories[target_idx]
        ax_overlay.plot(traj.positions[:, 0], traj.positions[:, 1], linestyle="--", linewidth=cfg.target_width, color=target.color)
    for series in series_list:
        positions = series.result.own_trajectory.positions
        ax_overlay.plot(positions[:, 0], positions[:, 1], linewidth=cfg.trajectory_width, color=series.color, label=series.label)
    ax_overlay.set_title("Environment + Trajectory")
    ax_overlay.set_aspect("equal", adjustable="box")
    ax_overlay.grid(True, alpha=cfg.style.grid_alpha)
    ax_overlay.legend(loc="best")

    metric_names, values = _select_metrics(series_list)
    normalized = _normalized(values, minimize_mask=[True, True, True, False, True, True])
    angles = np.linspace(0.0, 2.0 * np.pi, len(metric_names), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    for idx, series in enumerate(series_list):
        stats = np.concatenate([normalized[idx], [normalized[idx, 0]]])
        ax_radar.plot(angles, stats, color=series.color, linewidth=cfg.style.line_width)
        ax_radar.fill(angles, stats, color=series.color, alpha=cfg.radar_fill_alpha)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_names)
    ax_radar.set_title("Radar Comparison")

    if histories_by_label:
        for idx, (label, histories) in enumerate(histories_by_label.items()):
            curves = []
            for history in histories:
                curve = [row.get("best_risk", 0.0) for row in history]
                if curve:
                    curves.append(curve)
            if not curves:
                continue
            max_len = max(len(curve) for curve in curves)
            padded = np.asarray([curve + [curve[-1]] * (max_len - len(curve)) for curve in curves], dtype=float)
            mean_curve = np.mean(padded, axis=0)
            std_curve = np.std(padded, axis=0)
            x = np.arange(len(mean_curve))
            color = [cfg.own_ship_color, cfg.baseline_color, cfg.third_algo_color][idx % 3]
            ax_conv.plot(x, mean_curve, color=color, label=label)
            ax_conv.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=cfg.style.band_alpha)
    ax_conv.set_title("Risk-Convergence")
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Best risk")
    ax_conv.grid(True, alpha=cfg.style.grid_alpha)
    handles, labels = ax_conv.get_legend_handles_labels()
    if handles:
        ax_conv.legend(loc="best")

    ax_text.axis("off")
    representative_episode = series_list[0].episode
    if representative_episode is not None:
        lines = [
            f"Scenario: {scenario.name}",
            f"Representative episode: {representative_episode.optimizer_name}",
            f"Final local knee point: {representative_episode.knee_objectives.tolist() if representative_episode.knee_objectives is not None else 'N/A'}",
            f"Fuel / Time / Risk: {representative_episode.final_evaluation.objectives.round(3).tolist()}",
            f"Overall / static clearance: {representative_episode.analysis_metrics.get('minimum_clearance', 0.0):.2f} / {representative_episode.analysis_metrics.get('minimum_static_clearance', 0.0):.2f} m",
            f"Min ship distance: {representative_episode.analysis_metrics.get('minimum_ship_distance', 0.0):.2f} m",
            f"Min DCPA / TCPA: {representative_episode.analysis_metrics.get('minimum_dcpa', 0.0):.2f} / {representative_episode.analysis_metrics.get('minimum_tcpa', 0.0):.2f}",
            f"Planning steps: {len(representative_episode.steps)}",
            f"Terminated by: {representative_episode.terminated_reason}",
        ]
    else:
        result = series_list[0].result
        lines = [
            f"Scenario: {scenario.name}",
            f"Representative result: {series_list[0].label}",
            f"Fuel / Time / Risk: {result.objectives.round(3).tolist()}",
            f"Reached goal: {result.reached_goal}",
            f"Terminal distance: {result.terminal_distance:.2f} m",
            f"Max / Mean risk: {result.risk.max_risk:.2f} / {result.risk.mean_risk:.2f}",
        ]
    ax_text.text(0.02, 0.98, "\n".join(lines), va="top", ha="left")

    fig.suptitle(f"{scenario.name}: summary dashboard")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


# 兼容层：旧函数名继续保留，但默认语义已切换到新的论文图包。
@_styled_plot
def save_trajectory_comparison(output_path: Path, scenario: EncounterScenario, series_list: Iterable[ExperimentSeries], plot_config: ShipPlotConfig | None = None) -> None:
    save_environment_overlay(output_path, scenario, series_list, plot_config=plot_config)


@_styled_plot
def save_convergence_plot(output_path: Path, scenario_name: str, history: Sequence[dict[str, float]], plot_config: ShipPlotConfig | None = None) -> None:
    histories = {scenario_name: [history]}
    save_convergence_statistics(output_path, scenario_name, histories, plot_config=plot_config)


@_styled_plot
def save_pareto_scatter(output_path: Path, scenario_name: str, pareto_objectives: np.ndarray, plot_config: ShipPlotConfig | None = None) -> None:
    _ensure_parent(output_path)
    cfg = _resolve_plot_config(plot_config)
    if pareto_objectives is None or len(pareto_objectives) == 0:
        return
    span = np.ptp(pareto_objectives, axis=0)
    normalized = (pareto_objectives - pareto_objectives.min(axis=0)) / (span + 1e-9)
    knee_index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    knee = pareto_objectives[knee_index]
    fig = plt.figure(figsize=cfg.pareto3d_figsize)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pareto_objectives[:, 0],
        pareto_objectives[:, 1],
        pareto_objectives[:, 2],
        c=np.linalg.norm(pareto_objectives, axis=1),
        cmap=cfg.pareto_cmap,
        s=cfg.style.scatter_size,
        alpha=0.82,
    )
    ax.scatter([knee[0]], [knee[1]], [knee[2]], c=cfg.knee_color, marker=cfg.knee_marker, s=cfg.knee_size, label="Knee Point")
    ax.set_title(f"{scenario_name}: 3D Pareto front")
    ax.set_xlabel("Fuel")
    ax.set_ylabel("Time")
    ax.set_zlabel("Risk")
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Objective-space radius")
    ax.legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_risk_time_series(output_path: Path, scenario_name: str, series_list: Iterable[ExperimentSeries], plot_config: ShipPlotConfig | None = None) -> None:
    series_list = list(series_list)
    if not series_list:
        return
    fig, ax = plt.subplots(figsize=_resolve_plot_config(plot_config).time_series_figsize)
    cfg = _resolve_plot_config(plot_config)
    for series in series_list:
        risk_series = np.asarray(series.result.risk.risk_series, dtype=float)
        ax.plot(series.result.own_trajectory.times[: risk_series.size], risk_series, color=series.color, label=series.label)
    ax.axhline(cfg.risk_threshold, linestyle="--", color=cfg.risk_threshold_color, label="Boundary")
    ax.set_title(f"{scenario_name}: risk time series")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Risk")
    ax.grid(True, alpha=cfg.style.grid_alpha)
    ax.legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_speed_profiles(output_path: Path, scenario_name: str, series_list: Iterable[ExperimentSeries], plot_config: ShipPlotConfig | None = None) -> None:
    series_list = list(series_list)
    if not series_list:
        return
    fig, ax = plt.subplots(figsize=_resolve_plot_config(plot_config).time_series_figsize)
    cfg = _resolve_plot_config(plot_config)
    for series in series_list:
        ax.plot(series.result.own_trajectory.times, series.result.own_trajectory.speeds, color=series.color, label=series.label)
    ax.set_title(f"{scenario_name}: speed profile")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True, alpha=cfg.style.grid_alpha)
    ax.legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_normalized_objective_bars(output_path: Path, scenario_names: Sequence[str], kemm_objectives: np.ndarray, random_objectives: np.ndarray, plot_config: ShipPlotConfig | None = None) -> None:
    cfg = _resolve_plot_config(plot_config)
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 3, figsize=cfg.comparison_figsize)
    metric_names = ["Fuel", "Time", "Risk"]
    for metric_idx, ax in enumerate(axes):
        pair = np.vstack([kemm_objectives[:, metric_idx], random_objectives[:, metric_idx]])
        denom = np.max(pair, axis=0)
        normalized_kemm = kemm_objectives[:, metric_idx] / (denom + 1e-9)
        normalized_random = random_objectives[:, metric_idx] / (denom + 1e-9)
        x = np.arange(len(scenario_names))
        width = 0.36
        ax.bar(x - width / 2, normalized_kemm, width=width, color=cfg.own_ship_color, label="KEMM")
        ax.bar(x + width / 2, normalized_random, width=width, color=cfg.baseline_color, label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=10)
        ax.set_title(f"Normalized {metric_names[metric_idx]}")
    axes[0].legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)


@_styled_plot
def save_risk_bars(output_path: Path, scenario_names: Sequence[str], kemm_risk_triplets: np.ndarray, random_risk_triplets: np.ndarray, plot_config: ShipPlotConfig | None = None) -> None:
    cfg = _resolve_plot_config(plot_config)
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 3, figsize=cfg.comparison_figsize)
    metric_names = ["Max Risk", "Mean Risk", "Intrusion Time"]
    for metric_idx, ax in enumerate(axes):
        x = np.arange(len(scenario_names))
        width = 0.36
        ax.bar(x - width / 2, kemm_risk_triplets[:, metric_idx], width=width, color=cfg.own_ship_color, label="KEMM")
        ax.bar(x + width / 2, random_risk_triplets[:, metric_idx], width=width, color=cfg.baseline_color, label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=10)
        ax.set_title(metric_names[metric_idx])
    axes[0].legend(loc="best")
    fig.tight_layout()
    _finalize_figure(fig, output_path, cfg)
