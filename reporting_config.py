"""benchmark 与 ship 主线共享的绘图配置层。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import pickle
import warnings

try:
    import matplotlib

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

try:
    import scienceplots  # noqa: F401

    HAS_SCIENCEPLOTS = True
except ImportError:  # pragma: no cover
    HAS_SCIENCEPLOTS = False


DEFAULT_BENCHMARK_COLORS = {
    "RI": "#6c7a89",
    "PPS": "#0f6cbd",
    "KF": "#2f855a",
    "SVR": "#d97706",
    "Tr": "#7c3aed",
    "MMTL": "#c2410c",
    "KEMM": "#b42318",
}

DEFAULT_BENCHMARK_MARKERS = {
    "RI": "o",
    "PPS": "s",
    "KF": "^",
    "SVR": "D",
    "Tr": "v",
    "MMTL": "P",
    "KEMM": "*",
}


@dataclass
class PublicationStyle:
    """Matplotlib 的通用排版配置。"""

    dpi: int = 320
    font_family: str = "DejaVu Sans"
    chinese_font_fallback: tuple[str, ...] = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
    use_scienceplots: bool = False
    science_styles: tuple[str, ...] = ("science", "no-latex")
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10
    legend_size: int = 9
    line_width: float = 2.1
    emphasis_line_width: float = 2.8
    marker_size: float = 5.6
    scatter_size: float = 34.0
    grid_alpha: float = 0.18
    band_alpha: float = 0.12
    bar_alpha: float = 0.9
    figure_facecolor: str = "#ffffff"
    axes_facecolor: str = "#fcfcfd"
    grid_color: str = "#d0d7de"
    spine_color: str = "#9aa4b2"


PLOT_STYLE_PRESETS = {
    "default": dict(dpi=320, font_family="DejaVu Sans", use_scienceplots=False, science_styles=("science", "no-latex")),
    "paper": dict(dpi=360, font_family="DejaVu Sans", use_scienceplots=True, science_styles=("science", "no-latex"), title_size=15, label_size=12),
    "ieee": dict(dpi=380, font_family="DejaVu Sans", use_scienceplots=True, science_styles=("science", "ieee", "no-latex"), title_size=14, label_size=11, tick_size=9, legend_size=8),
    "nature": dict(dpi=380, font_family="DejaVu Sans", use_scienceplots=True, science_styles=("science", "nature", "no-latex"), title_size=15, label_size=11, tick_size=9, legend_size=8, grid_alpha=0.10),
    "thesis": dict(dpi=320, font_family="DejaVu Sans", use_scienceplots=False, science_styles=("science", "no-latex"), title_size=16, label_size=12, legend_size=10),
}


BENCHMARK_PLOT_PRESETS = {
    "default": dict(),
    "paper": dict(metric_panel_width=4.8, metrics_grid_height=3.2, dashboard_width=12.5, dashboard_height=8.2),
    "ieee": dict(metric_panel_width=4.4, metrics_grid_height=3.0, rank_bar_width=9.5, rank_bar_height=4.0),
    "nature": dict(metric_panel_width=4.6, metrics_grid_height=3.0, dashboard_width=12.0, dashboard_height=7.8),
    "thesis": dict(metric_panel_width=5.0, metrics_grid_height=3.4, dashboard_width=13.0, dashboard_height=8.8),
}


SHIP_PLOT_PRESETS = {
    "default": dict(),
    "paper": dict(
        overlay_figsize=(11.8, 7.0),
        route_panel_figsize=(13.2, 8.2),
        scenario_gallery_figsize=(12.8, 9.4),
        route_bundle_figsize=(13.0, 9.8),
        snapshot_figsize=(13.0, 8.4),
        spatiotemporal_figsize=(10.2, 7.0),
        control_figsize=(10.4, 8.8),
        pareto3d_figsize=(8.2, 6.2),
        pareto_projection_figsize=(12.2, 4.2),
        parallel_figsize=(11.8, 5.6),
        radar_figsize=(7.2, 7.0),
        violin_figsize=(11.0, 5.8),
        convergence_figsize=(9.4, 5.2),
        risk_breakdown_figsize=(10.2, 7.0),
        safety_envelope_figsize=(10.2, 7.2),
        change_timeline_figsize=(10.6, 6.8),
        statistics_figsize=(12.0, 4.8),
        dashboard_figsize=(15.2, 11.2),
    ),
    "ieee": dict(overlay_figsize=(10.8, 6.5), snapshot_figsize=(12.0, 7.6), dashboard_figsize=(14.0, 10.2)),
    "nature": dict(overlay_figsize=(10.8, 6.5), snapshot_figsize=(12.0, 7.6), dashboard_figsize=(14.2, 10.2)),
    "thesis": dict(overlay_figsize=(12.4, 7.4), snapshot_figsize=(13.6, 8.8), dashboard_figsize=(16.2, 12.0)),
}


@dataclass
class BenchmarkPlotConfig:
    style: PublicationStyle = field(default_factory=PublicationStyle)
    colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BENCHMARK_COLORS))
    markers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BENCHMARK_MARKERS))
    highlight_color: str = "#b42318"
    best_outline_color: str = "#f5b700"
    heatmap_cmap: str = "RdYlGn_r"
    significance_cmap: str = "YlOrRd_r"
    pairwise_cmap: str = "Blues"
    metrics_grid_height: float = 3.1
    metric_panel_width: float = 4.3
    rank_bar_width: float = 10.0
    rank_bar_height: float = 4.2
    dashboard_width: float = 12.0
    dashboard_height: float = 8.0
    appendix_plots: bool = False
    interactive_figures: bool = False


@dataclass
class ShipPlotConfig:
    style: PublicationStyle = field(default_factory=PublicationStyle)
    own_ship_color: str = "#0f6cbd"
    baseline_color: str = "#d97706"
    third_algo_color: str = "#2f855a"
    risk_threshold_color: str = "#b42318"
    scalar_cmap: str = "magma"
    pareto_cmap: str = "viridis"
    vector_color: str = "#334155"
    obstacle_facecolor: str = "#cbd5e1"
    obstacle_edgecolor: str = "#475569"
    knee_color: str = "#b42318"
    knee_marker: str = "*"
    knee_size: float = 180.0
    trajectory_width: float = 2.4
    target_width: float = 1.8
    overlay_figsize: tuple[float, float] = (11.5, 6.8)
    route_panel_figsize: tuple[float, float] = (12.6, 8.0)
    scenario_gallery_figsize: tuple[float, float] = (12.4, 9.2)
    route_bundle_figsize: tuple[float, float] = (12.8, 9.6)
    snapshot_figsize: tuple[float, float] = (12.5, 8.0)
    spatiotemporal_figsize: tuple[float, float] = (10.0, 6.8)
    control_figsize: tuple[float, float] = (10.0, 8.6)
    pareto3d_figsize: tuple[float, float] = (8.0, 6.0)
    pareto_projection_figsize: tuple[float, float] = (11.8, 4.0)
    parallel_figsize: tuple[float, float] = (11.5, 5.4)
    radar_figsize: tuple[float, float] = (7.0, 7.0)
    violin_figsize: tuple[float, float] = (10.8, 5.6)
    convergence_figsize: tuple[float, float] = (9.0, 5.0)
    risk_breakdown_figsize: tuple[float, float] = (10.0, 6.8)
    safety_envelope_figsize: tuple[float, float] = (10.0, 7.0)
    change_timeline_figsize: tuple[float, float] = (10.4, 6.8)
    statistics_figsize: tuple[float, float] = (11.4, 4.8)
    dashboard_figsize: tuple[float, float] = (15.0, 11.0)
    comparison_figsize: tuple[float, float] = (14.0, 4.8)
    time_series_figsize: tuple[float, float] = (8.8, 4.8)
    risk_threshold: float = 1.0
    scalar_grid_resolution: int = 80
    vector_grid_resolution: int = 18
    snapshot_alpha: float = 0.35
    radar_fill_alpha: float = 0.22
    violin_alpha: float = 0.55
    inset_zoom_alpha: float = 0.92
    velocity_arrow_scale: float = 55.0
    obstacle_alpha: float = 0.48
    boundary_alpha: float = 0.22
    route_bundle_alpha: float = 0.32
    representative_route_alpha: float = 0.96
    event_line_color: str = "#be123c"
    event_fill_color: str = "#fecdd3"
    appendix_plots: bool = False
    interactive_figures: bool = False
    interactive_html: bool = False
    interactive_html_include_plotlyjs: str = "cdn"


def list_plot_presets() -> list[str]:
    return sorted(PLOT_STYLE_PRESETS.keys())


def build_publication_style(preset: str = "default", **overrides) -> PublicationStyle:
    if preset not in PLOT_STYLE_PRESETS:
        raise ValueError(f"Unknown plot preset: {preset}. Available presets: {', '.join(list_plot_presets())}")
    payload = dict(PLOT_STYLE_PRESETS[preset])
    payload.update(overrides)
    return PublicationStyle(**payload)


def build_benchmark_plot_config(preset: str = "default", style_overrides: dict | None = None, **config_overrides) -> BenchmarkPlotConfig:
    style = build_publication_style(preset, **(style_overrides or {}))
    payload = dict(BENCHMARK_PLOT_PRESETS.get(preset, {}))
    payload.update(config_overrides)
    payload["style"] = style
    return BenchmarkPlotConfig(**payload)


def build_ship_plot_config(preset: str = "default", style_overrides: dict | None = None, **config_overrides) -> ShipPlotConfig:
    style = build_publication_style(preset, **(style_overrides or {}))
    payload = dict(SHIP_PLOT_PRESETS.get(preset, {}))
    payload.update(config_overrides)
    payload["style"] = style
    return ShipPlotConfig(**payload)


def _style_to_rc(style: PublicationStyle) -> dict[str, object]:
    font_candidates = [style.font_family, *style.chinese_font_fallback]
    if HAS_MPL:
        available = {item.name for item in matplotlib.font_manager.fontManager.ttflist}
        font_candidates = [name for name in font_candidates if name in available] or [style.font_family]
    return {
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
        "font.family": font_candidates,
        "font.sans-serif": font_candidates,
        "axes.unicode_minus": False,
        "axes.titlesize": style.title_size,
        "axes.labelsize": style.label_size,
        "xtick.labelsize": style.tick_size,
        "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.legend_size,
        "axes.facecolor": style.axes_facecolor,
        "figure.facecolor": style.figure_facecolor,
        "axes.edgecolor": style.spine_color,
        "axes.linewidth": 0.8,
        "grid.color": style.grid_color,
        "grid.alpha": style.grid_alpha,
        "grid.linestyle": "-",
        "lines.linewidth": style.line_width,
    }


def _resolve_style_sheets(style: PublicationStyle) -> list[str]:
    if not style.use_scienceplots:
        return []
    if not HAS_SCIENCEPLOTS:
        warnings.warn(
            "SciencePlots is not installed. Falling back to the built-in matplotlib style configuration.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    return list(style.science_styles)


def interactive_bundle_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(".fig.pickle")


def save_figure_bundle(
    fig,
    output_path: str | Path,
    *,
    dpi: int,
    interactive_figures: bool = False,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if interactive_figures:
        bundle_path = interactive_bundle_path(path)
        with bundle_path.open("wb") as handle:
            pickle.dump(fig, handle, protocol=pickle.HIGHEST_PROTOCOL)


@contextmanager
def plot_style_context(style: PublicationStyle) -> Iterator[None]:
    if not HAS_MPL:  # pragma: no cover
        yield
        return

    previous = matplotlib.rcParams.copy()
    style_sheets = _resolve_style_sheets(style)
    if style_sheets:
        matplotlib.style.use(style_sheets)
    matplotlib.rcParams.update(_style_to_rc(style))
    try:
        yield
    finally:
        matplotlib.rcParams.update(previous)


__all__ = [
    "BENCHMARK_PLOT_PRESETS",
    "BenchmarkPlotConfig",
    "DEFAULT_BENCHMARK_COLORS",
    "DEFAULT_BENCHMARK_MARKERS",
    "HAS_SCIENCEPLOTS",
    "PLOT_STYLE_PRESETS",
    "PublicationStyle",
    "SHIP_PLOT_PRESETS",
    "ShipPlotConfig",
    "build_benchmark_plot_config",
    "build_publication_style",
    "build_ship_plot_config",
    "interactive_bundle_path",
    "list_plot_presets",
    "plot_style_context",
    "save_figure_bundle",
]
