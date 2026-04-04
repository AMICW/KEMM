"""Shared plot-style configuration for benchmark and ship reports.

这个模块把“论文图表长什么样”从具体绘图函数里抽出来，形成显式配置层。
后续如果你要：

- 调整字体、字号、线宽、DPI
- 更换配色方案
- 统一 benchmark 与 ship 的图表风格
- 针对投稿期刊改版尺寸

优先改这里，而不是逐个图函数内找硬编码。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

try:
    import matplotlib

    HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MPL = False


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
    """Matplotlib 通用排版参数。"""

    dpi: int = 320
    font_family: str = "DejaVu Sans"
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


@dataclass
class BenchmarkPlotConfig:
    """benchmark 主线图表配置。"""

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


@dataclass
class ShipPlotConfig:
    """ship 主线图表配置。"""

    style: PublicationStyle = field(default_factory=PublicationStyle)
    own_ship_color: str = "#0f6cbd"
    baseline_color: str = "#d97706"
    risk_threshold_color: str = "#b42318"
    pareto_cmap: str = "cividis"
    trajectory_width: float = 2.4
    target_width: float = 1.7
    trajectory_figsize: tuple[float, float] = (10.5, 6.4)
    comparison_figsize: tuple[float, float] = (14.0, 4.8)
    convergence_figsize: tuple[float, float] = (8.8, 4.8)
    pareto_figsize: tuple[float, float] = (7.4, 5.4)
    time_series_figsize: tuple[float, float] = (8.8, 4.8)
    dashboard_figsize: tuple[float, float] = (14.0, 10.0)
    risk_threshold: float = 1.0


def _style_to_rc(style: PublicationStyle) -> dict[str, object]:
    return {
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
        "font.family": style.font_family,
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


@contextmanager
def plot_style_context(style: PublicationStyle) -> Iterator[None]:
    """临时应用论文风格的 matplotlib rcParams。"""

    if not HAS_MPL:  # pragma: no cover - matplotlib absent
        yield
        return

    previous = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(_style_to_rc(style))
    try:
        yield
    finally:
        matplotlib.rcParams.update(previous)


__all__ = [
    "BenchmarkPlotConfig",
    "DEFAULT_BENCHMARK_COLORS",
    "DEFAULT_BENCHMARK_MARKERS",
    "PublicationStyle",
    "ShipPlotConfig",
    "plot_style_context",
]
