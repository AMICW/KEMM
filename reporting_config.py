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
import warnings

try:
    import matplotlib

    HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MPL = False

try:
    import scienceplots  # noqa: F401

    HAS_SCIENCEPLOTS = True
except ImportError:  # pragma: no cover - optional dependency
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
    """Matplotlib 通用排版参数。"""

    dpi: int = 320
    font_family: str = "DejaVu Sans"
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
    # 项目当前的安全默认风格：不依赖 SciencePlots，兼容性最好。
    "default": dict(
        dpi=320,
        font_family="DejaVu Sans",
        use_scienceplots=False,
        science_styles=("science", "no-latex"),
        title_size=14,
        label_size=11,
        tick_size=10,
        legend_size=9,
        line_width=2.1,
        emphasis_line_width=2.8,
        marker_size=5.6,
        scatter_size=34.0,
        grid_alpha=0.18,
        band_alpha=0.12,
        bar_alpha=0.9,
        figure_facecolor="#ffffff",
        axes_facecolor="#fcfcfd",
        grid_color="#d0d7de",
        spine_color="#9aa4b2",
    ),
    # 通用论文图风格：推荐投稿前的大多数结果图先从这个预设开始。
    "paper": dict(
        dpi=360,
        font_family="DejaVu Sans",
        use_scienceplots=True,
        science_styles=("science", "no-latex"),
        title_size=15,
        label_size=12,
        tick_size=10,
        legend_size=9,
        line_width=2.15,
        emphasis_line_width=2.9,
        marker_size=5.8,
        scatter_size=36.0,
        grid_alpha=0.16,
        band_alpha=0.10,
        bar_alpha=0.9,
        figure_facecolor="#ffffff",
        axes_facecolor="#fcfcfd",
        grid_color="#d0d7de",
        spine_color="#9aa4b2",
    ),
    # IEEE 风格：适合 benchmark 主线的对比图和排名图。
    "ieee": dict(
        dpi=380,
        font_family="DejaVu Sans",
        use_scienceplots=True,
        science_styles=("science", "ieee", "no-latex"),
        title_size=14,
        label_size=11,
        tick_size=9,
        legend_size=8,
        line_width=2.0,
        emphasis_line_width=2.7,
        marker_size=5.2,
        scatter_size=30.0,
        grid_alpha=0.14,
        band_alpha=0.10,
        bar_alpha=0.9,
        figure_facecolor="#ffffff",
        axes_facecolor="#ffffff",
        grid_color="#d0d7de",
        spine_color="#8c959f",
    ),
    # Nature 风格：更强调简洁和页面展示效果。
    "nature": dict(
        dpi=380,
        font_family="DejaVu Sans",
        use_scienceplots=True,
        science_styles=("science", "nature", "no-latex"),
        title_size=15,
        label_size=11,
        tick_size=9,
        legend_size=8,
        line_width=2.0,
        emphasis_line_width=2.7,
        marker_size=5.0,
        scatter_size=28.0,
        grid_alpha=0.10,
        band_alpha=0.08,
        bar_alpha=0.88,
        figure_facecolor="#ffffff",
        axes_facecolor="#ffffff",
        grid_color="#e5e7eb",
        spine_color="#9ca3af",
    ),
    # Thesis 风格：更适合大图、答辩和说明性较强的图板。
    "thesis": dict(
        dpi=320,
        font_family="DejaVu Sans",
        use_scienceplots=False,
        science_styles=("science", "no-latex"),
        title_size=16,
        label_size=12,
        tick_size=11,
        legend_size=10,
        line_width=2.3,
        emphasis_line_width=3.0,
        marker_size=6.2,
        scatter_size=38.0,
        grid_alpha=0.20,
        band_alpha=0.12,
        bar_alpha=0.92,
        figure_facecolor="#ffffff",
        axes_facecolor="#fcfcfd",
        grid_color="#d0d7de",
        spine_color="#9aa4b2",
    ),
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
    "paper": dict(trajectory_figsize=(11.0, 6.6), comparison_figsize=(14.4, 5.0), dashboard_figsize=(14.5, 10.2)),
    "ieee": dict(trajectory_figsize=(10.2, 6.0), comparison_figsize=(13.5, 4.6), dashboard_figsize=(13.2, 9.4)),
    "nature": dict(trajectory_figsize=(10.0, 6.0), comparison_figsize=(13.2, 4.6), dashboard_figsize=(13.8, 9.6)),
    "thesis": dict(trajectory_figsize=(12.0, 7.0), comparison_figsize=(15.0, 5.2), dashboard_figsize=(15.6, 11.0)),
}


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


def list_plot_presets() -> list[str]:
    """列出当前支持的命名风格预设。"""

    return sorted(PLOT_STYLE_PRESETS.keys())


def build_publication_style(preset: str = "default", **overrides) -> PublicationStyle:
    """按命名预设构造 PublicationStyle。

    你后续如果想统一调整论文图风格，优先改本文件顶部的 `PLOT_STYLE_PRESETS`。
    外部代码尽量只写：

    `build_publication_style("paper")`
    `build_publication_style("ieee", dpi=420)`
    """

    if preset not in PLOT_STYLE_PRESETS:
        raise ValueError(f"Unknown plot preset: {preset}. Available presets: {', '.join(list_plot_presets())}")
    payload = dict(PLOT_STYLE_PRESETS[preset])
    payload.update(overrides)
    return PublicationStyle(**payload)


def build_benchmark_plot_config(
    preset: str = "default",
    style_overrides: dict | None = None,
    **config_overrides,
) -> BenchmarkPlotConfig:
    """按命名预设构造 benchmark 图表配置。"""

    style = build_publication_style(preset, **(style_overrides or {}))
    payload = dict(BENCHMARK_PLOT_PRESETS.get(preset, {}))
    payload.update(config_overrides)
    payload["style"] = style
    return BenchmarkPlotConfig(**payload)


def build_ship_plot_config(
    preset: str = "default",
    style_overrides: dict | None = None,
    **config_overrides,
) -> ShipPlotConfig:
    """按命名预设构造 ship 图表配置。"""

    style = build_publication_style(preset, **(style_overrides or {}))
    payload = dict(SHIP_PLOT_PRESETS.get(preset, {}))
    payload.update(config_overrides)
    payload["style"] = style
    return ShipPlotConfig(**payload)


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


def _resolve_style_sheets(style: PublicationStyle) -> list[str]:
    """Resolve optional matplotlib style sheets before applying local rcParams."""

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


@contextmanager
def plot_style_context(style: PublicationStyle) -> Iterator[None]:
    """临时应用论文风格的 matplotlib rcParams。"""

    if not HAS_MPL:  # pragma: no cover - matplotlib absent
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
    "BenchmarkPlotConfig",
    "BENCHMARK_PLOT_PRESETS",
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
    "list_plot_presets",
    "plot_style_context",
]
