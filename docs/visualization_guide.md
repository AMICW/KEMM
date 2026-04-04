# 可视化配置指南

本文档说明 benchmark 与 ship 两条主线的图表如何统一成“可调参数、可复用、可投稿”的风格层。

## 1. 设计目标

当前图表系统有两个明确目标：

1. 默认输出应接近论文插图级别。
2. 所有关键风格参数都应通过配置对象显式调整，而不是散落在绘图函数内部。

公共配置文件是：

- `reporting_config.py`

其中定义了：

- `PublicationStyle`
- `BenchmarkPlotConfig`
- `ShipPlotConfig`

## 2. 公共风格层

### 2.1 `PublicationStyle`

这是一层最通用的 matplotlib 风格控制，主要负责：

- `dpi`
- `font_family`
- `title_size`
- `label_size`
- `tick_size`
- `legend_size`
- `line_width`
- `emphasis_line_width`
- `marker_size`
- `scatter_size`
- `grid_alpha`
- `band_alpha`
- `bar_alpha`
- `figure_facecolor`
- `axes_facecolor`
- `grid_color`
- `spine_color`

如果你要统一改整套图的排版，优先从这里入手。

## 3. benchmark 图表配置

### 3.1 `BenchmarkPlotConfig`

这个对象在 `apps/reporting/benchmark_visualization.py` 中使用，负责：

- 算法颜色映射
- 算法 marker 映射
- KEMM 高亮色
- 最优条形轮廓色
- heatmap 颜色图
- 显著性 heatmap 颜色图
- pairwise win matrix 颜色图
- 各类图的默认尺寸

关键字段：

- `colors`
- `markers`
- `highlight_color`
- `best_outline_color`
- `heatmap_cmap`
- `significance_cmap`
- `pairwise_cmap`
- `metrics_grid_height`
- `metric_panel_width`
- `rank_bar_width`
- `rank_bar_height`
- `dashboard_width`
- `dashboard_height`

### 3.2 使用方式

```python
from apps.reporting import BenchmarkFigurePayload, BenchmarkPlotConfig, PublicationStyle, generate_all_figures

plot_config = BenchmarkPlotConfig(
    style=PublicationStyle(dpi=360, title_size=15, label_size=12),
    highlight_color="#8b1e3f",
    metric_panel_width=4.8,
)

payload = BenchmarkFigurePayload(
    results=results,
    problems=problems,
    igd_curves=igd_curves,
    diagnostics=diagnostics,
    plot_config=plot_config,
)

generate_all_figures(payload=payload, output_prefix="benchmark_outputs/custom/benchmark")
```

## 4. ship 图表配置

### 4.1 `ShipPlotConfig`

这个对象在 `ship_simulation/visualization/report_plots.py` 和 `ship_simulation/run_report.py` 中使用，负责：

- KEMM 与 baseline 的主配色
- 风险阈值颜色
- Pareto 散点 colormap
- 轨迹线宽、目标船线宽
- 各类 ship 图的默认尺寸

关键字段：

- `own_ship_color`
- `baseline_color`
- `risk_threshold_color`
- `pareto_cmap`
- `trajectory_width`
- `target_width`
- `trajectory_figsize`
- `comparison_figsize`
- `convergence_figsize`
- `pareto_figsize`
- `time_series_figsize`
- `dashboard_figsize`
- `risk_threshold`

### 4.2 使用方式

```python
from reporting_config import PublicationStyle, ShipPlotConfig
from ship_simulation.run_report import generate_report

plot_config = ShipPlotConfig(
    style=PublicationStyle(dpi=360, title_size=15, label_size=12),
    own_ship_color="#0f6cbd",
    baseline_color="#b54708",
    pareto_cmap="viridis",
    dashboard_figsize=(15.5, 10.5),
)

generate_report(plot_config=plot_config)
```

## 5. 为什么这样设计

这样设计的核心价值不是“漂亮”，而是：

1. 图表层不再把风格参数硬编码在函数内部。
2. benchmark 与 ship 两条主线可以共用同一套论文排版风格。
3. 新的 AI 助手或开发者只看 `reporting_config.py` 就能定位如何改风格。
4. 投稿前若需要按期刊模板调整尺寸、字体、DPI，只改配置对象即可。

## 6. 修改建议

如果你要改图表：

- 改整体风格：`PublicationStyle`
- 改 benchmark 图表颜色和尺寸：`BenchmarkPlotConfig`
- 改 ship 图表颜色和尺寸：`ShipPlotConfig`
- 改某张图的具体绘制逻辑：
  - benchmark：`apps/reporting/benchmark_visualization.py`
  - ship：`ship_simulation/visualization/report_plots.py`
