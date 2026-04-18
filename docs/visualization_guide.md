# 可视化指南

本文档说明 benchmark 与 ship 两条主线如何共用同一套论文风格配置，并解释当前 ship 图包的组织方式。

建议把这份文档当成“如何调图”的手册：

- 想知道命令怎么跑，去看 [run_commands.md](run_commands.md) 或 [how_to_run.md](how_to_run.md)
- 想知道每张图表达什么、图注怎么写，去看 [figure_catalog.md](figure_catalog.md)
- 想先看整个文档地图，去看 [README.md](README.md)

职责边界：

- `README.md` 负责总入口
- `docs/visualization_guide.md` 负责说明怎么调图、怎么换 preset、怎么开关 appendix/debug 图
- `docs/figure_catalog.md` 负责说明每张图表达什么、适合放在哪一节、正文和图注怎么写

---

## 1. 配置入口

统一配置文件：

- `reporting_config.py`

当前最重要的配置对象：

- `PublicationStyle`
- `BenchmarkPlotConfig`
- `ShipPlotConfig`

优先级原则：

1. 改模板、DPI、字体、SciencePlots 风格，优先改 `PublicationStyle`
2. 改 benchmark 图尺寸和配色，优先改 `BenchmarkPlotConfig`
3. 改 ship 图尺寸、knee 样式、障碍配色，优先改 `ShipPlotConfig`

---

## 2. Plot Preset

当前内置 preset：

- `default`
- `paper`
- `ieee`
- `nature`
- `thesis`

调用方式：

```python
from reporting_config import build_ship_plot_config

plot_config = build_ship_plot_config("paper")
```

命令行方式：

```powershell
python ship_simulation/run_report.py --plot-preset paper
python -m apps.benchmark_runner --plot-preset ieee
```

把命令和用途直接对应起来就是：

- 运行 `python ship_simulation/run_report.py --plot-preset paper`：生成 ship 主线默认论文风格图表，适合日常检查与论文初稿插图。
- 运行 `python ship_simulation/run_report.py --plot-preset ieee`：生成更紧凑的 IEEE 风格 ship 图表。
- 运行 `python ship_simulation/run_report.py --plot-preset nature`：生成更偏展示型的 ship 图表。
- 运行 `python ship_simulation/run_report.py --plot-preset thesis`：生成更适合毕业论文或长文档的 ship 图表。
- 运行 `python -m apps.benchmark_runner --plot-preset paper`：生成 benchmark 默认论文风格图表。
- 运行 `python -m apps.benchmark_runner --plot-preset ieee`：生成适合 IEEE 排版的 benchmark 图表。

---

## 3. SciencePlots 与字体

### 3.1 SciencePlots

`PublicationStyle` 里包含：

- `use_scienceplots`
- `science_styles`

如果安装了 `SciencePlots`，系统会自动应用样式表；如果没有安装，会自动回退到内置 matplotlib 风格，不会中断报告生成。

示例：

```powershell
python ship_simulation/run_report.py --science-style science,ieee,no-latex
```

直接理解成：

- 运行 `python ship_simulation/run_report.py --science-style science,ieee,no-latex`：不改其他配置，只手动把 SciencePlots 样式切成 `science + ieee + no-latex`。
- 运行 `python ship_simulation/run_report.py --plot-preset paper --science-style science,nature,no-latex`：保留 `paper` 预设的其他尺寸和字体设定，但把样式表强制切成 `nature` 风格。
- 运行 `python -m apps.benchmark_runner --quick --plot-preset paper --science-style science,ieee,no-latex`：对 benchmark 主线也做同样的样式覆盖。

### 3.2 中文字体 fallback

`PublicationStyle.chinese_font_fallback` 已内置：

- `Microsoft YaHei`
- `SimHei`
- `Noto Sans CJK SC`
- `Arial Unicode MS`

如果需要替换字体，不要去每个绘图函数里改 `fontname`，直接改这里。

---

## 4. 交互式输出

当前项目支持两种交互式结果：

- `*.fig.pickle`
  由 matplotlib figure bundle 构成，可重新打开、缩放、旋转、再保存
- `.html`
  当前 ship 的 3D 图会额外导出 Plotly HTML，可直接在浏览器里拖拽旋转视角

最常用命令：

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
python -m apps.benchmark_runner --quick --interactive-figures
python -m ship_simulation.visualization.figure_viewer ship_simulation/outputs/report_YYYYMMDD_HHMMSS/figures/crossing_pareto3d.fig.pickle
python -m ship_simulation.visualization.figure_viewer ship_simulation/outputs/report_YYYYMMDD_HHMMSS/figures/crossing_pareto3d.fig.pickle --elev 25 --azim 140 --save-path ship_simulation/outputs/report_YYYYMMDD_HHMMSS/figures/crossing_pareto3d_view.png --no-show
```

直接理解成：

- 运行 `python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html`：ship 报告在导出 PNG 的同时，只为 `pareto3d / spatiotemporal` 导出可重新打开的 `*.fig.pickle`，并额外导出浏览器可旋转的 `.html`。
- 运行 `python -m apps.benchmark_runner --quick --interactive-figures`：benchmark 默认图也都会额外导出 `*.fig.pickle`。
- 运行 `python -m ship_simulation.visualization.figure_viewer <bundle.fig.pickle>`：重新打开一个已保存的 matplotlib figure bundle。
- 运行 `python -m ship_simulation.visualization.figure_viewer <bundle.fig.pickle> --elev 25 --azim 140 --save-path <new.png> --no-show`：不弹窗口，直接把 3D 图切到新视角后另存为新的静态图。

当前命名规则：

- `crossing_pareto3d.png`：论文静态图
- `crossing_pareto3d.fig.pickle`：可重新打开的交互版
- `crossing_pareto3d.html`：浏览器交互版

---

## 5. ShipPlotConfig 关键字段

当前 ship 图表最常改的字段包括：

- `own_ship_color`
- `baseline_color`
- `third_algo_color`
- `scalar_cmap`
- `pareto_cmap`
- `obstacle_facecolor`
- `obstacle_edgecolor`
- `knee_color`
- `knee_marker`
- `knee_size`
- `overlay_figsize`
- `route_panel_figsize`
- `snapshot_figsize`
- `spatiotemporal_figsize`
- `control_figsize`
- `pareto3d_figsize`
- `pareto_projection_figsize`
- `parallel_figsize`
- `radar_figsize`
- `violin_figsize`
- `convergence_figsize`
- `risk_breakdown_figsize`
- `safety_envelope_figsize`
- `statistics_figsize`
- `dashboard_figsize`
- `inset_zoom_alpha`
- `velocity_arrow_scale`
- `appendix_plots`
- `interactive_figures`
- `interactive_html`

ship 图表默认将所有归一化指标画成“越高越好”的方向，便于阅读 radar 和 parallel coordinates。

---

## 6. BenchmarkPlotConfig 关键字段

benchmark 侧常改字段包括：

- `colors`
- `markers`
- `highlight_color`
- `best_outline_color`
- `heatmap_cmap`
- `significance_cmap`
- `pairwise_cmap`
- `metric_panel_width`
- `metrics_grid_height`
- `rank_bar_width`
- `rank_bar_height`
- `dashboard_width`
- `dashboard_height`
- `appendix_plots`
- `interactive_figures`

---

## 7. Ship 核心图包

当前 ship 完整报告的默认图包是：

- 每个场景 `19` 张
- 全局图 `2` 张
- 总计 `4 * 19 + 2 = 78` 张

全局图：

- `scenario_gallery.png`
- `route_bundle_gallery.png`

每场景图：

1. `*_environment_overlay.png`
2. `*_route_planning_panel.png`
3. `*_change_timeline.png`
4. `*_snapshots.png`
5. `*_spatiotemporal.png`
6. `*_control_timeseries.png`
7. `*_pareto3d.png`
8. `*_pareto_projection.png`
9. `*_risk_breakdown.png`
10. `*_safety_envelope.png`
11. `*_parallel.png`
12. `*_radar.png`
13. `*_convergence.png`
14. `*_distribution.png`
15. `*_run_statistics.png`
16. `*_dashboard.png`
17. `*_runtime_tradeoff.png`
18. `*_decision_projection.png`
19. `*_operator_allocation.png`

这些图都由 `ship_simulation/visualization/report_plots.py` 统一生成。

附录图：

- `appendix_objective_bars.png`
- `appendix_risk_bars.png`

只在 `appendix_plots=True` 或 `--appendix-plots` 时导出。

---

## 8. 新增报告产物

最新 ship 报告结构里，除了图之外还会额外导出：

- 严格可比组（`--strict-comparable`）会追加 `*_matched` 算法分组
- 统计显著性结果：
  - `raw/statistical_tests.json`
  - `raw/statistical_tests.csv`
  - `reports/statistical_significance.md`
- 鲁棒性扫描（`--robustness-sweep`）结果：
  - `raw/robustness_runs.csv`
  - `raw/robustness_curve.csv`
  - `raw/robustness_summary.json`
  - `reports/robustness_sweep.md`
  - `figures/robustness_success_curve.png`（启用渲染时）

这部分不是“新加一批默认主图”，而是报告层额外产物。论文写作时建议把它们作为统计支撑与鲁棒性支撑，不和主图叙事混在同一段里。

---

## 9. Benchmark 核心图包

benchmark 侧当前默认保留：

- `metrics_grid`
- `rank_bar`
- `heatmap`
- `igd_time`
- `operator_ratios`
- `response_quality`
- `prediction_confidence`
- `wilcoxon`
- `pairwise_wins`
- `change_dashboard`

统一入口：

- `apps/reporting/benchmark_visualization.py`
- `apps.benchmark_runner`

---

## 10. 常见修改场景

### 10.1 只想统一投稿风格

改：

- `reporting_config.py`

不要改：

- 各个 `plt.subplots(..., dpi=...)`
- 各个 `ax.tick_params(...)`

如果你只是临时切风格，不需要改代码，直接跑下面这些命令就够了：

```powershell
python ship_simulation/run_report.py --plot-preset paper
python ship_simulation/run_report.py --plot-preset ieee
python ship_simulation/run_report.py --plot-preset nature
python ship_simulation/run_report.py --plot-preset thesis
python -m apps.benchmark_runner --quick --plot-preset paper
python -m apps.benchmark_runner --quick --plot-preset ieee
```

这些命令分别对应：

- `paper`：默认论文风格
- `ieee`：更紧凑的投稿风格
- `nature`：更强调视觉展示
- `thesis`：更适合长文档，不强依赖 SciencePlots 版式

### 10.2 想增加 ship 新图

改：

- `ship_simulation/visualization/report_plots.py`
- `ship_simulation/visualization/__init__.py`
- 如需尺寸/颜色参数，再补 `ShipPlotConfig`

### 10.3 想清理 benchmark 重复图

改：

- `apps/reporting/benchmark_visualization.py`
- `apps/benchmark_runner.py`
- 如需默认开关，再补 `BenchmarkPlotConfig`

---

## 11. 推荐调用方式

### 11.1 Python 调用

```python
from reporting_config import build_ship_plot_config
from ship_simulation.run_report import generate_report_with_config
from ship_simulation.config import build_default_config, build_default_demo_config

plot_config = build_ship_plot_config("paper", appendix_plots=False)
generate_report_with_config(
    config=build_default_config(),
    demo_config=build_default_demo_config(),
    plot_config=plot_config,
)
```

### 11.2 CLI 调用

```powershell
python ship_simulation/run_report.py --plot-preset paper --scenarios crossing overtaking
python ship_simulation/run_report.py --plot-preset ieee --science-style science,ieee,no-latex
python ship_simulation/run_report.py --workers 4 --strict-comparable
python ship_simulation/run_report.py --workers 4 --robustness-sweep --robustness-levels 0,0.25,0.5,0.75,1.0 --robustness-scenarios crossing overtaking harbor_clutter
python -m apps.benchmark_runner --quick --plot-preset paper --appendix-plots
```

---

## 12. 维护原则

- 图表层只消费结构化输入，不直接读算法私有字段
- 新的风格参数先放 `reporting_config.py`
- 默认主图优先论文表达，不要把低价值调试图塞回默认输出
- 兼容层函数名可以保留，但默认语义允许升级到新图包
