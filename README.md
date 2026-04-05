# KEMM: Dynamic Multi-Objective Optimization + Ship Trajectory Simulation

一个面向研究工作的 Python 项目，统一维护两条主线：

1. `benchmark` 主线
   使用动态多目标标准测试问题验证 KEMM 与多种基线算法。
2. `ship_simulation` 主线
   使用纯代码生成的偏现实船舶会遇场景，验证 KEMM 在物理语义轨迹规划问题上的效果。

本仓库的目标不是做单次实验脚本，而是提供一个可维护、可复现、可扩展、可继续演化的研究型代码库。

---

## 1. 项目亮点

- 双主线统一：benchmark 理论验证 + ship 物理语义验证
- KEMM 已拆成可维护结构：主流程、子模块、adapter、报告层明确分层
- ship 主线已升级为滚动重规划 episode，而不是单次静态轨迹 demo
- ship 场景已支持：静态障碍、动态交通体、环境标量场/矢量场、COLREG 角色标签
- 图表系统支持统一论文风格配置，默认优先兼容 SciencePlots
- 文档体系同时面向：GitHub 访客、新开发者、新 AI 助手、论文写作

---

## 2. 仓库示例图

### 2.1 Benchmark 结果预览

![Benchmark Preview](docs/images/benchmark_preview.png)

### 2.2 Ship 仿真结果预览

![Ship Preview](docs/images/ship_preview.png)

---

## 3. 仓库结构

```text
.
├── AGENTS.md
├── README.md
├── reporting_config.py
├── requirements.txt
├── requirements-dev.txt
├── docs/
│   ├── ai_developer_handoff.md
│   ├── codebase_reference.md
│   ├── environment_setup.md
│   ├── figure_catalog.md
│   ├── formula_audit.md
│   ├── kemm_reference.md
│   ├── ship_simulation_reference.md
│   ├── visualization_guide.md
│   └── images/
├── apps/
│   ├── benchmark_runner.py
│   ├── ship_runner.py
│   └── reporting/
├── kemm/
├── ship_simulation/
├── tests/
├── run_experiments.py
├── benchmark_algorithms.py
└── visualization.py
```

---

## 4. 真实实现与兼容层

### 4.1 真实实现

优先关注这些文件和目录：

- `apps/benchmark_runner.py`
- `apps/ship_runner.py`
- `apps/reporting/benchmark_visualization.py`
- `kemm/algorithms/*`
- `kemm/adapters/*`
- `kemm/benchmark/*`
- `kemm/core/*`
- `kemm/reporting/*`
- `ship_simulation/*`
- `reporting_config.py`

### 4.2 兼容层

这些文件主要用于保留旧导入路径和旧命令，不应作为新逻辑的首选落点：

- `run_experiments.py`
- `benchmark_algorithms.py`
- `visualization.py`
- `adaptive_operator.py`
- `compressed_memory.py`
- `geodesic_flow.py`
- `pareto_drift.py`

---

## 5. 环境与安装

推荐 Python 版本：`3.10` 到 `3.12`，兼容目标 `3.9+`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

如果需要运行测试：

```powershell
pip install -r requirements-dev.txt
```

更详细环境说明见：

- [docs/environment_setup.md](docs/environment_setup.md)

---

## 6. 快速开始

如果你不想记复杂命令，只记下面这 4 条就够了：

```powershell
python -m apps.ship_runner
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
python run_experiments.py --quick
```

它们分别表示：

- `python -m apps.ship_runner`：走 ship 的兼容应用入口，执行一次默认演示
- `python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html`：生成 ship 快速交互报告，除了 PNG 还会导出 ship 侧 3D 图的 `*.fig.pickle` 和 `.html`
- `python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"`：跑一次 ship 单次演示，不生成整套报告
- `python run_experiments.py --quick`：跑 benchmark 快速验证

如果只想先看结果，优先跑 `python ship_simulation/run_report.py --quick --scenarios crossing`。

### 6.1 benchmark 快速验证

```powershell
python run_experiments.py --quick --plot-preset paper
python -m apps.benchmark_runner --quick --plot-preset ieee
```

说明：

- 运行 `python run_experiments.py --quick --plot-preset paper` 会用旧兼容入口执行一轮 benchmark 快速验证，并以 `paper` 预设导出论文风格图表。
- 运行 `python -m apps.benchmark_runner --quick --plot-preset ieee` 会直接使用真实 benchmark 入口，并按 `ieee` 风格生成结果，更适合当前主线回归与出图。

### 6.2 benchmark 完整实验

```powershell
python run_experiments.py --full --plot-preset paper
```

说明：

- 运行 `python run_experiments.py --full --plot-preset paper` 会执行完整 benchmark 实验，并按 `paper` 预设输出完整报告与图表。

### 6.3 ship 单次演示

```powershell
python -m apps.ship_runner
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
```

说明：

- 运行 `python -m apps.ship_runner` 会走 ship 的兼容命令入口，执行一次默认演示。
- 运行 `python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"` 会直接调用 ship 主线代码，在 `crossing` 场景下用 `KEMM` 跑一次不弹动画窗口的演示。

### 6.4 ship 批量报告

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
python ship_simulation/run_report.py --plot-preset ieee --science-style science,ieee,no-latex
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
```

说明：

- 运行 `python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper` 会快速生成一个 `crossing` 场景、每算法 1 次运行的 ship 报告，适合 smoke test 和看默认论文图包。
- 运行 `python ship_simulation/run_report.py --plot-preset ieee --science-style science,ieee,no-latex` 会按 `ieee` 预设并显式指定 `SciencePlots` 样式 tuple 生成 ship 报告，适合投稿前试版式。
- 运行 `python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html` 会在 PNG 之外为 ship 的 3D 图额外导出可交互的 `*.fig.pickle`，并为 `pareto3d / spatiotemporal` 额外导出 `.html`，便于自己旋转视角后再保存。

### 6.5 基础测试

```powershell
python -m unittest discover -s tests -v
```

说明：

- 运行 `python -m unittest discover -s tests -v` 会执行当前仓库的完整单元测试与 smoke 测试。

---

## 7. 输出目录约定

### 7.1 benchmark

```text
benchmark_outputs/
└── benchmark_YYYYMMDD_HHMMSS/
    ├── figures/
    ├── raw/
    └── reports/
```

### 7.2 ship

```text
ship_simulation/outputs/
└── report_YYYYMMDD_HHMMSS/
    ├── figures/
    ├── raw/
    └── reports/
```

两条主线保持统一输出结构，便于归档、写论文和批量分析。

---

## 8. Ship 主线现在做了什么

ship 主线已经不是单次静态规划 demo，而是：

- 近海增强场景：对遇、交叉、追越、高密障碍受限海域穿越
- 静态障碍：岛礁、禁航区、水道边界
- 动态交通体：多目标船
- 环境层：标量风险场 + 矢量流场
- 风险模型：船舶域 + DCPA/TCPA + 障碍侵入 + 环境暴露
- 执行方式：滚动重规划 episode
- 输出结果：最终执行轨迹、每步局部前沿、knee point、快照、控制时序、统计指标

核心代码位置：

- 场景：`ship_simulation/scenario/*`
- 问题定义：`ship_simulation/optimizer/problem.py`
- episode 执行层：`ship_simulation/optimizer/episode.py`
- KEMM 适配：`ship_simulation/optimizer/kemm_solver.py`
- NSGA-style 基线：`ship_simulation/optimizer/baseline_solver.py`
- 报告图：`ship_simulation/visualization/report_plots.py`

---

## 9. 可视化与论文图包

图表系统分三层：

1. 公共风格配置层：`reporting_config.py`
2. benchmark 图表层：`apps/reporting/benchmark_visualization.py`
3. ship 图表层：`ship_simulation/visualization/report_plots.py`

当前推荐的 ship 论文图包包括：

- 环境场/矢量场叠加轨迹图
- 高密障碍海域路线规划主图
- 动态避碰时空快照图
- 3D 时空轨迹图
- 动力学/控制多子图时序图
- 3D Pareto Front + Knee Point
- 2D Pareto 投影前沿曲线组
- 风险分解时间序列图
- 安全包络图
- Parallel Coordinates
- Radar Chart
- 带阴影误差带的收敛曲线
- Violin Plot
- repeated-run 安全统计图
- Summary dashboard

如果你想让输出图不只是固定 PNG，而是可以后续继续交互：

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
python -m apps.benchmark_runner --quick --interactive-figures
python -m ship_simulation.visualization.figure_viewer ship_simulation/outputs/report_YYYYMMDD_HHMMSS/figures/crossing_pareto3d.fig.pickle
python -m ship_simulation.visualization.figure_viewer ship_simulation/outputs/report_YYYYMMDD_HHMMSS/figures/crossing_pareto3d.fig.pickle --elev 25 --azim 140 --save-path ship_simulation/outputs/report_YYYYMMDD_HHMMSS/figures/crossing_pareto3d_view.png --no-show
```

这些命令分别表示：

- `--interactive-figures`：ship 侧只为 `pareto3d / spatiotemporal` 额外导出可重新打开的 `*.fig.pickle`；benchmark 侧仍按原逻辑导出
- `--interactive-html`：为当前支持的 3D ship 图额外导出浏览器可旋转的 `.html`
- `figure_viewer`：重新打开保存过的 figure bundle；对 3D 图可指定 `--elev` 和 `--azim` 后重新另存一个新视角

详细说明见：

- [docs/visualization_guide.md](docs/visualization_guide.md)
- [docs/figure_catalog.md](docs/figure_catalog.md)

如果你想快速切换不同论文风格，最常用的命令就是：

```powershell
python ship_simulation/run_report.py --plot-preset paper
python ship_simulation/run_report.py --plot-preset ieee
python ship_simulation/run_report.py --plot-preset nature
python ship_simulation/run_report.py --plot-preset thesis
python -m apps.benchmark_runner --quick --plot-preset paper
python -m apps.benchmark_runner --quick --plot-preset ieee
```

这些命令的含义是：

- `paper`：默认论文风格，适合日常出图与仓库展示
- `ieee`：更紧凑，适合 IEEE 类版式
- `nature`：更强调视觉展示
- `thesis`：不强依赖 SciencePlots，适合长文档或毕业论文

---

## 10. 改算法时应该改哪里

- 改 KEMM 主流程：`kemm/algorithms/kemm.py`
- 改 benchmark-only prior：`kemm/adapters/benchmark.py`
- 改 KEMM 子模块：`kemm/core/*.py`
- 改 ship 问题定义：`ship_simulation/optimizer/problem.py`
- 改 ship episode：`ship_simulation/optimizer/episode.py`
- 改图表风格：`reporting_config.py`
- 改 ship 论文图：`ship_simulation/visualization/report_plots.py`
- 改 benchmark 图：`apps/reporting/benchmark_visualization.py`

不要优先改根目录 legacy 文件。

---

## 11. 文档索引

### 总览 / 架构

- [AGENTS.md](AGENTS.md)
- [docs/codebase_reference.md](docs/codebase_reference.md)
- [docs/ai_developer_handoff.md](docs/ai_developer_handoff.md)

### KEMM 主线

- [docs/kemm_reference.md](docs/kemm_reference.md)
- [docs/formula_audit.md](docs/formula_audit.md)

### Ship 主线

- [docs/ship_simulation_reference.md](docs/ship_simulation_reference.md)

### 图表与论文写作

- [docs/visualization_guide.md](docs/visualization_guide.md)
- [docs/figure_catalog.md](docs/figure_catalog.md)

---

## 12. 当前已知说明

- `SciencePlots` 是可选依赖；未安装时会自动回退到内置 matplotlib 风格。
- Windows 下运行 benchmark 或 ship 报告时，末尾可能仍出现 `joblib/loky -> wmic` 的环境警告，但一般不影响结果生成。
- ship 主线当前仍是增强 Nomoto，而不是 MMG；偏现实性来自场景、风险建模、环境层和滚动重规划，而不是高保真水动力学本体。
