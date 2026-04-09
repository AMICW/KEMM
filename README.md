# KEMM: Dynamic Multi-Objective Optimization + Ship Trajectory Simulation

一个面向研究工作的 Python 项目，统一维护两条主线：

1. `benchmark` 主线
   使用动态多目标标准测试问题验证 KEMM 与多种基线算法。
2. `ship_simulation` 主线
   使用纯代码生成的偏现实船舶会遇场景，验证 KEMM 在物理语义轨迹规划问题上的效果。

本仓库的目标不是做单次实验脚本，而是提供一个可维护、可复现、可扩展、可继续演化的研究型代码库。

---

## 1. 项目亮点

- **约束支配排序 (CDNSGA - Constrained Domination)**：原生融合 Deb 的可行性规则 (Feasibility Rules)，将碰撞和边界惩罚彻底解耦至独立的约束标量 (CV)，极大保护了目标空间帕累托前沿的纯粹性，满足顶刊对物理语义优化的严格要求。
- **纯粹的上下文多臂老虎机 (Contextual MAB - UCB1)**：摒弃传统魔法参调控，基于环境漂移强度 (`change_magnitude`) 构建多状态桶 Contextual Bandits，完全通过统计学 UCB 自治收敛算子最佳分配策略。
- **双主线统一**：benchmark 理论验证 + ship 物理语义滚动验证。
- **benchmark 主线**：已补齐 `MIGD` 主表、四大核心自适应进化模块 (Memory/Predict/Transfer/Reinit) 消融与兼容比较。
- **ship 仿真主线**：集成滚动避碰重规划 (Episodes)，涵盖单/多船相遇、静态岛礁/高密障碍物受限区、COLREG规则与任意动态环境矢量场。
- **极其严苛的模块解耦与出图**：报告层、算法内核明确剥离。内置基于 SciencePlots/IEEE 的多场景三维、时序、流场映射与解集轨迹的可视化和交互分析。
- 文档体系同时面向：GitHub 访客、新开发者、新 AI 助手、论文写作 (涵盖详尽命令备忘与架构审视)。

---

## 2. 仓库示例图

### 2.1 Benchmark 结果预览

![Benchmark Preview](./docs/images/benchmark_preview.png)

### 2.2 Ship 仿真结果预览

![Ship Preview](./docs/images/ship_preview.png)

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
│   ├── how_to_run.md
│   ├── kemm_reference.md
│   ├── run_commands.md
│   ├── ship_experiment_playbook.md
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
- [docs/how_to_run.md](docs/how_to_run.md)
- [docs/ship_experiment_playbook.md](docs/ship_experiment_playbook.md)

---

## 6. 快速开始

如果你以后忘了命令，优先看：

- [docs/how_to_run.md](docs/how_to_run.md)

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
- benchmark 报告默认会额外跑三组动态设置 `(n_t, τ_t) = (5,10), (10,10), (10,20)`，导出论文风格的 `benchmark_migd_table.png`，并附带关闭 benchmark prior 的 KEMM 四模块消融图 `benchmark_ablation.png`。
- benchmark 主报告里的关键原始文件是 `raw/paper_table_metrics.csv` 与 `raw/ablation_delta_metrics.csv`，前者对应论文主表，后者对应相对 `KEMM-Full` 的消融退化百分比。

### 6.2 benchmark 完整实验

```powershell
python run_experiments.py --full --plot-preset paper
python -m apps.benchmark_runner --full --plot-preset paper
```

说明：

- 运行 `python run_experiments.py --full --plot-preset paper` 会执行完整 benchmark 实验，并按 `paper` 预设输出完整报告与图表。
- 运行 `python -m apps.benchmark_runner --full --plot-preset paper` 会直接使用 benchmark 真实入口。

### 6.3 benchmark 中等规模实验

当前命令行没有内置 `--medium` 选项。

如果你想跑一组介于 `quick` 和 `full` 之间的 benchmark，直接看：

- [docs/how_to_run.md](docs/how_to_run.md)
- [docs/run_commands.md](docs/run_commands.md)

里面已经给出了一段不修改源码文件的 PowerShell 临时脚本，默认采用：

- `N_RUNS = 3`
- `N_CHANGES = 7`
- `GENS_PER_CHANGE = 15`

### 6.4 ship 单次演示

```powershell
python -m apps.ship_runner
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
```

说明：

- 运行 `python -m apps.ship_runner` 会走 ship 的兼容命令入口，执行一次默认演示。
- 运行 `python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"` 会直接调用 ship 主线代码，在 `crossing` 场景下用 `KEMM` 跑一次不弹动画窗口的演示。

### 6.5 ship 批量报告与完整物理测试

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
python ship_simulation/run_report.py
python ship_simulation/run_report.py --scenarios harbor_clutter --experiment-profile shock --n-runs 3 --plot-preset paper
python ship_simulation/run_report.py --plot-preset ieee --science-style science,ieee,no-latex
python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html
```

说明：

- 运行 `python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper` 会快速生成一个 `crossing` 场景、每算法 1 次运行的 ship 报告，适合 smoke test 和看默认论文图包。
- 运行 `python ship_simulation/run_report.py` 才是 ship 主线的完整物理测试报告入口。默认会跑 `head_on / crossing / overtaking / harbor_clutter` 四个场景，以及 `kemm / nsga_style / random` 三种算法。
- 运行 `python ship_simulation/run_report.py --scenarios harbor_clutter --experiment-profile shock --n-runs 3 --plot-preset paper` 会启用突变型动态实验 profile，更适合验证 KEMM 在 closure shock 下的响应能力。
- 运行 `python ship_simulation/run_report.py --plot-preset ieee --science-style science,ieee,no-latex` 会按 `ieee` 预设并显式指定 `SciencePlots` 样式 tuple 生成 ship 报告，适合投稿前试版式。
- 运行 `python ship_simulation/run_report.py --quick --scenarios crossing --interactive-figures --interactive-html` 会在 PNG 之外为 ship 的 3D 图额外导出可交互的 `*.fig.pickle`，并为 `pareto3d / spatiotemporal` 额外导出 `.html`，便于自己旋转视角后再保存。
- ship 的 Plotly 3D HTML 现在会附带更完整的 hover 信息，例如 Pareto 点的加权分数、平均速度和首个 waypoint 摘要，以及时空轨迹上的速度、航向、风险和净空。

### 6.6 基础测试

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

- 多场景环境总览图
- 多场景最终轨迹带对比图
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

如果你想统一修改 ship 图的视觉主题，优先改 [reporting_config.py](reporting_config.py) 里的 `ShipPlotConfig`。现在颜色、卡片、图例和底图强度都已经集中到这一处，不需要再去每个绘图函数里找硬编码。最常改的是：

- `own_ship_color / baseline_color / third_algo_color`
- `start_marker_color / goal_marker_color / traffic_marker_color`
- `panel_facecolor / legend_facecolor / legend_edgecolor`
- `card_facecolor / card_edgecolor / card_alpha`
- `scalar_field_floor_quantile / scalar_field_alpha / vector_field_alpha`
- `dashboard_width_ratios / dashboard_height_ratios`

如果你想直接改 ship 场景生成参数，优先改：

- `ship_simulation/config.py` 里的 `ProblemConfig.scenario_generation`

例如：

```python
from ship_simulation.config import build_default_config

config = build_default_config()
config.scenario_generation.harbor_clutter.circular_obstacle_limit = 5
config.scenario_generation.harbor_clutter.polygon_obstacle_limit = 2
config.scenario_generation.harbor_clutter.target_limit = 2
config.scenario_generation.harbor_clutter.circular_radius_scale = 0.55
config.scenario_generation.harbor_clutter.channel_width_scale = 1.20
```

这组参数会直接影响 `harbor_clutter` 的障碍数量、障碍尺度、目标船数量和通航窗口宽度。

如果你想把场景改成“同一 family 下的可重复变体”，现在也不用回到 `generator.py` 里改硬编码。优先改：

- `family_name`
- `scenario_seed`
- `difficulty_scale`
- `geometry_jitter_m`
- `traffic_heading_jitter_deg`
- `current_direction_jitter_deg`

例如：

```python
config.scenario_generation.crossing.family_name = "crossing_family_a"
config.scenario_generation.crossing.scenario_seed = 17
config.scenario_generation.crossing.difficulty_scale = 1.20
config.scenario_generation.crossing.geometry_jitter_m = 120.0
config.scenario_generation.crossing.traffic_heading_jitter_deg = 6.0
config.scenario_generation.crossing.current_direction_jitter_deg = 8.0
```

这样生成的场景仍然保持 `crossing` 的语义，但会输出同 family 的可复现实验变体；对应 seed 和 family 会写进 `scenario_catalog.json`。

如果你想直接启用面向 KEMM 机制验证的动态实验 profile，优先改：

- `ship_simulation/config.py` 里的 `ProblemConfig.experiment`

例如：

```python
from ship_simulation.config import apply_experiment_profile, build_default_config

config = build_default_config()
apply_experiment_profile(config, "shock")
```

当前内置 profile：

- `baseline`
- `drift`
- `shock`
- `recurring_harbor`

这些 profile 会直接影响滚动重规划过程中的环境漂移、目标船意图变化和通道 closure 扰动；对应的解释图是每个场景新增的 `*_change_timeline.png`。

如果你想直接调 ship 侧 KEMM 的真实机制开关和预算，优先改：

- `ship_simulation/config.py` 里的 `DemoConfig.kemm`
- `ship_simulation/config.py` 里的 `DemoConfig.kemm.runtime`

其中：

- `DemoConfig.kemm.pop_size / generations / seed` 控制 ship 侧运行预算
- `DemoConfig.kemm.use_change_response` 控制滚动重规划时是否调用 KEMM 的 change response
- `DemoConfig.kemm.runtime.enable_memory / enable_prediction / enable_transfer / enable_adaptive` 对应真实 KEMM 模块开关

例如：

```python
from ship_simulation.config import build_default_demo_config

demo = build_default_demo_config()
demo.kemm.generations = 28
demo.kemm.use_change_response = True
demo.kemm.runtime.enable_prediction = True
demo.kemm.runtime.enable_transfer = False
```

当前 ship 主线里，同一个 episode 内会复用同一个 KEMM 求解器 session，而不是每个 replanning step 都重新新建算法对象。这意味着 `memory / prediction / transfer / adaptive` 的状态会沿滚动重规划继续积累，更接近真正的动态优化语义。

如果你想调整 ship 报告里到底比较哪些算法，优先改：

- `ship_simulation/config.py` 里的 `DemoConfig.report_algorithms`

例如：

```python
demo = build_default_demo_config()
demo.report_algorithms = ("kemm", "random")
```

报告原始目录还会额外导出：

- `raw/figure_manifest.json`
- `raw/representative_runs.json`

其中：

- `figure_manifest.json` 说明默认理论导出了哪些图
- `representative_runs.json` 记录每个场景、每个算法被选中的代表性 run

当前 summary 表和 CSV 统计的是全部 repeated runs，而路线、时空轨迹、风险分解这类图仍然使用 representative run。后面写论文时，优先把这层区别说清楚。

它们和 `reports/figure_inventory.md` 由同一套 figure registry 自动生成。后面如果继续加图，优先在这套 registry 上扩，不要再手工同步多个列表。

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
- [docs/how_to_run.md](docs/how_to_run.md)

### KEMM 主线

- [docs/kemm_reference.md](docs/kemm_reference.md)
- [docs/formula_audit.md](docs/formula_audit.md)

### Ship 主线

- [docs/ship_simulation_reference.md](docs/ship_simulation_reference.md)
- [docs/ship_experiment_playbook.md](docs/ship_experiment_playbook.md)

### 图表与论文写作

- [docs/visualization_guide.md](docs/visualization_guide.md)
- [docs/figure_catalog.md](docs/figure_catalog.md)

---

## 12. 当前已知说明

- `SciencePlots` 是可选依赖；未安装时会自动回退到内置 matplotlib 风格。
- Windows 下运行 benchmark 或 ship 报告时，末尾可能仍出现 `joblib/loky -> wmic` 的环境警告，但一般不影响结果生成。
- ship 主线当前仍是增强 Nomoto，而不是 MMG；偏现实性来自场景、风险建模、环境层和滚动重规划，而不是高保真水动力学本体。
- benchmark 消融图 `benchmark_ablation.png` 的纵轴是“相对 `KEMM-Full` 的 MIGD 退化百分比”，正值表示删掉模块后性能变差，不是性能提升。
- 如果你看到某份旧 benchmark 报告里 `FDA1` 与 `dMOP3` 的数值整行相同，那是修复前的过期结果；当前代码已经修正 `dMOP3` 的问题定义，旧结果不应继续引用。
