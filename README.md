# KEMM: Dynamic Multi-Objective Optimization + Ship Trajectory Simulation

一个面向研究工作的 Python 项目，统一维护两条主线：

1. `benchmark` 主线  
   使用动态多目标标准测试问题验证 KEMM 与多种基线算法。
2. `ship_simulation` 主线  
   使用纯代码生成的船舶会遇场景验证 KEMM 在物理语义轨迹规划问题上的效果。

本仓库的目标不是做一个单次实验脚本，而是做一个可维护、可复现、可扩展、可继续演化的研究型代码库。

---

## 1. 项目亮点

- 双主线统一：benchmark 理论验证 + ship 物理语义验证
- KEMM 已拆成可维护结构：主流程、子模块、adapter、报告层明确分层
- 兼容旧入口，但真实实现已迁入新架构
- 图表系统支持统一论文风格配置，可调 DPI、字体、颜色、尺寸、highlight 规则
- 文档体系同时面向：
  - GitHub 访客
  - 新开发者
  - 新 AI 助手

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
│   ├── adapters/
│   ├── algorithms/
│   ├── benchmark/
│   ├── core/
│   └── reporting/
├── ship_simulation/
│   ├── core/
│   ├── optimizer/
│   ├── scenario/
│   ├── visualization/
│   ├── config.py
│   ├── main_demo.py
│   └── run_report.py
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

### 5.1 Python 版本

推荐：

- Python `3.10` 到 `3.12`

兼容目标：

- Python `3.9+`

### 5.2 依赖文件

- `requirements.txt`
- `requirements-dev.txt`

当前核心依赖主要是：

- `numpy`
- `scipy`
- `matplotlib`
- `SciencePlots`

### 5.3 使用 `venv`

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 5.4 更详细环境说明

详见：

- `docs/environment_setup.md`

---

## 6. 快速开始

### 6.1 benchmark 快速验证

```bash
python run_experiments.py --quick
python -m apps.benchmark_runner --quick
```

### 6.2 benchmark 完整实验

```bash
python run_experiments.py --full
```

### 6.3 自定义 benchmark 输出目录

```bash
python run_experiments.py --quick --output-dir benchmark_outputs/my_report
```

### 6.4 ship 演示

```bash
python -m apps.ship_runner
```

### 6.5 直接运行单个 ship 场景

```bash
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
```

### 6.6 生成 ship 批量报告

```bash
python ship_simulation/run_report.py
```

### 6.7 基础测试

```bash
python -m unittest discover -s tests -v
```

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

输出习惯在两条主线中保持一致，便于归档、写论文和后续批量分析。

---

## 8. KEMM 架构概览

KEMM 当前的核心机制包括：

1. 自适应策略分配
2. 压缩记忆检索
3. 漂移预测
4. 几何迁移

环境变化后的主流程可以概括为：

1. 归档上一环境的精英解与前沿特征
2. 更新漂移检测与预测状态
3. 通过 UCB1 + 启发式修正分配 operator ratio
4. 生成 memory / prediction / transfer / reinit 候选
5. 融合 elite、previous population、可选 benchmark prior
6. 统一环境选择
7. 估计响应质量并回写 reward

核心代码位置：

- 主流程：`kemm/algorithms/kemm.py`
- benchmark prior adapter：`kemm/adapters/benchmark.py`
- adaptive：`kemm/core/adaptive.py`
- memory：`kemm/core/memory.py`
- drift：`kemm/core/drift.py`
- transfer：`kemm/core/transfer.py`
- 配置与诊断对象：`kemm/core/types.py`

---

## 9. 后续改算法时，应该改哪里

如果你要继续改 KEMM 结构，请优先按下面的边界修改：

- 改通用变化响应主流程：`kemm/algorithms/kemm.py`
- 改 benchmark-only 的结构先验：`kemm/adapters/benchmark.py`
- 改子模块真实实现：`kemm/core/*.py`
- 改参数、预算、启发式系数：`kemm/core/types.py`
- 看变化响应后的结构化中间结果：`KEMMChangeDiagnostics`

不要优先改根目录 legacy 文件。

---

## 10. 可视化系统

当前图表系统分成三层：

1. 公共风格配置层  
   `reporting_config.py`
2. benchmark 图表层  
   `apps/reporting/benchmark_visualization.py`
3. ship 图表层  
   `ship_simulation/visualization/report_plots.py`

### 10.1 图表风格配置

统一风格配置对象包括：

- `PublicationStyle`
- `BenchmarkPlotConfig`
- `ShipPlotConfig`
- `build_publication_style(...)`
- `build_benchmark_plot_config(...)`
- `build_ship_plot_config(...)`

你可以显式调整：

- DPI
- 字体和字号
- 线宽、marker、透明度
- 颜色映射
- heatmap colormap
- dashboard / panel / rank bar 的尺寸
- ship 风险阈值和 Pareto colormap

目前内置预设有：

- `default`
- `paper`
- `ieee`
- `nature`
- `thesis`

### 10.2 benchmark 图表接口

benchmark 图表层使用：

- `BenchmarkFigurePayload`
- `KEMMChangeDiagnostics`

这意味着图表层不再直接依赖 KEMM 私有属性名。以后只要这些结构化接口保持稳定，算法内部重构不会直接打碎报告层。

### 10.3 示例：自定义 benchmark 图表风格

```python
from apps.reporting import BenchmarkFigurePayload, generate_all_figures
from reporting_config import build_benchmark_plot_config

plot_config = build_benchmark_plot_config(
    preset="ieee",
    style_overrides={"dpi": 420},
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

### 10.4 示例：自定义 ship 图表风格

```python
from reporting_config import build_ship_plot_config
from ship_simulation.run_report import generate_report

plot_config = build_ship_plot_config(
    preset="paper",
    style_overrides={"dpi": 380},
    own_ship_color="#0f6cbd",
    baseline_color="#b54708",
    dashboard_figsize=(15.5, 10.5),
)

generate_report(plot_config=plot_config)
```

更多细节见：

- `docs/visualization_guide.md`

说明：

- `SciencePlots` 现在作为可选样式后端接入项目
- 如果环境里没有安装 `SciencePlots`，系统会自动回退到当前内置的 `matplotlib rcParams` 风格
- 为避免中文和 LaTeX 依赖冲突，默认建议在 `science_styles` 中保留 `no-latex`
- 如果你想“以后只改一个配置文件”，优先编辑 `reporting_config.py` 顶部的 `PLOT_STYLE_PRESETS`、`BENCHMARK_PLOT_PRESETS`、`SHIP_PLOT_PRESETS`

---

## 11. 关键数学公式

### 11.1 UCB1 策略评分

\[
\mathrm{UCB}_i = Q_i + c \sqrt{\frac{\ln T}{N_i}}
\]

当前工程实现不是标准“只选一个臂”，而是进一步通过 softmax 转成比例，因此更准确地说是：

\[
p_i = \frac{\exp(\mathrm{UCB}_i / \tau)}{\sum_j \exp(\mathrm{UCB}_j / \tau)}
\]

即 `UCB-guided allocation`。

### 11.2 GPR 漂移预测

\[
k(x, x') = \sigma_f^2 \exp\left(- \frac{\|x - x'\|^2}{2l^2}\right)
\]

\[
\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y
\]

\[
\sigma_*^2 = k_{**} - k_*^T (K + \sigma_n^2 I)^{-1} k_*
\]

### 11.3 VAE 压缩记忆

\[
\mathcal{L}_{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta D_{KL}(q(z|x)\|p(z))
\]

### 11.4 Grassmann 几何迁移

\[
\Phi(t) = P_S U \cos(t\Theta) + R_S V \sin(t\Theta), \quad t \in [0, 1]
\]

### 11.5 动态多目标指标

\[
\mathrm{MIGD} = \frac{1}{T} \sum_{t=1}^{T} \mathrm{IGD}(P_t, PF_t^*)
\]

\[
\mathrm{SP} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar d)^2}
\]

\[
\mathrm{MS} = \sqrt{\frac{1}{m}\sum_{j=1}^{m}
\left(
\frac{\min(f_j^{max}, F_j^{max}) - \max(f_j^{min}, F_j^{min})}{F_j^{max} - F_j^{min}}
\right)^2}
\]

### 11.6 船舶 Nomoto 一阶模型

\[
\dot r = \frac{K r_c - r}{T}
\]

\[
\dot \psi = r
\]

---

## 12. 文档导航

如果你想快速理解仓库，建议按这个顺序阅读：

1. `README.md`
2. `AGENTS.md`
3. `docs/codebase_reference.md`
4. `docs/kemm_reference.md`
5. `docs/ship_simulation_reference.md`
6. `docs/environment_setup.md`
7. `docs/formula_audit.md`
8. `docs/visualization_guide.md`
9. `docs/ai_developer_handoff.md`

---

## 13. 给新的 AI 助手 / 开发者

为了让新开的对话也能快速接手，本仓库已经额外提供：

- `AGENTS.md`
  - 快速说明“真实实现在哪、怎么改、哪些边界不能打破”
- `docs/ai_developer_handoff.md`
  - 面向零上下文接手的详细说明

如果你是新的 AI 助手，先读这两个文件，再开始改代码。

---

## 14. 当前已知限制

- benchmark quick 模式只是 smoke regression，不代表最终论文统计结论。
- `joblib/loky` 在 Windows 环境下可能出现 `wmic` 警告，但通常不影响实验结果产物。
- ship 动画层与风险域几何还有进一步统一空间。
- 仍有部分历史输出目录和缓存可继续清理。

---

## 15. License / Notes

当前仓库更偏研究开发状态。若计划公开发布到 GitHub，建议补充：

- License
- 结果复现实验说明
- 示例图或 GIF
