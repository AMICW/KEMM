# KEMM: Dynamic Multi-Objective Optimization + Ship Trajectory Simulation

KEMM 是一个面向研究工作的 Python 代码库，统一维护两条主线：

1. `benchmark`
   用动态多目标标准测试问题验证 KEMM 与多种基线算法。
2. `ship_simulation`
   用纯代码生成的偏现实船舶会遇场景，验证 KEMM 在物理语义轨迹规划问题上的效果。

这个仓库的目标不是堆一次性实验脚本，而是提供一个可维护、可复现、可持续演化的研究型项目。

## 项目概览

- 统一的 KEMM 核心：`kemm/algorithms/`、`kemm/core/`、`kemm/adapters/`
- 双实验主线：benchmark 理论验证 + ship 物理语义验证
- 统一报告体系：benchmark 与 ship 共用一套风格配置和导出约定
- 面向论文与专利的文档体系：运行手册、架构说明、图表目录、专利草稿与专利附图

## 结果预览

### Benchmark

![Benchmark Preview](./docs/images/benchmark_preview.png)

### Ship Simulation

![Ship Preview](./docs/images/ship_preview.png)

## 先从哪里开始

如果你是第一次打开仓库，建议按这个顺序：

1. 看这份 `README.md`
2. 看 [docs/README.md](docs/README.md)
3. 根据你的目标继续往下：
   - 想运行项目：看 [docs/environment_setup.md](docs/environment_setup.md)、[docs/run_commands.md](docs/run_commands.md)、[docs/how_to_run.md](docs/how_to_run.md)
   - 想改代码：看 [AGENTS.md](AGENTS.md)、[docs/ai_developer_handoff.md](docs/ai_developer_handoff.md)、[docs/codebase_reference.md](docs/codebase_reference.md)
   - 想理解算法：看 [docs/kemm_reference.md](docs/kemm_reference.md)、[docs/formula_audit.md](docs/formula_audit.md)
   - 想理解 ship 主线：看 [docs/ship_simulation_reference.md](docs/ship_simulation_reference.md)、[docs/ship_experiment_playbook.md](docs/ship_experiment_playbook.md)
   - 想整理图表或论文：看 [docs/visualization_guide.md](docs/visualization_guide.md)、[docs/figure_catalog.md](docs/figure_catalog.md)

## 快速开始

推荐 Python 版本：`3.10` 到 `3.12`，兼容目标 `3.9+`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

基础验证：

```powershell
python -m unittest discover -s tests -v
```

benchmark 快速回归：

```powershell
python -m apps.benchmark_runner --quick --plot-preset paper
```

ship 快速物理报告：

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
```

完整命令速查请直接看：

- 中文版：[docs/run_commands.md](docs/run_commands.md)
- 英文版：[docs/how_to_run.md](docs/how_to_run.md)

## 仓库结构

```text
.
├── AGENTS.md
├── README.md
├── reporting_config.py
├── docs/
├── apps/
├── kemm/
├── ship_simulation/
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── run_experiments.py
├── benchmark_algorithms.py
└── visualization.py
```

## 真实实现与兼容层

优先修改这些真实实现位置：

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

这些文件主要用于兼容旧导入路径和旧命令，不是新逻辑的首选落点：

- `run_experiments.py`
- `benchmark_algorithms.py`
- `visualization.py`
- `adaptive_operator.py`
- `compressed_memory.py`
- `geodesic_flow.py`
- `pareto_drift.py`

## 输出目录

benchmark 默认输出：

```text
benchmark_outputs/benchmark_YYYYMMDD_HHMMSS/
├── figures/
├── raw/
└── reports/
```

ship 默认输出：

```text
ship_simulation/outputs/report_YYYYMMDD_HHMMSS/
├── figures/
├── raw/
└── reports/
```

## 文档地图

建议把 [docs/README.md](docs/README.md) 当成 `docs/` 首页。里面按读者类型和任务类型整理了文档入口。

高频文档：

- [docs/environment_setup.md](docs/environment_setup.md)：安装和运行环境
- [docs/run_commands.md](docs/run_commands.md)：中文命令速查
- [docs/how_to_run.md](docs/how_to_run.md)：英文命令速查
- [docs/ai_developer_handoff.md](docs/ai_developer_handoff.md)：新开发者和 AI 助手最短接手路径
- [docs/codebase_reference.md](docs/codebase_reference.md)：代码级结构说明
- [docs/kemm_reference.md](docs/kemm_reference.md)：KEMM 主线详解
- [docs/ship_simulation_reference.md](docs/ship_simulation_reference.md)：ship 子系统详解
- [docs/visualization_guide.md](docs/visualization_guide.md)：图表配置与导出
- [docs/figure_catalog.md](docs/figure_catalog.md)：图表含义、图注和论文用途

## 当前边界

- benchmark 图表层不应直接访问算法私有属性
- ship 主线禁止依赖 benchmark-only prior
- 图表参数优先集中在 `reporting_config.py`
- 根目录 legacy 文件继续保持薄兼容

这些边界的细化版本见 [AGENTS.md](AGENTS.md) 和 [docs/ai_developer_handoff.md](docs/ai_developer_handoff.md)。
