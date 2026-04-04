# Repository Handoff Guide

本文件面向新的 AI 助手或新的开发者。目标不是重复 README，而是提供“拿到仓库后该怎么改”的最短路径。

## 1. 项目定位

仓库有两条主线：

1. `benchmark`
   - 用动态多目标标准测试问题验证 KEMM 与多种基线算法。
2. `ship_simulation`
   - 用纯代码生成的船舶会遇场景验证 KEMM 在物理语义问题上的效果。

请把它理解为一个研究型代码库，而不是纯教学 demo。

## 2. 真实实现和兼容层

优先修改这些真实实现文件：

- `apps/benchmark_runner.py`
- `apps/ship_runner.py`
- `apps/reporting/benchmark_visualization.py`
- `kemm/algorithms/*.py`
- `kemm/adapters/*.py`
- `kemm/core/*.py`
- `kemm/benchmark/*.py`
- `ship_simulation/*`
- `reporting_config.py`

这些文件主要是兼容层，不应作为首选修改目标：

- `run_experiments.py`
- `benchmark_algorithms.py`
- `visualization.py`
- `adaptive_operator.py`
- `compressed_memory.py`
- `geodesic_flow.py`
- `pareto_drift.py`

## 3. 修改目标与对应文件

如果用户要改 KEMM 主流程：

- `kemm/algorithms/kemm.py`

如果用户要改 benchmark 专用结构先验：

- `kemm/adapters/benchmark.py`

如果用户要改 KEMM 子模块：

- `kemm/core/adaptive.py`
- `kemm/core/memory.py`
- `kemm/core/drift.py`
- `kemm/core/transfer.py`

如果用户要改 KEMM 参数预算和启发式系数：

- `kemm/core/types.py`

如果用户要改 benchmark 图表：

- `apps/reporting/benchmark_visualization.py`
- `reporting_config.py`

如果用户要改 ship 静态图表：

- `ship_simulation/visualization/report_plots.py`
- `reporting_config.py`

如果用户要改 ship 动画：

- `ship_simulation/visualization/animator.py`

如果用户要改 ship 问题定义：

- `ship_simulation/optimizer/problem.py`

## 4. 必须遵守的边界

1. benchmark 图表层不应直接访问算法私有属性。
   - 使用 `BenchmarkFigurePayload`
   - 使用 `KEMMChangeDiagnostics`

2. ship 主线禁止依赖 benchmark-only prior。
   - `benchmark_aware_prior` 在 ship 侧必须保持关闭
   - benchmark 专用逻辑应放在 `kemm/adapters/benchmark.py`

3. 图表参数应优先放入 `reporting_config.py`。
   - 不要把 DPI、字体、颜色重新散落回各个绘图函数

4. 根目录旧文件应继续保持薄兼容。
   - 新逻辑不要再灌回根目录 legacy 文件

## 5. 关键结构化接口

未来重构时，尽量保持这些接口稳定：

- `KEMMConfig`
- `KEMMChangeDiagnostics`
- `BenchmarkFigurePayload`
- `ExperimentSeries`
- `ShipPlotConfig`
- `BenchmarkPlotConfig`

只要这些接口保持稳定，算法内部重构通常不会连带打碎实验入口和可视化层。

## 6. 常用验证命令

基础测试：

```bash
python -m unittest discover -s tests -v
```

benchmark 快速回归：

```bash
python run_experiments.py --quick --output-dir benchmark_outputs/smoke
```

ship 报告回归：

```bash
python ship_simulation/run_report.py
```

## 7. 进一步阅读

- `README.md`
- `docs/codebase_reference.md`
- `docs/kemm_reference.md`
- `docs/ship_simulation_reference.md`
- `docs/formula_audit.md`
- `docs/visualization_guide.md`
- `docs/ai_developer_handoff.md`
