# AI / 开发者接手说明

本文档是给“新开一个 AI 助手对话后，从零开始也能马上接手仓库”的专用说明。

目标不是讲理论，而是直接回答：

- 这个仓库的用户真实需求是什么
- 当前架构的意图是什么
- 改某类功能时应该去哪改
- 哪些边界不能打破

## 1. 用户的核心目标

用户当前的真实需求有三条：

1. 持续优化 KEMM 的算法结构与实验体系。
2. 把 `ship_simulation` 做成具有物理语义的验证平台，而不是纯 demo。
3. 让整个仓库足够清晰，未来不管是人还是 AI，都能快速定位并修改对应模块。

这意味着你在接手仓库时，不应只关注“能跑”，还要同时关注：

- 架构清晰度
- 可扩展性
- 文档完整度
- 可视化质量
- benchmark 与 ship 双主线的一致性

## 2. 当前架构意图

仓库目前按“双主线统一”设计：

- benchmark 主线负责理论验证
- ship 主线负责物理语义验证

共享的关键骨架是：

- `kemm/algorithms/`
- `kemm/core/`
- `kemm/adapters/`
- `reporting_config.py`
- `apps/reporting/`

用户后续最在意的是：

- 以后继续修改算法时，要很方便
- 图表层不要因为算法内部重构就一起崩
- 新助手接手时，不要再从根目录一堆 legacy 文件里猜测真实实现在哪里

## 3. 真实实现位置

请优先从这些文件开始读：

### 3.1 benchmark 主线

- `apps/benchmark_runner.py`
- `apps/reporting/benchmark_visualization.py`
- `kemm/algorithms/kemm.py`
- `kemm/adapters/benchmark.py`
- `kemm/core/types.py`
- `kemm/core/adaptive.py`
- `kemm/core/memory.py`
- `kemm/core/drift.py`
- `kemm/core/transfer.py`

### 3.2 ship 主线

- `ship_simulation/main_demo.py`
- `ship_simulation/run_report.py`
- `ship_simulation/optimizer/problem.py`
- `ship_simulation/optimizer/kemm_solver.py`
- `ship_simulation/visualization/report_plots.py`
- `ship_simulation/visualization/animator.py`

### 3.3 风格与文档

- `reporting_config.py`
- `README.md`
- `AGENTS.md`
- `docs/codebase_reference.md`
- `docs/kemm_reference.md`
- `docs/ship_simulation_reference.md`
- `docs/visualization_guide.md`

## 4. 哪些文件不应优先修改

这些文件主要是兼容层：

- `run_experiments.py`
- `benchmark_algorithms.py`
- `visualization.py`
- `adaptive_operator.py`
- `compressed_memory.py`
- `geodesic_flow.py`
- `pareto_drift.py`

如果用户没有明确要求兼容旧导入路径，不要把新实现重新塞回这些文件。

## 5. 修改任务到文件的映射

如果任务是“改算法结构”：

- 先看 `kemm/algorithms/kemm.py`
- 如果涉及 benchmark-only prior，再看 `kemm/adapters/benchmark.py`
- 如果涉及子模块机制，再看 `kemm/core/*.py`

如果任务是“改可视化质量或论文图”：

- 风格参数：`reporting_config.py`
- benchmark 图：`apps/reporting/benchmark_visualization.py`
- ship 图：`ship_simulation/visualization/report_plots.py`

如果任务是“改 ship 仿真目标或约束”：

- `ship_simulation/optimizer/problem.py`
- `ship_simulation/core/ship_model.py`
- `ship_simulation/core/collision_risk.py`

如果任务是“改文档或 GitHub 展示”：

- `README.md`
- `AGENTS.md`
- `docs/*.md`

## 6. 必须保持的接口边界

### 6.1 KEMM 到图表层

不要让 benchmark 图表层重新直接访问算法私有属性。应通过：

- `KEMMChangeDiagnostics`
- `BenchmarkFigurePayload`

### 6.2 benchmark prior 到通用 KEMM

不要把 benchmark-only prior 重新塞回通用 KEMM 主体。应通过：

- `BenchmarkPriorAdapter`

### 6.3 图表风格到绘图函数

不要把字体、DPI、配色重新散落在各个图函数里。应优先通过：

- `PublicationStyle`
- `BenchmarkPlotConfig`
- `ShipPlotConfig`

## 7. 建议的接手顺序

当你第一次进入仓库时，建议按这个顺序理解：

1. `README.md`
2. `AGENTS.md`
3. `docs/codebase_reference.md`
4. `docs/kemm_reference.md`
5. `docs/ship_simulation_reference.md`
6. 再进入具体代码文件

## 8. 基础回归命令

```bash
python -m unittest discover -s tests -v
python run_experiments.py --quick --output-dir benchmark_outputs/smoke
python ship_simulation/run_report.py
```

## 9. 典型判断标准

当你完成修改后，至少应确认：

- benchmark 主线还能跑
- ship 主线还能跑
- 图表还能生成
- 结构化 payload 和 diagnostics 没被破坏
- 文档仍能指出真实实现位置
