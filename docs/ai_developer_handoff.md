# AI Developer Handoff

本文档给新的 AI 助手和新的开发者一个最短接手路径。

目标不是重复 README，而是明确三件事：

1. 真实实现在哪里
2. 改不同任务时应该落到哪些文件
3. 哪些边界不能打破

---

## 1. 仓库结构

仓库有两条主线：

1. `benchmark`
   用动态多目标标准测试问题验证 KEMM
2. `ship_simulation`
   用偏现实船舶会遇场景验证 KEMM 在物理语义问题上的表现

它是研究代码库，不是单次 demo 仓库。

---

## 2. 真实实现入口

优先修改这些位置：

- `apps/benchmark_runner.py`
- `apps/ship_runner.py`
- `apps/reporting/benchmark_visualization.py`
- `kemm/algorithms/*.py`
- `kemm/adapters/*.py`
- `kemm/core/*.py`
- `kemm/benchmark/*.py`
- `kemm/reporting/*.py`
- `ship_simulation/*`
- `reporting_config.py`

这些文件主要是兼容层，不是新逻辑的首选落点：

- `run_experiments.py`
- `benchmark_algorithms.py`
- `visualization.py`
- `adaptive_operator.py`
- `compressed_memory.py`
- `geodesic_flow.py`
- `pareto_drift.py`

---

## 3. 改动任务到文件的映射

### 3.1 改 KEMM 主流程

- `kemm/algorithms/kemm.py`

### 3.2 改 benchmark-only prior

- `kemm/adapters/benchmark.py`

### 3.3 改 KEMM 子模块

- `kemm/core/adaptive.py`
- `kemm/core/memory.py`
- `kemm/core/drift.py`
- `kemm/core/transfer.py`
- `kemm/core/types.py`

### 3.4 改 ship 场景和规划

- `ship_simulation/scenario/encounter.py`
- `ship_simulation/scenario/generator.py`
- `ship_simulation/optimizer/problem.py`
- `ship_simulation/optimizer/episode.py`
- `ship_simulation/optimizer/selection.py`
- `ship_simulation/core/collision_risk.py`
- `ship_simulation/core/ship_model.py`

### 3.5 改图表与论文风格

- benchmark 图：`apps/reporting/benchmark_visualization.py`
- ship 图：`ship_simulation/visualization/report_plots.py`
- 公共风格：`reporting_config.py`

---

## 4. 当前 ship 主线状态

ship 主线当前默认语义是：

- 偏现实场景
- 静态障碍 + 动态交通体 + 环境场
- 滚动重规划 episode
- 同一个 episode 内复用同一个 KEMM session
- 3 主目标优化
- 额外输出分析指标
- 核心论文图包默认开启
- 轨迹类图默认展示 representative run，统计表保留 repeated-run aggregates

关键公共接口：

- `EncounterScenario`
- `PlanningStepResult`
- `PlanningEpisodeResult`
- `ExperimentSeries`
- `ShipPlotConfig`

---

## 5. 当前 benchmark 主线状态

benchmark 主线已经把图表层与算法层隔离：

- 图表层消费 `BenchmarkFigurePayload`
- 机制诊断通过 `KEMMChangeDiagnostics`
- benchmark 图表不应继续直接探测算法私有字段

这条边界要保持住。

---

## 6. 必须遵守的边界

1. benchmark 图表层不直接访问算法私有属性
2. ship 主线禁止依赖 benchmark-only prior
3. 绘图参数优先集中在 `reporting_config.py`
4. 根目录 legacy 文件继续保持薄兼容
5. 新图和新逻辑不要灌回兼容层
6. ship 侧真实 KEMM 机制开关优先走 `DemoConfig.kemm.runtime`，不要只改包装层预算字段
7. ship 报告算法顺序和集合优先走 `DemoConfig.report_algorithms`
8. 如果动 ship 场景生成逻辑，同步保证 `scenario_catalog.json` 还能记录 family / seed / difficulty

---

## 7. 常用验证命令

```powershell
python -m unittest discover -s tests -v
python run_experiments.py --quick --output-dir benchmark_outputs/smoke
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1
```

---

## 8. 推荐阅读顺序

1. `README.md`
2. `AGENTS.md`
3. `docs/codebase_reference.md`
4. `docs/kemm_reference.md`
5. `docs/ship_simulation_reference.md`
6. `docs/visualization_guide.md`

然后再进入具体代码文件。

---

## 9. 提交前最低检查

至少确认：

- `tests/` 通过
- benchmark 主线入口还能跑
- ship 主线入口还能跑
- 默认报告图能导出
- 结构化结果对象字段没有被破坏
- README 和细分文档没有继续引用旧接口或旧图名

---

## 10. 新增图或删图时必须同步更新的文档

只要默认导出图表发生变化，至少同步更新这些文件：

- `README.md`
- `docs/figure_catalog.md`
- `docs/visualization_guide.md`
- `docs/ship_simulation_reference.md`
- `docs/kemm_reference.md`

原则是：

1. `README.md` 只保留总索引和入口说明
2. `docs/visualization_guide.md` 说明怎么调图
3. `docs/figure_catalog.md` 说明每张图表达什么、论文里怎么写
4. ship 或 benchmark 专题文档负责把图表映射回实验叙事
