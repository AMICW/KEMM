# KEMM 项目代码级说明

本文档是仓库的“细节版说明书”。如果 `README.md` 负责快速理解，那么这份文档负责逐目录、逐文件解释项目到底是怎么组织和运行的。

建议阅读顺序：

1. 先看 [../README.md](../README.md)
2. 再看 [ai_developer_handoff.md](ai_developer_handoff.md)
3. 然后回到本文档做目录级定位

本文档重点回答：

- 每个目录和文件做什么
- benchmark 主线和 `ship_simulation` 主线如何衔接
- 哪些文件是当前真实实现，哪些文件只是兼容层
- 一次 benchmark 实验和一次 ship 仿真是如何被执行出来的
- KEMM 主流程在代码里是怎样落地的

---

## 1. 总体架构

当前仓库按“双主线统一”的目标组织：

1. `benchmark` 主线
   - 用标准动态多目标测试问题验证 KEMM 与多种基线算法的性能。
2. `ship_simulation` 主线
   - 用纯代码生成的船舶会遇场景，验证 KEMM 在具有物理意义的轨迹规划问题上的效果。

这两条主线共享三类核心资产：

- 算法核心：`kemm/core/` 与 `kemm/algorithms/`
- 适配层：`kemm/adapters/`
- 报告体系：`kemm/reporting/` 与 `apps/reporting/`
- 图表风格配置：`reporting_config.py`
- 文档与审查：`README.md`、`docs/kemm_reference.md`、`docs/ship_simulation_reference.md`、`docs/formula_audit.md`、本文档

---

## 2. 目录级说明

### 2.1 `apps/`

这是应用层入口目录。它不负责写算法公式，而是负责把“算法模块 + 实验配置 + 结果展示”拼成可以运行的程序。

包含：

- `apps/benchmark_runner.py`
  - benchmark 实验的真实应用入口。
  - 负责实验配置、批量运行、结果展示、报告导出。
- `apps/ship_runner.py`
  - ship 主线的应用入口。
  - 对 `ship_simulation.main_demo` 与 `ship_simulation.run_report` 做包装。
- `apps/reporting/`
  - 对 benchmark 和 ship 两条线的图表与报告接口做统一导出。

### 2.2 `kemm/`

这是算法主体目录，是本仓库当前的“真实实现区”。

内部再分成四层：

- `kemm/algorithms/`
  - 存放 DMOEA 基类、基线算法和 KEMM 主体。
- `kemm/benchmark/`
  - 存放 benchmark 问题定义和性能指标。
- `kemm/core/`
  - 存放 KEMM 的子模块，如 adaptive、memory、drift、transfer。
- `kemm/adapters/`
  - 存放问题专用增强逻辑，例如 benchmark prior adapter。
- `kemm/reporting/`
  - 存放 benchmark 结果导出逻辑。

### 2.3 `ship_simulation/`

这是船舶仿真验证子系统。

内部包含：

- `core/`
  - 船模、环境、碰撞风险、燃油模型。
- `scenario/`
  - 场景生成与会遇定义。
- `optimizer/`
  - 问题接口、KEMM 求解封装。
- `visualization/`
  - 动画与静态图表。

### 2.4 `docs/`

这是文档目录，当前主要包含：

- `docs/formula_audit.md`
  - 公式实现审查记录。
- `docs/codebase_reference.md`
  - 本文档，偏工程结构与代码说明。
- `docs/kemm_reference.md`
  - KEMM 主线逐模块详细说明。
- `docs/ship_simulation_reference.md`
  - ship 主线逐模块详细说明。
- `docs/visualization_guide.md`
  - benchmark 与 ship 图表参数的统一配置说明。
- `docs/figure_catalog.md`
  - 当前默认导出图表的论文用途、视觉编码与图注模板索引。
- `docs/ai_developer_handoff.md`
  - 面向零上下文 AI/开发者接手的说明。

---

## 3. 文件级说明

### 3.1 当前真实实现文件

#### `apps/benchmark_runner.py`

作用：

- 定义 benchmark 批量实验配置类 `ExperimentConfig`
- 定义运行器 `ExperimentRunner`
- 定义结果展示器 `ResultPresenter`
- 定义命令行入口 `run_benchmark()` 和 `main()`

调用链：

1. `main()` 解析 `--quick / --full / --with-jy / --output-dir`
2. `run_benchmark()` 创建输出目录
3. `ExperimentRunner.run_all()` 逐算法、逐问题、逐重复运行实验
4. `ResultPresenter` 打印表格并导出图表
5. `export_benchmark_report()` 导出 CSV/JSON/Markdown 报告

#### `apps/ship_runner.py`

作用：

- 提供 `python -m apps.ship_runner` 入口
- 当前默认调用 `run_demo(show_animation=False)`
- 同时暴露 `generate_report()` 供批量 ship 实验调用

#### `apps/reporting/benchmark_visualization.py`

作用：

- benchmark 图表的真实实现文件
- 原来根目录 `visualization.py` 的主要内容已迁移到这里

当前负责的图表大类包括：

- 性能对比图
- 过程分析图
- 统计图
- 算法机制图
- 消融图
- 一键批量导出函数 `generate_all_figures()`

这个文件当前的设计重点是“图表层只消费结构化结果”。也就是说：

- benchmark runner 会先构造 `BenchmarkFigurePayload`
- KEMM 变化响应会输出 `KEMMChangeDiagnostics`
- 图表层不再依赖算法私有属性名

因此如果你后续重构 KEMM 内部流程，优先保证 payload 和 diagnostics 结构稳定，
就不需要同步重写图表代码。

#### `reporting_config.py`

作用：

- 存放 benchmark 与 ship 共用的图表风格配置

当前主要对象：

- `PublicationStyle`
- `BenchmarkPlotConfig`
- `ShipPlotConfig`

如果你要改论文图的 DPI、字体、颜色、尺寸和高亮规则，优先改这里。

#### `kemm/algorithms/base.py`

作用：

- 提供所有 DMOEA 的基础能力

典型内容：

- 初始化种群
- 评价函数调用
- 非支配排序
- 拥挤距离
- 环境选择
- 进化骨架

#### `kemm/algorithms/baselines.py`

作用：

- 保存 benchmark 对比算法实现

当前包含：

- `RI_DMOEA`
- `PPS_DMOEA`
- `KF_DMOEA`
- `SVR_DMOEA`
- `Tr_DMOEA`
- `MMTL_DMOEA`

#### `kemm/algorithms/kemm.py`

作用：

- KEMM 主算法实现

这是当前最重要的算法文件之一。它负责把多个核心模块组织成一个完整的动态多目标响应流程。

主要属性：

- `_transfer_module`
- `_multi_src_transfer`
- `_vae_memory`
- `_operator_selector`
- `_drift_detector`
- `_drift_predictor`
- `_centroid_history`

主要流程函数：

- `respond_to_change()`
- `_archive_current_environment()`
- `_adjust_operator_ratios()`
- `_allocate_operator_counts()`
- `_build_memory_candidates()`
- `_build_prediction_candidates()`
- `_build_transfer_candidates()`
- `_build_prior_candidates()`
- `_build_elite_candidates()`
- `_build_previous_population_candidates()`
- `_build_reinitialization_candidates()`
- `_estimate_response_quality()`

同时它现在还暴露结构化变化诊断：

- `last_change_diagnostics`
- `change_diagnostics_history`
- `get_last_change_diagnostics()`

这些接口的作用是让你以后改算法时，不必再从图表层或实验层直接访问私有属性。

#### `kemm/adapters/benchmark.py`

作用：

- 保存 benchmark-only 的结构先验候选生成器

价值：

- 把 `_problem_aware_candidates()` 的真实逻辑从 KEMM 主体中剥离
- 让 benchmark 强化逻辑与通用 KEMM 核心分层
- 为后续接入新问题专用 adapter 预留结构

#### `kemm/benchmark/problems.py`

作用：

- 动态测试问题定义

当前包含：

- `FDA1`
- `FDA2`
- `FDA3`
- `dMOP1`
- `dMOP2`
- `dMOP3`
- `JY1`
- `JY4`

补充说明：

- `dMOP3` 在旧版本里曾错误退化为 `FDA1` 的复制实现
- 当前代码已经修复这一问题；新的 benchmark 结果中，`dMOP3` 不应再与 `FDA1` 逐项一致
- 审查历史结果时，应优先参考修复后的输出目录，而不是旧报告缓存

每个问题通常包含：

- 目标函数
- 真实 POF 生成函数
- 时间变量生成逻辑

#### `kemm/benchmark/metrics.py`

作用：

- benchmark 性能指标定义与计算

当前包括：

- `IGD`
- `MIGD`
- `Spacing`
- `Maximum Spread`
- 部分 `HV`

#### `kemm/core/adaptive.py`

作用：

- KEMM 的策略自适应部分

对应概念：

- UCB1
- 滑动窗口 reward
- 比例型算子分配
- 漂移检测辅助量

#### `kemm/core/drift.py`

作用：

- KEMM 的前沿漂移预测部分

对应概念：

- Pareto front feature
- Gaussian Process Regression
- 预测置信度

#### `kemm/core/memory.py`

作用：

- KEMM 的压缩记忆部分

对应概念：

- VAE 压缩
- 历史环境指纹
- 相似环境检索

#### `kemm/core/transfer.py`

作用：

- KEMM 的知识迁移部分

对应概念：

- Grassmann geodesic
- 单源迁移
- 多源迁移

#### `kemm/core/types.py`

作用：

- 定义配置对象、结果对象、类型结构

价值在于：

- 参数更清晰
- 类型更稳定
- IDE 更友好
- 重构更安全

当前里面最重要的两个对象是：

- `KEMMConfig`
  - 用于集中管理 KEMM 超参数与候选池启发式系数
- `KEMMChangeDiagnostics`
  - 用于保存一次变化响应后的 operator ratios、candidate counts、confidence 和响应质量

#### `kemm/reporting/benchmark_report.py`

作用：

- 把 benchmark 结果转成结构化报告

输出内容：

- `metrics.csv`
- `ranks.csv`
- `summary.json`
- `summary.md`

#### `ship_simulation/config.py`

作用：

- 集中管理船舶仿真配置与 demo 配置

包含：

- 船舶动力学参数
- 风流场参数
- 风险参数
- 轨迹规划参数
- KEMM 求解参数

#### `ship_simulation/core/ship_model.py`

作用：

- 本船运动学/简化动力学仿真

当前版本：

- Nomoto 一阶模型
- 速度一阶滞后
- 航向跟踪

输出通常包括：

- 时序位置
- 航向
- 速度
- 是否到达终点
- 终点残差

#### `ship_simulation/core/environment.py`

作用：

- 定义风、流等环境影响

当前是 MVP 版本：

- 常值风
- 常值流

#### `ship_simulation/core/collision_risk.py`

作用：

- 计算基于船舶域的碰撞风险

输出：

- `max_risk`
- `mean_risk`
- `intrusion_time`
- `risk_series`

#### `ship_simulation/core/fuel_model.py`

作用：

- 计算轨迹对应的燃油消耗

当前是工程模型，不是完整主机热力学模型。

#### `ship_simulation/scenario/encounter.py`

作用：

- 定义会遇场景数据结构

例如：

- 本船初始状态
- 他船初始状态
- 目标点
- 场景名称

#### `ship_simulation/scenario/generator.py`

作用：

- 生成典型会遇场景

当前支持：

- `head_on`
- `crossing`
- `overtaking`

#### `ship_simulation/optimizer/interface.py`

作用：

- 把船舶仿真问题包装成优化算法可调用的接口

关键职责：

- 构造决策变量边界
- 生成初始猜测
- 提供 `evaluate()` / `evaluate_population()`
- 生成兼容 benchmark 风格的 `obj_func(pop, t)`

#### `ship_simulation/optimizer/problem.py`

作用：

- 真正定义船舶三目标问题

通常包括：

- 决策变量解析
- 轨迹仿真
- 燃油目标
- 时间目标
- 风险目标
- 越界惩罚
- 未到达终点惩罚

#### `ship_simulation/optimizer/kemm_solver.py`

作用：

- 把 benchmark 里的 KEMM 包装成 ship 问题求解器

职责包括：

- 注入初始解
- 初始化种群
- 控制进化代数
- 选择代表解
- 返回 Pareto 集与最优展示解

#### `ship_simulation/visualization/animator.py`

作用：

- 生成轨迹动画

当前展示内容包括：

- 本船轨迹
- 目标船轨迹
- 航路点
- 船舶域

#### `ship_simulation/visualization/report_plots.py`

作用：

- 生成静态报告图

当前包括：

- 轨迹对比图
- 风险时间序列
- 速度剖面
- Pareto 散点
- 收敛图
- dashboard

#### `ship_simulation/main_demo.py`

作用：

- ship 主线的最小演示入口

当前支持：

- `optimizer_name="kemm"`
- `optimizer_name="random"`

#### `ship_simulation/run_report.py`

作用：

- 批量运行 ship 场景并生成报告

当前比较：

- `KEMM`
- `Random`

输出目录：

- `raw/`
- `figures/`
- `reports/`

---

## 4. 兼容层文件说明

下面这些文件已经不再是“真实实现主体”，但为了兼容旧导入路径仍然保留：

### `run_experiments.py`

现在是 thin wrapper。真实逻辑已迁到 `apps/benchmark_runner.py`。

### `benchmark_algorithms.py`

现在是兼容导出层。真实逻辑已迁到：

- `kemm/algorithms/*`
- `kemm/benchmark/*`

### `visualization.py`

现在是兼容导出层。真实逻辑已迁到 `apps/reporting/benchmark_visualization.py`。

### `adaptive_operator.py`
### `compressed_memory.py`
### `geodesic_flow.py`
### `pareto_drift.py`

这些文件现在已经压缩为最薄兼容层。真实实现已迁到：

- `kemm/core/adaptive.py`
- `kemm/core/memory.py`
- `kemm/core/transfer.py`
- `kemm/core/drift.py`

---

## 5. benchmark 主线执行流程

一次 benchmark 快速实验的执行流程如下：

1. 入口 `python run_experiments.py --quick`
2. wrapper 转发到 `apps.benchmark_runner.main()`
3. `ExperimentConfig` 给出问题集、算法集、重复次数
4. `ExperimentRunner.run_all()` 三重循环：
   - 算法
   - 问题
   - 重复运行
5. 每次运行调用 `_run_single()`
6. 在每个环境变化点：
   - 计算当前时间变量 `t`
   - 首次直接评价
   - 后续调用 `algo.respond_to_change()`
   - 再执行若干代 `evolve_one_gen()`
7. 每个变化阶段结束后：
   - 取获得的 Pareto front
   - 生成真实 POF
   - 计算 `IGD / SP / MS`
8. 整体结束后：
   - `ResultPresenter` 打印结果
   - 画图
   - `export_benchmark_report()` 导出报告

---

## 6. ship 主线执行流程

一次 ship 仿真 demo 的执行流程如下：

1. 入口 `python -m apps.ship_runner`
2. 调用 `ship_simulation.main_demo.run_demo()`
3. 读取默认配置
4. 场景生成器生成一个会遇场景
5. `ShipOptimizerInterface` 构造优化上下文
6. 如果选 `kemm`：
   - `ShipKEMMOptimizer.optimize()`
   - 调用 benchmark 风格 KEMM
7. 如果选 `random`：
   - 使用随机搜索 baseline
8. 选择一个展示解
9. 输出 summary
10. 如开启动画则显示轨迹

批量报告流程：

1. 入口 `python ship_simulation/run_report.py`
2. 依次运行 `head_on / crossing / overtaking`
3. 每个场景分别跑：
   - `KEMM`
   - `Random`
4. 生成每场景图表
5. 汇总到统一报告目录

---

## 7. KEMM 主流程与代码对应

### 7.1 变化检测与环境归档

代码位置：

- `kemm/algorithms/kemm.py`
- `_archive_current_environment()`

职责：

- 提取上一环境精英
- 计算环境指纹
- 存入 VAE memory
- 更新漂移检测器
- 更新 GP 预测器

### 7.2 自适应算子比例

代码位置：

- `kemm/core/adaptive.py`
- `kemm/algorithms/kemm.py::_adjust_operator_ratios()`

### 7.3 压缩记忆检索

代码位置：

- `kemm/core/memory.py`
- `kemm/algorithms/kemm.py::_build_memory_candidates()`

### 7.4 漂移预测

代码位置：

- `kemm/core/drift.py`
- `kemm/algorithms/kemm.py::_build_prediction_candidates()`

### 7.5 几何迁移

代码位置：

- `kemm/core/transfer.py`
- `kemm/algorithms/kemm.py::_build_transfer_candidates()`

### 7.6 候选池合并与环境选择

代码位置：

- `kemm/algorithms/kemm.py::respond_to_change()`

候选来源：

- memory
- prediction
- transfer
- benchmark-aware prior
- elite keep
- previous population reuse
- random reinitialization

这些候选先合并，再统一评价，再用 `env_selection()` 选出下一环境种群。

---

## 8. 建议阅读顺序

推荐按下面顺序读：

1. `README.md`
2. `docs/codebase_reference.md`
3. `docs/kemm_reference.md`
4. `docs/ship_simulation_reference.md`
5. `docs/formula_audit.md`
6. `docs/visualization_guide.md`
7. `docs/figure_catalog.md`
8. `kemm/algorithms/kemm.py`
9. `apps/benchmark_runner.py`
10. `ship_simulation/optimizer/problem.py`
11. `ship_simulation/optimizer/kemm_solver.py`
12. `ship_simulation/run_report.py`
