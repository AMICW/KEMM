# Ship Simulation 参考文档

本文档面向三类读者：

1. 想快速理解 `ship_simulation/` 架构的开发者
2. 需要定位修改入口的新 AI 助手
3. 需要把系统结构写进论文或 GitHub 文档的作者

当前 ship 主线已经从“单次黑盒轨迹 demo”升级为“偏现实场景 + 滚动重规划 + 完整 batch report + 论文级图包”的实验系统。

如果你只是想看运行命令，优先转到 [run_commands.md](run_commands.md) 或 [how_to_run.md](how_to_run.md)。
如果你想知道 ship 实验应该怎么设计，再配合 [ship_experiment_playbook.md](ship_experiment_playbook.md) 一起看。

---

## 1. 子系统定位

`ship_simulation/` 用纯代码生成船舶会遇场景，验证 KEMM 在物理语义轨迹规划问题上的表现。

它现在重点回答的是：

- 在近海交叉、追越、狭水道等场景里，KEMM 能否兼顾燃油、时间和碰撞风险
- 在存在静态障碍、动态交通体和环境场时，滚动重规划是否能保持可执行与可解释
- 在 3 主目标之外，是否还能稳定输出论文需要的过程指标、统计表和图表对象

本轮仍然使用增强版 `Nomoto + speed lag`，没有切到 MMG。

---

## 2. 目录结构

```text
ship_simulation/
├── core/
│   ├── collision_risk.py
│   ├── environment.py
│   ├── fuel_model.py
│   └── ship_model.py
├── optimizer/
│   ├── baseline_solver.py
│   ├── episode.py
│   ├── interface.py
│   ├── kemm_solver.py
│   ├── problem.py
│   └── selection.py
├── scenario/
│   ├── encounter.py
│   └── generator.py
├── visualization/
│   ├── animator.py
│   ├── report_plots.py
│   └── __init__.py
├── config.py
├── main_demo.py
└── run_report.py
```

### 2.1 `core/`

- `ship_model.py`
  - 增强版 Nomoto 一阶船模，输出 `heading / yaw_rate / commanded_yaw_rate / speed / drift`
- `environment.py`
  - 组合环境层，支持标量场和矢量场采样
  - 新增批量接口：`scalar_risk_series()`、`vector_field_series()`、`drift_series()`
- `collision_risk.py`
  - 组合风险模型，整合船舶域、DCPA/TCPA、障碍侵入、环境暴露和 COLREG 角色权重
  - `ShipDomainRiskModel.evaluate()` 现在按整条轨迹批量计算，而不是逐时刻 Python 循环拼指标
- `fuel_model.py`
  - 简化燃油代理模型

### 2.2 `scenario/`

- `encounter.py`
  - 定义 `EncounterScenario`、动态交通体、静态障碍和场景元信息
- `generator.py`
  - 生成当前默认论文场景：`head_on / crossing / overtaking / harbor_clutter`

### 2.3 `optimizer/`

- `problem.py`
  - ship 主线的核心问题定义，负责把场景包装成三目标优化问题
  - 初始化时会预构建静态障碍描述符，避免每次评估都重复整理圆障碍和多边形边信息
- `interface.py`
  - 对外暴露统一优化接口，保持 KEMM、随机基线、NSGA-style 基线复用同一问题层
- `kemm_solver.py`
  - ship 侧 KEMM 适配器；同一个 episode 内会复用同一个 KEMM 算法 session，而不是每个 replanning step 都重建
- `baseline_solver.py`
  - 轻量 NSGA-II 风格基线
- `episode.py`
  - 滚动重规划编排层，定义 `PlanningStepResult / PlanningEpisodeResult`
  - 在进入每个场景前，会先解析当前 `ScenarioSolveProfile`
- `selection.py`
  - 代表解选择策略，先保安全净空，再看终点推进和折中目标

### 2.4 `visualization/`

- `report_plots.py`
  - ship 论文图包实现，统一消费结构化结果对象
- `animator.py`
  - 单次演示动画

---

## 3. 配置系统

ship 配置现在分成三层：

### 3.1 `ProblemConfig`

负责问题语义：

- 目标权重
- `safety_clearance`
- 各类 penalty
- `domain_risk_weight`
- 场景生成参数
- experiment change schedule

### 3.2 `DemoConfig`

负责求解与报告：

- 算法集合 `report_algorithms`
- 可选严格可比组 `*_matched`（通过 `--strict-comparable` 自动注入）
- repeated runs `n_runs`
- 随机搜索预算
- baseline 预算
- KEMM 预算
- `episode_cache_enabled`
- `render_workers`
- `plot_preset`

### 3.3 `ScenarioSolveProfiles`

负责“按场景分配预算”的第二层配置：

- `legacy_uniform`
- `full_tuned`

其中 `full_tuned` 会按场景覆盖：

- `random_search_samples`
- `evolutionary_baseline_pop_size / generations`
- `kemm_pop_size / generations`
- `kemm_initial_guess_copies`
- `kemm_heuristic_detour_limit`
- `kemm_reuse_solver_state`
- `local_horizon / execution_horizon / max_replans`
- `objective_weights`
- `safety_clearance`
- 软/硬净空惩罚
- 时间/风险安全惩罚权重
- `domain_risk_weight`

当前默认行为：

- `build_default_demo_config()` -> `full_tuned`
- `_build_quick_demo_config()` -> `legacy_uniform`

---

## 4. 关键数据接口

尽量保持这些接口稳定：

- `EncounterScenario`
- `PlanningStepResult`
- `PlanningEpisodeResult`
- `EvaluationResult`
- `ExperimentSeries`
- `ShipPlotConfig`

只要这些接口稳定，内部换风险项、换优化器、换画图实现时不会把入口层一起打碎。

其中 `ShipPlotConfig` 现在也是 ship 图表的统一视觉主题入口。后续如果要整体换色板、换图例样式、减弱底图、调整 dashboard 比例，优先改 `reporting_config.py` 里的 `ShipPlotConfig` 字段，而不是回到 `report_plots.py` 里逐张图搜硬编码颜色。

---

## 5. 场景系统

### 5.1 `EncounterScenario`

当前场景对象包含：

- `own_ship`
- `target_ships`
- `static_obstacles`
- `scalar_fields`
- `vector_fields`
- `metadata`
- `area`

### 5.2 静态障碍

当前已支持：

- `CircularObstacle`
- `PolygonObstacle`
- `KeepOutZone`
- `ChannelBoundary`

可用来表达岛礁、禁航区、水道边界或浅水危险区。

### 5.3 环境层

当前环境层支持：

- `UniformVectorField`
- `VortexVectorField`
- `GaussianScalarField`
- `GridScalarField`
- `GridVectorField`

并且已经支持按轨迹批量取样：

- `EnvironmentField.scalar_risk_series(positions, times)`
- `EnvironmentField.vector_field_series(positions, times)`
- `EnvironmentField.drift_series(positions, times)`

这意味着报告图里既可以画环境热图，也可以画矢量背景场，同时运行时的环境查询不再严重依赖逐时刻 Python 分派。

### 5.4 场景调参入口

如果你想在不改 `generator.py` 的前提下直接调整场景复杂度，优先改：

- `ProblemConfig.scenario_generation.head_on`
- `ProblemConfig.scenario_generation.crossing`
- `ProblemConfig.scenario_generation.overtaking`
- `ProblemConfig.scenario_generation.harbor_clutter`

其中 `harbor_clutter` 额外支持：

- `circular_obstacle_limit`
- `polygon_obstacle_limit`
- `target_limit`
- `circular_radius_scale`
- `polygon_scale`
- `channel_width_scale`

如果你要把单一场景扩成“同一 family 下的可复现实验变体”，优先再看这些字段：

- `family_name`
- `scenario_seed`
- `difficulty_scale`
- `geometry_jitter_m`
- `traffic_heading_jitter_deg`
- `current_direction_jitter_deg`

这些参数的设计目标不是做完全随机地图，而是在保留 `head_on / crossing / overtaking / harbor_clutter` 语义的前提下，生成同 family、可复现、可分层的几何与交通扰动变体。

### 5.5 动态 experiment profile 入口

如果你想让 ship 主线不只跑静态场景，而是在 rolling-horizon 过程中注入变化，优先改：

- `ProblemConfig.experiment`

或者直接调用：

- `apply_experiment_profile(config, "drift")`
- `apply_experiment_profile(config, "shock")`
- `apply_experiment_profile(config, "recurring_harbor")`

---

## 6. 风险、目标和约束语义

### 6.1 问题定义

`ship_simulation/optimizer/problem.py` 把船舶规划写成：

- `fuel`
- `time`
- `risk`

三目标问题。

### 6.2 `risk` 的当前含义

当前 `risk` 目标主要表达沿轨迹的真实安全暴露，不再把 terminal progress 直接塞进风险维。

它主要由这些量组成：

- `max_risk`
- `mean_risk`
- 安全净距相关惩罚
- 风险侵入时间相关惩罚

### 6.3 `cv` 的当前含义

当前 `cv` 主要保留约束语义，而不是混入终点推进不足。

它聚合的是：

- `bounds_penalty`
- `safety_penalty`
- `intrusion_penalty`

### 6.4 批量评估优化

本轮提速里最关键的 ship 侧变更是：

- 风险评估从逐时刻 Python 循环，改成按整条轨迹批量计算
- own/target 速度序列、DCPA/TCPA、船舶域、净空和环境风险都尽量按数组处理
- `_obstacle_clearance()` 不再在每次评估里重复构造多边形数组和边对
- `NomotoShip.simulate_route()` 仍保持逐步积分，但环境查询已经下沉到更轻量的批量/半批量路径

---

## 7. 规划流程

ship 主线现在默认走 episode，而不是一次性全局规划。

`RollingHorizonPlanner.run()` 的流程是：

1. 用当前本船与目标船状态构造局部场景
2. 根据场景 key 解析当前 `ScenarioSolveProfile`
3. 生成局部 horizon 内的 ship planning problem
4. 调用局部优化器求解一批候选解
5. 选出代表解
6. 只执行前一段轨迹
7. 更新全局时间、本船状态、目标船状态
8. 若当前 step 有 experiment schedule，则先注入动态变化
9. 继续下一轮重规划，直到到达、风险切断或达到最大重规划次数

当前代表解选择规则不是简单“目标最小”，而是：

1. 优先不侵入障碍
2. 再优先满足 `safety_clearance`
3. 再比较终点推进程度
4. 最后才比较多目标折中分数

### 7.1 ship 侧算法架构口径

如果你要把 ship 主线写进论文、汇报或答辩，当前更准确的口径不是“直接把 benchmark 版 KEMM 搬到船舶场景”，而是：

- `kemm/` 继续承担通用动态多目标响应骨架
- `ship_simulation/optimizer/problem.py` 负责把航迹规划写成三目标问题
- `ship_simulation/optimizer/kemm_solver.py` 负责把 ship 专用的初始候选注入到 KEMM 种群
- `ship_simulation/optimizer/selection.py` 负责从当前 Pareto 解集中选出真正可执行的代表轨迹
- `episode.py` 负责把“求解一次”变成“滚动重规划闭环”

### 7.2 场景感知绕行初始候选

`ShipTrajectoryProblem.heuristic_seed_vectors()` 会先根据：

- 起点
- 终点
- 航路法向量
- 静态障碍中心
- 目标船初始位置
- `safety_clearance`

生成一批显式绕行种子。

这些种子不是纯随机点，而是“直航骨架 + 侧向偏移 + 危险区局部抬升”的 warm start。

---

## 8. 批量报告流程

`ship_simulation/run_report.py` 现在不再是“边算边画”的单阶段脚本，而是一个三阶段 batch pipeline。

### 8.1 阶段 1：episode 计算与缓存

完整报告会先跑完：

- `4` 个场景
- `3` 个算法
- `3` 次重复

也就是 `36` 个 episode。

缓存粒度固定为：

- `(scenario, algorithm, run, cache_signature)`

默认缓存位置：

- `raw/episode_cache/`

### 8.2 阶段 2：图表渲染

episode 结果准备好之后，图表渲染会独立进入第二阶段。

默认：

- `render_workers = 2`

如果渲染进程不可用，会自动退回串行渲染。

### 8.3 阶段 3：summary / metadata / inventory

最后才写：

- `reports/summary.md`
- `reports/statistical_significance.md`
- `reports/figure_inventory.md`
- `reports/robustness_sweep.md`（仅启用扰动扫描时）
- `figures/robustness_success_curve.png`（仅启用扰动扫描且开启渲染时）
- `raw/report_metadata.json`
- `raw/figure_manifest.json`
- `raw/representative_runs.json`
- `raw/planning_steps.json`
- `raw/scenario_catalog.json`
- `raw/statistical_tests.json`
- `raw/statistical_tests.csv`
- `raw/robustness_summary.json`（仅启用扰动扫描时）
- `raw/robustness_runs.csv`（仅启用扰动扫描时）
- `raw/robustness_curve.csv`（仅启用扰动扫描时）

### 8.4 representative run 与 repeated runs

当前 ship 报告明确区分两类输出：

- 统计表与 summary：聚合全部 repeated runs
- 路径、风险分解、时空轨迹等图：使用 representative run

representative run 的索引和选择方式会写进：

- `raw/representative_runs.json`

---

## 9. 输出目录

默认输出结构：

```text
ship_simulation/outputs/report_YYYYMMDD_HHMMSS/
├── figures/
├── raw/
└── reports/
```

当前 `raw/` 重点文件：

- `summary.csv`
- `aggregate_summary.csv`
- `summary.json`
- `planning_steps.json`
- `representative_runs.json`
- `report_metadata.json`
- `scenario_catalog.json`
- `figure_manifest.json`
- `statistical_tests.json`
- `statistical_tests.csv`
- `robustness_summary.json`（仅 `--robustness-sweep`）
- `robustness_runs.csv`（仅 `--robustness-sweep`）
- `robustness_curve.csv`（仅 `--robustness-sweep`）

其中：

- `summary.csv` 会带主指标、分析指标和 knee point 字段
- `planning_steps.json` 会带每次重规划的 step 摘要、knee、snapshots 和终止原因
- `report_metadata.json` 会记录当前 experiment profile、solve profile、缓存命中和分阶段耗时
- `statistical_tests.*` 会记录 `KEMM` 对各 baseline 的置信区间和显著性检验结果
- `robustness_*` 会记录扰动强度扫描下的成功率曲线和原始运行记录

---

## 10. 入口与命令

### 10.1 单次演示

```powershell
python -m apps.ship_runner
python -c "from ship_simulation.main_demo import run_demo; run_demo('crossing', optimizer_name='kemm', show_animation=False)"
```

### 10.2 批量报告

```powershell
python ship_simulation/run_report.py --quick --scenarios crossing --n-runs 1 --plot-preset paper
python ship_simulation/run_report.py --workers 4
python ship_simulation/run_report.py --appendix-plots
python ship_simulation/run_report.py --summary-only
python ship_simulation/run_report.py --algorithms kemm random nsga_style --strict-comparable
python ship_simulation/run_report.py --robustness-sweep --robustness-levels 0,0.25,0.5,0.75,1.0 --robustness-scenarios crossing overtaking harbor_clutter
```

### 10.3 测试

```powershell
python -m unittest discover -s tests -v
```

---

## 11. 优先修改入口

如果你要改：

- 场景结构：`ship_simulation/scenario/encounter.py`
- 场景模板：`ship_simulation/scenario/generator.py`
- 风险/目标定义：`ship_simulation/optimizer/problem.py`
- 滚动重规划：`ship_simulation/optimizer/episode.py`
- 局部求解器：`ship_simulation/optimizer/kemm_solver.py`
- 代表解选择：`ship_simulation/optimizer/selection.py`
- ship 图表：`ship_simulation/visualization/report_plots.py`
- 统一绘图风格：`reporting_config.py`

不要把新逻辑优先写回根目录 legacy 兼容层。

---

## 12. 当前边界

- 本轮仍是增强版 Nomoto，不是 MMG
- 目标船仍是恒向恒速，只是已经统一放在动态交通体层
- ship 主线禁止依赖 benchmark-only prior
- 图表层只消费结构化结果对象，不读求解器私有字段
- 风格参数优先改 `reporting_config.py`
- 完整报告对外仍然保持“一条命令跑完”，只是内部阶段化执行

---

## 13. 推荐图表与实验叙事映射

如果你在写论文或整理 GitHub 展示，当前 ship 主线的推荐叙事顺序是：

1. 用 `*_environment_overlay.png` 先交代场景、障碍、环境场和代表轨迹
2. 用 `*_snapshots.png` 展示关键交会时刻的动态避碰
3. 用 `*_spatiotemporal.png` 展示时空穿越关系
4. 用 `*_control_timeseries.png` 说明轨迹具备动力学可执行性
5. 用 `*_pareto3d.png` 解释三目标折中与 knee point 推荐解
6. 用 `*_parallel.png` 和 `*_radar.png` 展示扩展分析指标
7. 用 `*_convergence.png`、`*_run_statistics.png` 和 `*_distribution.png` 展示 repeated-run 稳定性
8. 用 `*_dashboard.png` 做一页式总览，而不是替代主图

对应文档索引：

- 图表表达方式：`docs/figure_catalog.md`
- 图表参数与模板：`docs/visualization_guide.md`
