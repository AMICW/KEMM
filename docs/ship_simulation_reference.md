# Ship Simulation 参考文档

本文档面向三类读者：

1. 想快速理解 `ship_simulation/` 架构的开发者
2. 需要定位修改入口的新 AI 助手
3. 需要把系统结构写进论文或 GitHub 文档的作者

当前 ship 主线已经从“单次黑盒轨迹 demo”升级为“偏现实场景 + 滚动重规划 + 论文级图包”的实验系统。

如果你只是想看运行命令，优先转到 [run_commands.md](run_commands.md) 或 [how_to_run.md](how_to_run.md)。
如果你想知道 ship 实验应该怎么设计，再配合 [ship_experiment_playbook.md](ship_experiment_playbook.md) 一起看。

---

## 1. 子系统定位

`ship_simulation/` 用纯代码生成船舶会遇场景，验证 KEMM 在物理语义轨迹规划问题上的表现。

它现在重点回答的是：

- 在近海交叉、追越、狭水道等场景里，KEMM 能否兼顾燃油、时间和碰撞风险
- 在存在静态障碍、动态交通体和环境场时，滚动重规划是否能保持可执行与可解释
- 在 3 主目标之外，是否还能稳定输出论文需要的过程指标和图表对象

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
  增强版 Nomoto 一阶船模，输出 `heading / yaw_rate / commanded_yaw_rate / speed / drift`
- `environment.py`
  组合环境层，支持标量场和矢量场采样
- `collision_risk.py`
  组合风险模型，整合船舶域、DCPA/TCPA、障碍侵入、环境暴露和 COLREG 角色权重
- `fuel_model.py`
  简化燃油代理模型

### 2.2 `scenario/`

- `encounter.py`
  定义 `EncounterScenario`、动态交通体、静态障碍和场景元信息
- `generator.py`
  生成当前默认论文场景：`head_on / crossing / overtaking / harbor_clutter`
- `config.py`
  `ProblemConfig.scenario_generation` 现在可以直接调场景生成参数，例如 `harbor_clutter` 的障碍数量、障碍尺度、目标船数量和通航窗口宽度
  `ProblemConfig.experiment` 现在可以直接注入滚动重规划阶段的动态实验变化，例如 drift、shock 和 recurring_harbor

### 2.3 `optimizer/`

- `problem.py`
  ship 主线的核心问题定义，负责把场景包装成三目标优化问题
- `interface.py`
  对外暴露统一优化接口，保持 KEMM、随机基线、NSGA-style 基线复用同一问题层
- `kemm_solver.py`
  ship 侧 KEMM 适配器；同一个 episode 内会复用同一个 KEMM 算法 session，而不是每个 replanning step 都重建
- `baseline_solver.py`
  轻量 NSGA-II 风格基线
- `episode.py`
  滚动重规划编排层，新增 `PlanningStepResult / PlanningEpisodeResult`
- `selection.py`
  代表解选择策略，先保安全净空，再看终点推进和折中目标

### 2.4 `visualization/`

- `report_plots.py`
  ship 论文图包实现，统一消费结构化结果对象
- `animator.py`
  单次演示动画

---

## 3. 关键数据接口

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

## 4. 场景系统

### 4.1 `EncounterScenario`

当前场景对象包含：

- `own_ship`
- `target_ships`
- `static_obstacles`
- `scalar_fields`
- `vector_fields`
- `metadata`
- `area`

### 4.2 静态障碍

当前已支持：

- `CircularObstacle`
- `PolygonObstacle`
- `KeepOutZone`
- `ChannelBoundary`

可用来表达岛礁、禁航区、水道边界或浅水危险区。

### 4.3 环境层

当前环境层支持：

- `UniformVectorField`
- `VortexVectorField`
- `GaussianScalarField`
- `GridScalarField`
- `GridVectorField`

这意味着报告图里既可以画环境热图，也可以画矢量背景场。

### 4.4 场景调参入口

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

这使得场景难度可以通过配置对象直接重现实验，而不是把数值散落回 `generator.py`。

如果你要把单一场景扩成“同一 family 下的可复现实验变体”，优先再看这些字段：

- `family_name`
- `scenario_seed`
- `difficulty_scale`
- `geometry_jitter_m`
- `traffic_heading_jitter_deg`
- `current_direction_jitter_deg`

这组参数的设计目标不是做完全随机地图，而是在保留 `head_on / crossing / overtaking / harbor_clutter` 语义的前提下，生成同 family、可复现、可分层的几何与交通扰动变体。

### 4.5 动态实验 profile 入口

如果你想让 ship 主线不只跑静态场景，而是在 rolling-horizon 过程中注入变化，优先改：

- `ProblemConfig.experiment`

或者直接调用：

- `apply_experiment_profile(config, "drift")`
- `apply_experiment_profile(config, "shock")`
- `apply_experiment_profile(config, "recurring_harbor")`

当前变化事件支持：

- 环境标量场强度缩放
- 环境矢量场强度缩放
- 背景 current 强度缩放
- 目标船速度缩放
- 目标船航向偏移
- 通道宽度收缩
- 临时 closure 障碍注入

---

## 5. 规划流程

ship 主线现在默认走 episode，而不是一次性全局规划。

`RollingHorizonPlanner.run()` 的流程是：

1. 用当前本船与目标船状态构造局部场景
2. 生成局部 horizon 内的 ship planning problem
3. 调用局部优化器求解一批候选解
4. 选出代表解
5. 只执行前一段轨迹
6. 更新全局时间、本船状态、目标船状态
7. 若当前 step 有 experiment schedule，则先注入动态变化
8. 继续下一轮重规划，直到到达、风险切断或达到最大重规划次数

当前代表解选择规则不是简单“目标最小”，而是：

1. 优先不侵入障碍
2. 再优先满足 `safety_clearance`
3. 再比较终点推进程度
4. 最后才比较多目标折中分数

### 5.1 ship 侧算法架构口径

如果你要把 ship 主线写进论文、汇报或答辩，当前更准确的口径不是“直接把 benchmark 版 KEMM 搬到船舶场景”，而是：

- `kemm/` 继续承担通用动态多目标响应骨架
- `ship_simulation/optimizer/problem.py` 负责把航迹规划写成三目标问题
- `ship_simulation/optimizer/kemm_solver.py` 负责把 ship 专用的初始候选注入到 KEMM 种群
- `ship_simulation/optimizer/selection.py` 负责从当前 Pareto 解集中选出真正可执行的代表轨迹
- `episode.py` 负责把“求解一次”变成“滚动重规划闭环”

从论文结构上，可以把 ship 侧算法理解成“通用 KEMM 响应引擎 + 船舶场景专用安全壳层”，而不是另起一套新算法。

当前 ship 实现里最需要说明的 3 个点是：

1. `场景感知绕行初始候选`
   - `ShipTrajectoryProblem.heuristic_seed_vectors()` 会先根据起点、终点、航路法向量、静态障碍中心、目标船初始位置和 `safety_clearance` 生成一批显式绕行种子。
   - 这些种子不是纯随机点，而是“直航骨架 + 侧向偏移 + 危险区局部抬升”的 warm start。
   - `ship_simulation/optimizer/kemm_solver.py` 会在初始化种群和变化响应阶段都把这批候选混入 KEMM 的候选池，避免 crossing、harbor_clutter 一类场景一开始就被高风险直穿路径主导。

2. `风险目标与终端推进解耦`
   - 当前 `risk` 目标主要表达“沿轨迹的碰撞与侵入风险”，核心由 `max_risk`、`mean_risk`、安全净距惩罚和侵入时间惩罚组成。
   - 终点没走到位这件事，不再直接加到 `risk` 上。
   - 终端推进不足主要通过 `fuel/time` 维度里的 terminal penalty，以及代表解选择阶段的“终点推进优先级”来体现。
   - 这个调整的目的，是避免“没走到终点”和“真的高风险”被混成同一种风险信号。

3. `CV 只保留约束语义`
   - 当前 `cv` 聚合的是 `bounds_penalty + safety_penalty + intrusion_penalty`。
   - 也就是说，`cv` 现在更接近“可行性/安全性违背度”，而不是把终端推进不足也塞进去。
   - 这样做以后，环境选择阶段先比较“是否越界、是否侵入、是否安全不足”，再比较多目标优劣，语义更干净，也更适合做物理场景解释。

如果你需要一句很凝练的话，可以直接说：

`ship 侧 KEMM 的本质是：用通用动态多目标响应框架搜索候选轨迹，再用船舶场景专用的绕行初值、安全约束和滚动执行机制把它落到可解释的航迹规划问题上。`

### 5.2 ship 侧 KEMM 配置入口

如果你要改 ship 主线里的 KEMM，不要只改包装层预算，还要区分这两层：

- `DemoConfig.kemm`
- `DemoConfig.kemm.runtime`

其中：

- `DemoConfig.kemm.pop_size / generations / seed` 控制 ship 侧求解预算
- `DemoConfig.kemm.use_change_response` 控制每次 replanning 时是否调用 KEMM 的环境响应流程
- `DemoConfig.kemm.runtime` 透传给 `kemm.core.types.KEMMConfig`

也就是说，真正的机制开关在：

- `DemoConfig.kemm.runtime.enable_memory`
- `DemoConfig.kemm.runtime.enable_prediction`
- `DemoConfig.kemm.runtime.enable_transfer`
- `DemoConfig.kemm.runtime.enable_adaptive`
- `DemoConfig.kemm.inject_heuristic_detours`
- `DemoConfig.kemm.heuristic_detour_limit`
- `DemoConfig.kemm.heuristic_detour_offset_scale`

ship 侧会强制保持 `benchmark_aware_prior=False`，避免把 benchmark-only prior 混进物理场景主线。

其中最后三项是 ship 侧最近新增的场景感知绕行注入参数：

- `inject_heuristic_detours`
  控制是否把绕行初值注入到初始种群和变化响应候选中
- `heuristic_detour_limit`
  控制最多保留多少条启发式绕行种子
- `heuristic_detour_offset_scale`
  控制绕行侧向偏移幅度

这组参数属于 ship 实例化层，不改变 KEMM 通用主架构，但会明显影响 crossing 和 harbor_clutter 这类物理语义场景的恢复质量。

### 5.3 ship 报告算法入口

如果你想让 ship 报告不再固定比较 `kemm / nsga_style / random`，优先改：

- `DemoConfig.report_algorithms`

报告层现在会按这个顺序构造算法列表、生成代表性 series，并同步写入 `report_metadata.json`。后续新增算法时，不要再回到 `run_report.py` 里手工改多处列表。

---

## 6. 目标与分析指标

### 6.1 主优化目标

默认仍保留 3 目标：

- `fuel`
- `time`
- `collision risk`

### 6.2 分析指标

`EvaluationResult.analysis_metrics` 与 `PlanningEpisodeResult.analysis_metrics` 当前会输出：

- `minimum_clearance`
- `minimum_dcpa`
- `minimum_tcpa`
- `smoothness`
- `control_effort`
- `heading_variation`
- `max_yaw_rate`
- `terminal_error`
- `runtime`
- `planning_steps`
- `clearance_shortfall`
- `hard_intrusion`

这些指标默认服务于 radar、parallel、violin 和 summary 表。

另外，`ShipTrajectoryProblem` 现在默认开启了批量去重与 objective cache：

- `population_evaluation_cache`
- `population_cache_decimals`

它的目的不是改变算法语义，而是避免在同一批 population 中对完全重复或数值上等价的候选反复做整条轨迹仿真。

---

## 7. 风险模型

`core/collision_risk.py` 当前不再只看最近距离，而是组合以下因素：

- 船舶域侵入风险
- DCPA/TCPA 风险
- 静态障碍侵入惩罚
- 环境标量场暴露
- COLREG 角色缩放

在 `problem.py` 中，目标层和约束层当前已经做了更明确的语义拆分：

- `risk` 目标表达沿轨迹本身的风险暴露，当前可概括为  
  `risk = alpha * max_risk + (1 - alpha) * mean_risk + lambda_rs * safety_penalty + lambda_intr * intrusion_penalty`
- `fuel` 和 `time` 目标继续承担终端推进不足的 terminal penalty
- `cv` 只保留 `bounds_penalty + safety_penalty + intrusion_penalty`

这意味着：

- “离终点还差多远”不再伪装成“碰撞风险”
- “是否越界 / 是否侵入 / 是否净距不足”优先通过 `cv` 比较
- “是否更省油 / 更快 / 风险更低”留给 Pareto 目标去比较

这个拆分对物理语义解释非常重要，因为老师或审稿人看到 `risk` 指标时，可以把它理解成真实的安全风险，而不是被终端推进误污染的混合分数。

---

## 8. 可视化与报告

ship 默认论文图包包括：

1. 环境标量/矢量场叠加轨迹图
2. 高密障碍海域路线规划主图
3. 动态避碰时空快照图
4. 3D 时空轨迹图
5. 动力学/控制多子图时序图
6. 3D Pareto Front + Knee Point
7. 2D Pareto projection panel
8. 风险分解时间序列图
9. 安全包络图
10. 重规划变化时间轴图
11. Parallel Coordinates
12. Radar Chart
13. 带均值线和阴影的收敛曲线
14. Violin Plot
15. repeated-run 安全统计图
16. Summary dashboard

其中 `pareto3d / spatiotemporal` 在打开 `interactive_html=True` 时会额外导出 Plotly HTML，且 hover 已包含更完整的科研语义信息，而不只是坐标值。

### 8.1 representative run 与 aggregate 的边界

当前 ship 报告刻意把两类结果分开：

- `summary.csv / aggregate_summary.csv / summary.json`：聚合所有 repeated runs
- 轨迹类图：使用 representative run

代表性 run 的选择结果会写到：

- `raw/representative_runs.json`

这个文件应该和正文里的案例分析一起读；如果你在论文里同时使用路线图和统计表，必须说明前者展示代表性轨迹，后者统计所有 repeated runs。

### 8.2 图表注册表

`run_report.py` 现在不再手写逐条调用 inventory 列表，而是使用统一的 figure registry：

- `SCENARIO_FIGURE_SPECS`
- `GLOBAL_FIGURE_SPECS`

这意味着后续你要新增图时，优先补：

1. 一个 renderer
2. 一条 figure spec

然后 `figure_inventory.md` 和 `figure_manifest.json` 会自动同步。

附录图默认关闭，只在 `--appendix-plots` 时导出旧条形对比图。

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

其中：

- `summary.csv` 会带主指标、分析指标和 knee point 字段
- `planning_steps.json` 会带每次重规划的 step 摘要、knee、snapshots 和终止原因

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
python ship_simulation/run_report.py --plot-preset ieee --science-style science,ieee,no-latex
python ship_simulation/run_report.py --appendix-plots
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

---

## 13. 推荐图表与实验叙事映射

如果你在写论文或整理 GitHub 展示，当前 ship 主线的推荐叙事顺序是：

1. 用 `*_environment_overlay.png` 先交代场景、障碍、环境场和代表轨迹
2. 用 `*_snapshots.png` 展示关键交会时刻的动态避碰
3. 用 `*_spatiotemporal.png` 展示时空穿越关系
4. 用 `*_control_timeseries.png` 说明轨迹具备动力学可执行性
5. 用 `*_pareto3d.png` 解释三目标折中与 knee point 推荐解
6. 用 `*_parallel.png` 和 `*_radar.png` 展示扩展分析指标
7. 用 `*_convergence.png` 和 `*_distribution.png` 展示重复运行稳定性
8. 用 `*_dashboard.png` 做一页式总览，而不是替代主图

对应文档索引：

- 图表表达方式：`docs/figure_catalog.md`
- 图表参数与模板：`docs/visualization_guide.md`
