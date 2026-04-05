# Ship Simulation 参考文档

本文档面向三类读者：

1. 想快速理解 `ship_simulation/` 架构的开发者
2. 需要定位修改入口的新 AI 助手
3. 需要把系统结构写进论文或 GitHub 文档的作者

当前 ship 主线已经从“单次黑盒轨迹 demo”升级为“偏现实场景 + 滚动重规划 + 论文级图包”的实验系统。

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

### 2.3 `optimizer/`

- `problem.py`
  ship 主线的核心问题定义，负责把场景包装成三目标优化问题
- `interface.py`
  对外暴露统一优化接口，保持 KEMM、随机基线、NSGA-style 基线复用同一问题层
- `kemm_solver.py`
  ship 侧 KEMM 适配器
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
7. 继续下一轮重规划，直到到达、风险切断或达到最大重规划次数

当前代表解选择规则不是简单“目标最小”，而是：

1. 优先不侵入障碍
2. 再优先满足 `safety_clearance`
3. 再比较终点推进程度
4. 最后才比较多目标折中分数

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

---

## 7. 风险模型

`core/collision_risk.py` 当前不再只看最近距离，而是组合以下因素：

- 船舶域侵入风险
- DCPA/TCPA 风险
- 静态障碍侵入惩罚
- 环境标量场暴露
- COLREG 角色缩放

在 `problem.py` 中，`safety_clearance`、实际侵障和高风险暴露还会被继续转成目标惩罚，避免滚动 episode 在局部窗口里“先贴边再说”。

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
10. Parallel Coordinates
11. Radar Chart
12. 带均值线和阴影的收敛曲线
13. Violin Plot
14. repeated-run 安全统计图
15. Summary dashboard

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
