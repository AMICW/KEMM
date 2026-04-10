# Ship 实验手册

本文档只回答一个问题：`ship_simulation` 现在应该怎么设计实验，才能和 KEMM 的机制对得上。

它更偏“实验设计与解释”，不是运行命令手册。

如果你需要：

- 命令速查：看 [run_commands.md](run_commands.md) 或 [how_to_run.md](how_to_run.md)
- ship 子系统结构说明：看 [ship_simulation_reference.md](ship_simulation_reference.md)
- 整体文档导航：看 [README.md](README.md)

---

## 1. 当前主线定位

当前 `ship_simulation` 不是远洋级大尺度航线规划，而是：

- 受限海域 / 近岸会遇环境
- 滚动重规划
- 多目标局部轨迹规划
- 简化动力学闭环可执行验证

因此，最合适的实验方向不是“全球 weather routing”，而是：

- 近海会遇
- 港区穿越
- 狭水道通行
- 动态交通与环境变化下的局部重规划

---

## 2. KEMM 机制与 ship 实验的对应关系

### 2.1 `memory`

适合用“重复出现的场景族”验证：

- 同类港区布局再次出现
- 同类潮流模式再次出现
- 同一类通道拓扑经过轻微漂移后再次出现

建议重点看：

- `route_bundle_gallery.png`
- `*_change_timeline.png`
- `*_run_statistics.png`

### 2.2 `prediction`

适合用“平滑漂移的环境变化”验证：

- 潮流持续增强
- 风险场逐步上升
- 目标船速度 / 航向逐步漂移

建议重点看：

- `*_change_timeline.png`
- `*_risk_breakdown.png`
- `*_safety_envelope.png`

### 2.3 `transfer`

适合用“拓扑相似、几何偏移”的场景变化验证：

- 同一类通道但宽度改变
- 同一类障碍群但尺度改变
- 同一类会遇结构但交通体相对位置偏移

建议重点看：

- `scenario_gallery.png`
- `*_route_planning_panel.png`
- `*_pareto_projection.png`

### 2.4 `adaptive`

适合用“变化规律突然切换”验证：

- 平稳漂移后突然封航
- 风险场温和变化后突然插入 closure
- 交通意图从平滑变化切到突变

建议重点看：

- `*_change_timeline.png`
- `*_risk_breakdown.png`
- `*_control_timeseries.png`

---

## 3. 已内置的 experiment profile

当前 `ship_simulation/run_report.py` 已支持：

- `baseline`
- `drift`
- `shock`
- `recurring_harbor`

CLI 示例：

```powershell
python ship_simulation/run_report.py --experiment-profile baseline
python ship_simulation/run_report.py --scenarios harbor_clutter --experiment-profile drift
python ship_simulation/run_report.py --scenarios harbor_clutter --experiment-profile shock
python ship_simulation/run_report.py --scenarios harbor_clutter --experiment-profile recurring_harbor
```

### 3.1 `baseline`

含义：

- 不注入额外动态变化
- 只验证默认场景与默认滚动重规划

适用：

- 论文主结果的基线组
- smoke test

### 3.2 `drift`

含义：

- 在后续 planning step 中逐步增强流场 / 风险场
- 同时轻度改变目标船速度与意图

适用：

- 检验 `prediction`
- 检验变化后恢复速度

### 3.3 `shock`

含义：

- 在指定 step 注入 sudden closure
- 同时伴随更强的交通偏转与环境扰动

适用：

- 检验 `adaptive`
- 检验极端变化下的风险峰值与恢复能力

### 3.4 `recurring_harbor`

含义：

- 港区场景在中间发生漂移，然后部分回到熟悉模式

适用：

- 检验 `memory + transfer`

---

## 4. 最推荐的实验矩阵

如果你现在只想做一套能写论文的 ship 实验，我建议先跑这 4 组。

### 4.1 静态难度分层

- 场景：`head_on / crossing / overtaking / harbor_clutter`
- 配置：只改 `ProblemConfig.scenario_generation.*`
- 目的：验证算法在障碍密度、交通密度和环境强度上的稳定性

### 4.2 港区迁移组

- 场景：`harbor_clutter`
- profile：`recurring_harbor`
- 配置：调 `circular_radius_scale / polygon_scale / channel_width_scale`
- 目的：验证 `memory + transfer`

### 4.3 平滑漂移组

- 场景：`crossing` 或 `harbor_clutter`
- profile：`drift`
- 目的：验证 `prediction`

### 4.4 突变冲击组

- 场景：`harbor_clutter`
- profile：`shock`
- 目的：验证 `adaptive`

---

## 5. 最重要的配置入口

### 5.1 静态几何与环境复杂度

改：

- `ProblemConfig.scenario_generation.head_on`
- `ProblemConfig.scenario_generation.crossing`
- `ProblemConfig.scenario_generation.overtaking`
- `ProblemConfig.scenario_generation.harbor_clutter`

### 5.2 动态实验 profile

改：

- `ProblemConfig.experiment`

或者直接用：

- `apply_experiment_profile(config, "drift")`
- `apply_experiment_profile(config, "shock")`
- `apply_experiment_profile(config, "recurring_harbor")`

### 5.3 滚动重规划尺度

改：

- `DemoConfig.episode.local_horizon`
- `DemoConfig.episode.execution_horizon`
- `DemoConfig.episode.max_replans`

这组参数决定你是在做更“短视”的战术规划，还是更“长视”的局部规划。

---

## 6. 推荐优先看的图

如果你只想快速判断实验是否合理，优先看：

1. `scenario_gallery.png`
2. `route_bundle_gallery.png`
3. `*_route_planning_panel.png`
4. `*_change_timeline.png`
5. `*_risk_breakdown.png`
6. `*_safety_envelope.png`

其中：

- `scenario_gallery` 负责解释实验场景本身
- `route_bundle_gallery` 负责解释重复运行稳定性
- `route_planning_panel` 负责解释代表路径
- `change_timeline` 负责解释动态实验触发点
- `risk_breakdown / safety_envelope` 负责解释风险响应是否合理

---

## 7. 一个最实用的起点

如果你现在只打算先改一轮实验，不要同时改太多。

建议从这一条开始：

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper --experiment-profile shock
```

然后重点检查：

- `harbor_clutter_change_timeline.png`
- `harbor_clutter_route_planning_panel.png`
- `harbor_clutter_risk_breakdown.png`
- `harbor_clutter_safety_envelope.png`

如果这 4 张图讲不清楚“变化是什么、KEMM 怎么响应、响应后风险是否恢复”，说明实验设计还不够好。
