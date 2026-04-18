# Ship 实验手册

本文档只回答一个问题：`ship_simulation` 现在应该怎么设计实验，才能和当前实现里的 KEMM 机制、ship 预算 profile 与滚动重规划逻辑对得上。

它更偏“实验设计与解释”，不是命令速查表。

如果你需要：

- 命令速查：看 [run_commands.md](run_commands.md) 或 [how_to_run.md](how_to_run.md)
- ship 子系统结构说明：看 [ship_simulation_reference.md](ship_simulation_reference.md)
- 整体文档导航：看 [README.md](README.md)

---

## 1. 当前主线定位

当前 `ship_simulation` 不是远洋 weather routing，也不是只跑一次的静态 demo，而是：

- 受限海域 / 近岸会遇环境
- 滚动重规划
- 多目标局部轨迹规划
- 简化动力学闭环可执行验证
- 完整 batch report 和论文图包导出

因此，最合适的实验方向不是“全球航线级最优”，而是：

- 近海会遇
- 港区穿越
- 狭水道通行
- 动态交通与环境变化下的局部重规划

---

## 2. 先区分两套 profile

ship 现在有两套不同目的的 profile。文档和实验设计时必须把这两套东西分开说，不然很容易把“动态变化事件”和“求解预算调优”混成一件事。

### 2.1 `ProblemConfig.experiment`

这一套 profile 控制的是 rolling-horizon 过程中注入什么变化。

当前内置：

- `baseline`
- `drift`
- `shock`
- `recurring_harbor`

它们影响的是：

- 环境标量场强度缩放
- 环境矢量场强度缩放
- 背景 current 强度缩放
- 目标船速度缩放
- 目标船航向偏移
- 通道宽度收缩
- 临时 closure 障碍注入

这套 profile 用来验证 `prediction / adaptive / memory / transfer` 等变化响应语义。

### 2.2 `DemoConfig.scenario_profiles`

这一套 profile 控制的是不同场景到底给多少求解预算、局部时域和安全权重。

当前有两组：

- `legacy_uniform`
- `full_tuned`

它们影响的是：

- `random_search_samples`
- `evolutionary_baseline_pop_size / generations`
- `kemm_pop_size / generations`
- `kemm_initial_guess_copies`
- `kemm_heuristic_detour_limit`
- `kemm_reuse_solver_state`
- `local_horizon / execution_horizon / max_replans`
- `objective_weights`
- `safety_clearance`
- 净空惩罚与风险权重

这套 profile 用来控制“完整实验怎么更合理、更快、也更利于 ship 结果稳定”。

### 2.3 当前默认口径

当前 CLI 的默认行为是：

- `python ship_simulation/run_report.py --quick`
  - 使用 `legacy_uniform`
- `python ship_simulation/run_report.py`
  - 使用 `full_tuned`

因此你在写实验说明时，至少要把这两层都交代清楚：

1. 这次有没有注入 `baseline / drift / shock / recurring_harbor` 之类的动态变化事件
2. 这次用的是 `legacy_uniform` 还是 `full_tuned` 预算配置

---

## 3. `full_tuned` 的场景级求解配置

如果你现在要做完整 ship 实验，默认应以 `full_tuned` 为主。下面这些值就是当前代码里的实际口径。

### 3.1 `head_on`

- `random_search_samples = 32`
- `evolutionary_baseline_pop_size = 28`
- `evolutionary_baseline_generations = 10`
- `kemm_pop_size = 24`
- `kemm_generations = 10`
- `kemm_initial_guess_copies = 3`
- `kemm_heuristic_detour_limit = 2`
- `kemm_reuse_solver_state = True`
- `local_horizon = 300`
- `execution_horizon = 240`
- `max_replans = 6`
- `objective_weights = (1.0, 1.0, 1.10)`
- 其余 penalty 继承统一默认值

解释：

- `head_on` 结构简单，主要矛盾是规则顺从与稳定让路，不需要太大的搜索预算。
- 允许复用 solver state，适合重复 replanning 中的连续收敛。

### 3.2 `crossing`

- `random_search_samples = 40`
- `evolutionary_baseline_pop_size = 32`
- `evolutionary_baseline_generations = 12`
- `kemm_pop_size = 30`
- `kemm_generations = 12`
- `kemm_initial_guess_copies = 4`
- `kemm_heuristic_detour_limit = 4`
- `kemm_reuse_solver_state = False`
- `local_horizon = 420`
- `execution_horizon = 210`
- `max_replans = 6`
- `objective_weights = (1.0, 0.95, 1.35)`
- `safety_clearance = 210`
- `soft_clearance_penalty_per_meter = 2.6`
- `hard_clearance_penalty_per_meter = 18.0`
- `risk_safety_penalty_weight = 0.035`
- `domain_risk_weight = 0.60`

解释：

- `crossing` 是最容易出现“直穿高风险区”的场景之一，所以需要更强的风险项权重和更高的净空约束。
- 关闭 solver state 复用，是为了减少把上一轮局部几何结构硬带到下一轮的风险。

### 3.3 `overtaking`

- `random_search_samples = 32`
- `evolutionary_baseline_pop_size = 28`
- `evolutionary_baseline_generations = 10`
- `kemm_pop_size = 26`
- `kemm_generations = 10`
- `kemm_initial_guess_copies = 3`
- `kemm_heuristic_detour_limit = 3`
- `kemm_reuse_solver_state = True`
- `local_horizon = 320`
- `execution_horizon = 240`
- `max_replans = 6`
- `objective_weights = (1.0, 0.95, 1.25)`
- `safety_clearance = 200`
- `soft_clearance_penalty_per_meter = 2.4`
- `hard_clearance_penalty_per_meter = 18.0`
- `time_safety_penalty_weight = 0.30`
- `risk_safety_penalty_weight = 0.032`
- `domain_risk_weight = 0.60`

解释：

- `overtaking` 通常没有 `crossing` 那么强的横向冲突，但对净空和跟船风险依然敏感。
- 这里允许一定程度的 state reuse，因为局部拓扑通常比 `crossing` 更平滑。

### 3.4 `harbor_clutter`

- `random_search_samples = 40`
- `evolutionary_baseline_pop_size = 30`
- `evolutionary_baseline_generations = 12`
- `kemm_pop_size = 28`
- `kemm_generations = 12`
- `kemm_initial_guess_copies = 4`
- `kemm_heuristic_detour_limit = 4`
- `kemm_reuse_solver_state = False`
- `local_horizon = 420`
- `execution_horizon = 240`
- `max_replans = 6`
- `objective_weights = (1.0, 1.0, 1.45)`
- `safety_clearance = 220`
- `soft_clearance_penalty_per_meter = 2.8`
- `hard_clearance_penalty_per_meter = 18.0`
- `risk_safety_penalty_weight = 0.040`
- `domain_risk_weight = 0.62`

解释：

- `harbor_clutter` 是当前最重的场景：障碍密度高、通道受限、局部几何变化复杂。
- 因此它得到最高的一组风险权重与较大的局部时域。
- 关闭 solver state 复用，是为了减少把上一轮局部拥堵拓扑错误延续到下一轮。

---

## 4. 动态 experiment profile 应该怎么用

### 4.1 `baseline`

含义：

- 不注入额外动态变化
- 只验证默认场景与默认滚动重规划

适用：

- 论文主结果
- `full_tuned` 与 `legacy_uniform` 的 A/B
- smoke test

### 4.2 `drift`

含义：

- 在后续 planning step 中逐步增强流场 / 风险场
- 同时轻度改变目标船速度与意图

适用：

- 检验 `prediction`
- 检验变化后恢复速度
- 检验 risk breakdown 是否随漂移平滑变化

### 4.3 `shock`

含义：

- 在指定 step 注入 sudden closure
- 同时伴随更强的交通偏转与环境扰动

适用：

- 检验 `adaptive`
- 检验极端变化下的风险峰值与恢复能力
- 检验 representative route 是否会明显切换

### 4.4 `recurring_harbor`

含义：

- 港区场景在中间发生漂移，然后部分回到熟悉模式

适用：

- 检验 `memory + transfer`
- 检验旧知识在回归拓扑中的复用价值

---

## 5. 最推荐的实验矩阵

如果你现在只想做一套“既能写论文主结果、又能做机制验证”的 ship 实验，建议按下面这个顺序走。

### 5.1 主结果组：`baseline + full_tuned`

- 场景：`head_on / crossing / overtaking / harbor_clutter`
- event profile：`baseline`
- solve profile：`full_tuned`
- 目的：形成当前 ship 主结果和完整图包

这是最应该先跑的一组，因为 README 和默认完整命令就是围绕这组配置写的。

### 5.2 回归对比组：`baseline + legacy_uniform`

- 场景：同上
- event profile：`baseline`
- solve profile：`legacy_uniform`
- 目的：验证 tuned profile 是否真的改善了完整实验设置，而不是偶然更换了任务定义

这一组最适合写成“配置层面增强，而非核心机制变化”的对照。

### 5.3 平滑漂移组：`drift + full_tuned`

- 场景：优先 `crossing` 或 `harbor_clutter`
- 目的：验证 `prediction`
- 建议重点看：`*_change_timeline.png`, `*_risk_breakdown.png`, `*_operator_allocation.png`

### 5.4 突变冲击组：`shock + full_tuned`

- 场景：优先 `harbor_clutter`
- 目的：验证 `adaptive`
- 建议重点看：`*_change_timeline.png`, `*_snapshots.png`, `*_control_timeseries.png`

### 5.5 回归拓扑组：`recurring_harbor + full_tuned`

- 场景：`harbor_clutter`
- 目的：验证 `memory + transfer`
- 建议重点看：`*_route_planning_panel.png`, `*_pareto_projection.png`, `*_convergence.png`

---

## 6. 最重要的配置入口

### 6.1 静态几何与环境复杂度

改：

- `ProblemConfig.scenario_generation.head_on`
- `ProblemConfig.scenario_generation.crossing`
- `ProblemConfig.scenario_generation.overtaking`
- `ProblemConfig.scenario_generation.harbor_clutter`

### 6.2 动态 experiment profile

改：

- `ProblemConfig.experiment`

或者直接用：

- `apply_experiment_profile(config, "drift")`
- `apply_experiment_profile(config, "shock")`
- `apply_experiment_profile(config, "recurring_harbor")`

### 6.3 场景求解 profile

改：

- `DemoConfig.scenario_profiles.active_profile_name`

典型值：

- `full_tuned`
- `legacy_uniform`

### 6.4 KEMM 运行时机制开关

改：

- `DemoConfig.kemm`
- `DemoConfig.kemm.runtime`

### 6.5 滚动重规划尺度

改：

- `DemoConfig.episode.local_horizon`
- `DemoConfig.episode.execution_horizon`
- `DemoConfig.episode.max_replans`

注意：如果你已经用 `full_tuned`，再去单独改这组值，本质上就是在偏离默认场景 solve profile 了。写实验说明时需要把这点讲清楚。

### 6.6 严格工程可比组

如果你要做“同预算、同重规划频率”的可比实验，不要手工复制参数，直接在报告命令上开启：

- `--strict-comparable`

它会在原算法组之外自动追加：

- `random_matched`
- `nsga_style_matched`

这两个分组会按 KEMM 当前预算与场景 profile 自动对齐。

### 6.7 统计显著性与鲁棒性输出

报告默认会导出：

- `raw/statistical_tests.json`
- `raw/statistical_tests.csv`
- `reports/statistical_significance.md`

如果你要做扰动强度扫描，再加：

- `--robustness-sweep`
- `--robustness-levels`
- `--robustness-scenarios`

对应导出：

- `raw/robustness_runs.csv`
- `raw/robustness_curve.csv`
- `raw/robustness_summary.json`
- `reports/robustness_sweep.md`
- `figures/robustness_success_curve.png`（启用图渲染时）

---

## 7. 推荐优先看的图

如果你只想快速判断实验是否合理，优先看：

1. `scenario_gallery.png`
2. `route_bundle_gallery.png`
3. `*_route_planning_panel.png`
4. `*_change_timeline.png`
5. `*_risk_breakdown.png`
6. `*_safety_envelope.png`
7. `*_operator_allocation.png`
8. `*_dashboard.png`

其中：

- `scenario_gallery` 负责解释场景本身
- `route_bundle_gallery` 负责解释 repeated-run 稳定性
- `route_planning_panel` 负责解释 representative path
- `change_timeline` 负责解释动态实验触发点
- `risk_breakdown / safety_envelope` 负责解释安全响应是否合理
- `operator_allocation` 负责解释 KEMM 在 episode 内如何重新分配机制权重
- `dashboard` 负责做最终汇总，不替代单图分析

---

## 8. 一个最实用的起点

如果你现在只打算先改一轮实验，不要同时改太多。建议从下面这两条开始：

完整主结果：

```powershell
python ship_simulation/run_report.py --workers 4
```

严格可比 + 统计显著性：

```powershell
python ship_simulation/run_report.py --workers 4 --strict-comparable
```

扰动扫描：

```powershell
python ship_simulation/run_report.py --workers 4 --robustness-sweep --robustness-levels 0,0.25,0.5,0.75,1.0 --robustness-scenarios crossing overtaking harbor_clutter
```

高风险机制验证：

```powershell
python ship_simulation/run_report.py --scenarios harbor_clutter --n-runs 3 --plot-preset paper --experiment-profile shock
```

然后重点检查：

- `harbor_clutter_change_timeline.png`
- `harbor_clutter_route_planning_panel.png`
- `harbor_clutter_risk_breakdown.png`
- `reports/summary.md`
- `raw/report_metadata.json`

---

## 9. 结论性建议

如果你在写论文或整理仓库说明，当前最稳妥的 ship 叙事是：

1. 主结果默认以 `baseline + full_tuned` 为准
2. `legacy_uniform` 只作为回归比较，不作为新的默认完整配置
3. `drift / shock / recurring_harbor` 主要用来解释 KEMM 各模块的变化响应语义
4. 结果表统计全部 repeated runs，而轨迹类图使用 representative run，这个口径必须明确写出
5. 完整报告已经是“episode compute -> figure render -> summary”的 staged pipeline，但对外仍保持一步命令跑完
