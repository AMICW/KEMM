# KEMM 船舶专利实施例参数表与结果表

本文档用于补足专利说明书“具体实施方式”中的表格材料。文档分为两层：

- 可直接转入正式说明书的表格
- 仅内部留档、供代理人和老师参考的结果支撑表

---

## 1. 使用说明

建议正式说明书优先吸收：

- 表1 核心物理与规划参数
- 表2 KEMM 优选实施参数
- 表3 交叉会遇场景参数
- 表4 港口高密障碍场景参数
- 表5 对比例设置
- 表6 可用于说明技术效果的对比结果

建议仅内部留档：

- 表7 研发版前后改进结果

---

## 2. 表1 核心物理与规划参数

数据来源：

- [ship_simulation/config.py](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/config.py#L1)

| 类别 | 参数 | 数值 | 含义 |
| --- | --- | ---: | --- |
| 船舶物理 | `length` | 120.0 m | 船长 |
| 船舶物理 | `beam` | 20.0 m | 船宽 |
| 船舶物理 | `draft` | 7.0 m | 吃水 |
| 船舶控制 | `nomoto_k` | 1.0 | Nomoto 增益 |
| 船舶控制 | `nomoto_t` | 25.0 s | Nomoto 时间常数 |
| 船舶控制 | `heading_gain` | 1.6 | 航向误差控制增益 |
| 船舶控制 | `max_turn_rate_deg` | 3.0 deg/s | 最大转向角速度 |
| 船舶控制 | `speed_time_constant` | 35.0 s | 速度一阶滞后常数 |
| 速度约束 | `max_speed` | 9.0 m/s | 最大速度 |
| 速度约束 | `min_speed` | 2.0 m/s | 最小速度 |
| 环境 | `current_speed` | 0.6 m/s | 基础海流速度 |
| 环境 | `current_direction_deg` | 35.0 deg | 基础海流方向 |
| 环境 | `wind_speed` | 6.0 m/s | 基础风速 |
| 环境 | `wind_direction_deg` | 75.0 deg | 基础风向 |
| 环境 | `wind_drift_coefficient` | 0.015 | 风致漂移系数 |
| 仿真 | `dt` | 5.0 s | 积分步长 |
| 仿真 | `horizon` | 1800.0 s | 单次轨迹仿真总时域 |
| 仿真 | `arrival_tolerance` | 80.0 m | 到达容差 |
| 仿真区域 | `area` | `(-1000,9000,-4000,4000)` | 平面规划区域 |
| 重规划 | `local_horizon` | 360.0 s | 局部规划时域 |
| 重规划 | `execution_horizon` | 180.0 s | 实际执行时域 |
| 重规划 | `max_replans` | 8 | 最大重规划次数 |
| 安全 | `safety_clearance` | 180.0 m | 安全净距阈值 |
| 问题建模 | `num_intermediate_waypoints` | 3 | 中间航路点个数 |
| 速度边界 | `speed_bounds` | `(3.0, 8.5)` m/s | 优化变量速度范围 |

---

## 3. 表2 KEMM 优选实施参数

数据来源：

- [ship_simulation/config.py](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/config.py#L188)
- [kemm/core/types.py](d:/VS%20Code/code/KEMM%28renew%29/kemm/core/types.py#L20)

### 3.1 ship 侧运行参数

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| `pop_size` | 30 | ship 侧种群规模 |
| `generations` | 14 | 每次局部规划迭代代数 |
| `refresh_interval` | 8 | 状态刷新间隔 |
| `seed` | 42 | 随机种子 |
| `inject_initial_guess` | `True` | 是否注入初始猜测 |
| `initial_guess_copies` | 4 | 初始猜测复制数 |
| `initial_guess_jitter_ratio` | 0.04 | 初始猜测扰动比例 |
| `use_change_response` | `True` | 是否启用变化响应 |
| `reuse_solver_state_across_replans` | `False` | 是否跨重规划复用求解状态 |
| `benchmark_aware_prior` | `False` | ship 侧关闭 benchmark-only prior |

### 3.2 KEMM 核心机制参数

| 类别 | 参数 | 数值 | 含义 |
| --- | --- | ---: | --- |
| 自适应分配 | `exploration_c` | 0.5 | UCB 探索系数 |
| 自适应分配 | `reward_window` | 10 | 奖励窗口长度 |
| 自适应分配 | `temperature` | 0.8 | softmax 温度 |
| 自适应分配 | `min_operator_ratio` | 0.05 | 单机制最小比例 |
| 记忆 | `memory_capacity` | 50 | 历史记忆库容量 |
| 记忆 | `memory_hidden_dim` | 64 | 记忆压缩隐层维度 |
| 记忆 | `memory_beta` | 0.1 | beta-VAE 权重 |
| 记忆 | `memory_online_epochs` | 15 | 在线训练轮数 |
| 记忆 | `memory_top_k` | 3 | 相似环境检索数量 |
| 记忆 | `memory_min_keep` | 4 | 最少记忆样本保留数 |
| 预测 | `drift_window` | 6 | 漂移分析窗口长度 |
| 预测 | `drift_feature_dim` | 10 | 前沿特征维度 |
| 预测 | `drift_history` | 8 | 历史序列长度 |
| 预测 | `gp_lengthscale` | 1.0 | GP 长度尺度 |
| 预测 | `gp_noise_var` | 0.05 | GP 噪声方差 |
| 预测 | `prediction_confidence_threshold` | 0.3 | 预测回退阈值 |
| 预测 | `prediction_pool_multiplier` | 2 | 预测候选扩展系数 |
| 预测 | `prediction_min_pool` | 8 | 最小预测候选数 |
| 迁移 | `transfer_n_clusters` | 4 | 聚类数 |
| 迁移 | `transfer_n_subspaces` | 5 | 子空间数 |
| 迁移 | `transfer_var_threshold` | 0.95 | PCA 累计方差阈值 |
| 迁移 | `transfer_target_sample_size` | 30 | 目标锚点样本数 |
| 候选池 | `elite_keep_min` | 6 | 精英最小保留数 |
| 候选池 | `elite_keep_fraction` | 0.10 | 精英保留比例 |
| 候选池 | `previous_keep_min` | 10 | 历史种群最小保留数 |
| 候选池 | `previous_keep_fraction` | 0.20 | 历史种群保留比例 |

---

## 4. 表3 交叉会遇场景参数

数据来源：

- [ship_simulation/scenario/generator.py](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/scenario/generator.py#L293)

| 对象 | 参数 | 数值/描述 |
| --- | --- | --- |
| 本船 | 初始位置 | `(0.0, -480.0)` |
| 本船 | 初始航向 | `0.03 rad` |
| 本船 | 初始速度 | `6.2 m/s` |
| 本船 | 目标位置 | `(7300.0, 320.0)` |
| 目标船A | 初始位置 | `(3250.0, -3200.0)` |
| 目标船A | 初始航向 | `π/2` |
| 目标船A | 初始速度 | `5.7 m/s` |
| 目标船A | 规则角色 | `crossing_give_way` |
| 目标船B | 初始位置 | `(5400.0, 1700.0)` |
| 目标船B | 初始航向 | `-π/2` |
| 目标船B | 初始速度 | `4.8 m/s` |
| 目标船B | 规则角色 | `crossing_stand_on` |
| 静态障碍物 | 圆形障碍物 | `Islet`，中心 `(3600.0, -150.0)`，半径 `420.0 m` |
| 静态障碍物 | 禁入区 | `Restricted Zone` 四边形 |
| 风险场 | 高斯风险场 | `Harbor Approach Risk` |
| 风险场 | 网格风险场 | `Shallow Patch` |
| 流场 | 均匀流场 | `Littoral Current` |
| 流场 | 网格流场 | `Coastal Shear` |
| 流场 | 涡流场 | `Breakwater Eddy` |
| 元数据 | 场景标签 | `crossing`、`island`、`risk_field`、`multi_ship` |

可直接写入说明书的实施例描述：

“在一优选实施例中，交叉会遇场景中本船由坐标 `(0.0,-480.0)` 起航，目标点为 `(7300.0,320.0)`，场景中设置两艘目标船、一处圆形岛礁障碍、一处禁入区以及一组风险场和流场，用于模拟典型交叉会遇条件下的动态避碰与路径规划问题。”

---

## 5. 表4 港口高密障碍场景参数

数据来源：

- [ship_simulation/scenario/generator.py](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/scenario/generator.py#L445)
- [ship_simulation/config.py](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/config.py#L94)

### 5.1 原始场景定义

| 对象 | 参数 | 数值/描述 |
| --- | --- | --- |
| 本船 | 初始位置 | `(150.0, -1450.0)` |
| 本船 | 初始航向 | `0.08 rad` |
| 本船 | 初始速度 | `5.9 m/s` |
| 本船 | 目标位置 | `(7650.0, 1260.0)` |
| 目标船 | 原始数量 | `3` 艘 |
| 边界障碍物 | 数量 | `2` 个港区边界 |
| 圆形障碍物 | 数量 | `7` 个 |
| 禁入区 | 数量 | `4` 个 |
| 风险场 | 数量 | `4` 个 |
| 流场 | 数量 | `3` 个 |

### 5.2 默认调优后实际生成参数

由于默认 `HarborClutterTuningConfig` 会限制目标船和障碍物数量，因此优选实施方式下的实际生成规模为：

| 项目 | 数值 | 说明 |
| --- | ---: | --- |
| `target_limit` | 2 | 优选保留前两艘目标船 |
| `circular_obstacle_limit` | 6 | 优选保留 6 个圆形障碍物 |
| `polygon_obstacle_limit` | 3 | 优选保留 3 个禁入区 |
| `channel_width_scale` | 1.15 | 港区通道纵向尺度调节 |
| `own_speed_scale` | 1.04 | 本船速度缩放 |
| `target_speed_scale` | 0.9 | 目标船速度缩放 |
| `vector_speed_scale` | 0.86 | 流场速度缩放 |

可直接写入说明书的实施例描述：

“在另一优选实施例中，港口高密障碍场景中本船由 `(150.0,-1450.0)` 起航，目标点为 `(7650.0,1260.0)`。场景中设置受限港区边界、多个圆形障碍物和禁入区，并优选保留两艘目标船、六个圆形障碍物和三个禁入区，以形成窄通道、高密障碍和多船交互并存的受限水域规划环境。”

---

## 6. 表5 对比例设置建议

该表可供说明书或后续答复中使用。

| 方案编号 | 方案名称 | 保留机制 | 删除或弱化机制 | 用途 |
| --- | --- | --- | --- | --- |
| 实施例E1 | 完整 KEMM 方案 | 记忆、预测、迁移、重初始化、自适应分配、统一候选池、安全优先选择 | 无 | 作为本发明优选实施例 |
| 对比例C1 | 纯重初始化方案 | 重初始化、统一环境选择 | 删除记忆、预测、迁移、自适应分配 | 对应完全重启型现有思路 |
| 对比例C2 | 无预测方案 | 记忆、迁移、重初始化、自适应分配 | 删除前沿预测 | 验证预测模块作用 |
| 对比例C3 | 无记忆方案 | 预测、迁移、重初始化、自适应分配 | 删除历史记忆 | 验证压缩记忆作用 |
| 对比例C4 | 无迁移方案 | 记忆、预测、重初始化、自适应分配 | 删除跨环境迁移 | 验证迁移模块作用 |
| 对比例C5 | 仅保留旧种群方案 | 旧种群保留、统一环境选择 | 删除变化诊断和多源候选 | 对应机械复用旧解思路 |

---

## 7. 表6 可用于说明书的技术效果对比结果

数据来源：

- [ship_simulation/outputs/report_20260409_222555/reports/summary.md](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/outputs/report_20260409_222555/reports/summary.md#L1)

### 7.1 交叉会遇场景结果

| 算法 | Fuel | Time | Risk | Clearance | Ship Dist | Runtime | Success Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KEMM | 19909.456 | 927.661 | 1.844 | 147.842 | 828.524 | 42.290 | 100.00% |
| NSGA-style | 19910.024 | 944.607 | 4.553 | 91.332 | 802.169 | 55.799 | 100.00% |
| Random | 21418.996 | 1007.170 | 0.578 | 225.406 | 310.973 | 5.427 | 100.00% |

可写入说明书的结果分析句式：

“在交叉会遇场景下，本发明方法在燃油消耗、航行时间和综合风险之间获得了更均衡的多目标解。与对比的进化式基线相比，本发明方法的综合风险由 `4.553` 降至 `1.844`，最小安全净距由 `91.332m` 提升至 `147.842m`，且保持 `100%` 的到达成功率。”

### 7.2 港口高密障碍场景结果

| 算法 | Fuel | Time | Risk | Clearance | Ship Dist | Runtime | Success Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KEMM | 23318.617 | 913.861 | 0.392 | 193.712 | 199.125 | 107.320 | 100.00% |
| NSGA-style | 20884.815 | 965.342 | 1.520 | 159.836 | 159.836 | 175.573 | 100.00% |
| Random | 16050.513 | 1055.084 | 0.645 | 193.818 | 236.840 | 14.201 | 100.00% |

可写入说明书的结果分析句式：

“在港口高密障碍场景下，本发明方法的综合风险为 `0.392`，低于对比进化式基线的 `1.520` 和随机基线的 `0.645`，同时最小安全净距达到 `193.712m`，表明本发明方法在高密障碍受限水域中能够兼顾安全性与规划有效性。”

---

## 8. 表7 内部研发改进结果对照表

该表建议仅作为内部留档材料，不建议直接进入正式说明书。

数据来源：

- 旧版本报告：[report_20260409_第二版201213](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/outputs/report_20260409_第二版201213/reports/summary.md#L1)
- 修正后报告：[report_20260409_222555](d:/VS%20Code/code/KEMM%28renew%29/ship_simulation/outputs/report_20260409_222555/reports/summary.md#L1)

| 场景 | 指标 | 旧版本 KEMM | 修正后 KEMM | 变化 |
| --- | --- | ---: | ---: | ---: |
| crossing | Risk | 8.803 | 1.844 | -6.959 |
| crossing | Clearance | 6.785 | 147.842 | +141.057 |
| crossing | Runtime | 119.130 | 42.290 | -76.840 |
| harbor_clutter | Risk | 1.944 | 0.392 | -1.552 |
| harbor_clutter | Clearance | 149.286 | 193.712 | +44.426 |
| harbor_clutter | Runtime | 272.180 | 107.320 | -164.860 |

内部备注：

- 该表更适合证明当前实现版本已经收敛到较合理状态；
- 若未来审查阶段需要补送实验或对比例说明，可作为内部依据重新组织。

---

## 9. 可直接写入说明书的表格优先级

如果后续只准备在说明书中放少量表格，建议优先级如下：

1. 表1 核心物理与规划参数
2. 表2 KEMM 优选实施参数
3. 表3 交叉会遇场景参数
4. 表4 港口高密障碍场景参数
5. 表6 技术效果对比结果

如果篇幅更紧，可进一步压缩为：

1. 一张总参数表
2. 一张实施例场景表
3. 一张结果表

