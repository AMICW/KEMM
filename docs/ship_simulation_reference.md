# Ship Simulation 详细说明

本文档专门解释 `ship_simulation/` 子系统，目标是让读者在不先通读全部源码的情况下，
也能理解：

1. 这个子系统解决什么问题
2. 每个目录和文件负责什么
3. 一次仿真从入口到结果是怎么流动的
4. 里面使用了哪些关键公式和工程化取舍

---

## 1. 子系统定位

`ship_simulation/` 是本仓库的物理仿真验证主线。它不使用真实 AIS 数据，而是通过代码
生成标准化会遇场景，把 KEMM 从理论动态多目标测试函数推进到具有物理语义的轨迹规划
验证问题。

它当前回答的核心问题是：

- 在对遇、交叉、追越这些典型会遇场景中，KEMM 能否找到兼顾安全、时间和燃油的轨迹
- 当问题不再是抽象测试函数，而是带运动学、环境扰动和终点约束时，KEMM 是否仍然有效

当前版本是 MVP：

- 本船模型使用 Nomoto 一阶响应
- 目标船采用恒向恒速预测
- 环境为恒定流和恒定风
- 风险模型基于船舶域连续侵入风险

这套设计的目的不是直接替代高保真海上仿真器，而是提供一个结构清晰、便于和优化算法
对接、便于写论文与做消融的中间验证平台。

---

## 2. 目录结构与职责

```text
ship_simulation/
├── core/
│   ├── ship_model.py
│   ├── environment.py
│   ├── collision_risk.py
│   └── fuel_model.py
├── scenario/
│   ├── generator.py
│   └── encounter.py
├── optimizer/
│   ├── interface.py
│   ├── problem.py
│   └── kemm_solver.py
├── visualization/
│   ├── animator.py
│   ├── report_plots.py
│   └── __init__.py
├── config.py
├── main_demo.py
└── run_report.py
```

### 2.1 `core/`

这一层只关心物理和代价模型，不关心优化算法。

- `ship_model.py`
  负责本船与目标船轨迹仿真。当前本船使用 Nomoto 一阶转向响应和速度滞后模型。
- `environment.py`
  描述恒定流场和风场，给船模和燃油模型提供扰动输入。
- `collision_risk.py`
  计算本船与目标船之间的连续碰撞风险，并返回 `max_risk / mean_risk / intrusion_time`
  等分解指标。
- `fuel_model.py`
  把速度和环境阻力积分成燃油消耗目标。

### 2.2 `scenario/`

这一层负责“生成问题”。

- `encounter.py`
  定义场景实体、船舶状态、初始条件和终点等数据结构。
- `generator.py`
  生成 `head_on / crossing / overtaking` 这三类经典会遇场景。

### 2.3 `optimizer/`

这一层负责“把仿真包装成优化问题”。

- `problem.py`
  把场景封装成三目标优化问题，是 ship 主线的核心接口。
- `interface.py`
  对外暴露统一优化接口，方便随机搜索、KEMM 或其他算法调用。
- `kemm_solver.py`
  把通用 KEMM 适配到静态船舶轨迹规划问题上，负责初始化、演化、Pareto 解选择和历史记录。

### 2.4 `visualization/`

这一层只消费结果对象，不直接依赖求解器内部细节。

- `animator.py`
  用于单次仿真的轨迹动画展示。
- `report_plots.py`
  用于批量实验报告出图，如轨迹对比、风险时序、速度剖面、Pareto 散点和汇总面板。
- `__init__.py`
  统一导出可视化函数。

### 2.5 根层脚本

- `config.py`
  管理 ship 主线的配置 dataclass。
- `main_demo.py`
  偏演示和单次运行，适合快速看某个场景的结果。
- `run_report.py`
  偏正式实验与报告导出，适合批量输出图表和表格。

---

## 3. 一次 ship 仿真的完整调用链

### 3.1 从 demo 入口出发

最常见入口是：

```bash
python -m apps.ship_runner
```

调用链大致是：

1. `apps.ship_runner.main()`
2. `ship_simulation.main_demo.run_demo()`
3. `ScenarioGenerator.generate()` 生成场景
4. `ShipOptimizerInterface` 包装问题
5. `ShipKEMMOptimizer.optimize()` 或 `random_search()` 搜索解
6. `ShipTrajectoryProblem.simulate()` 评估候选轨迹
7. `visualization.animator` 负责动画展示

### 3.2 从报告入口出发

如果要正式输出图表和表格，入口是：

```bash
python ship_simulation/run_report.py
```

调用链大致是：

1. `generate_report()` 建立输出目录
2. 遍历 `head_on / crossing / overtaking`
3. 对每个场景分别运行 KEMM 和随机 baseline
4. 将结果整理成 `ExperimentSeries`
5. 交给 `report_plots.py` 统一画图
6. 导出 `summary.csv / summary.json / summary.md`

---

## 4. 核心问题定义 `optimizer/problem.py`

这个文件是 ship 主线里最需要优先读懂的模块。

### 4.1 决策变量

当前采用固定长度编码：

\[
[x_1, y_1, v_1, x_2, y_2, v_2, \dots, x_n, y_n, v_n]
\]

其中：

- `x_i, y_i` 表示第 `i` 个中间航路点的位置
- `v_i` 表示从上一个航路点到该航路点这一段的目标航速
- 终点不参与优化编码，而由场景直接给定

这种设计的好处是：

- 容易和一般进化算法对接
- 变量边界清晰
- 后续扩展到更多航路点、速度段时很直接

### 4.2 评估流程

`ShipTrajectoryProblem.simulate()` 的内部流程如下：

1. 对输入向量做边界裁剪
2. 把越界量转换为惩罚项
3. 解码出航路点序列和速度序列
4. 调用本船动力学模型生成本船轨迹
5. 调用目标船恒速模型生成目标船轨迹
6. 计算碰撞风险
7. 计算燃油消耗
8. 加入终点残差惩罚
9. 返回 `EvaluationResult`

### 4.3 三个目标

#### 目标 1：燃油

燃油目标由轨迹积分结果与终点惩罚共同组成：

\[
J_1 = Fuel(\tau) + \lambda_f d_{term} + \lambda_b P_{bounds}
\]

其中：

- `Fuel(τ)` 是轨迹积分燃油
- `d_term` 是终点残差
- `P_bounds` 是变量越界惩罚

#### 目标 2：时间

\[
J_2 = T_{arrive} + \lambda_t d_{term} + \lambda_b P_{bounds}
\]

这里的关键是：若船没有真正到达目标点，就会因终点残差而被惩罚，从而避免“跑得少
所以时间短”的伪优解。

#### 目标 3：风险

当前风险目标使用最大风险和平均风险的加权：

\[
J_3 = 0.7 \cdot R_{max} + 0.3 \cdot R_{mean} + \lambda_r d_{term} + \lambda_b P_{bounds}
\]

这里把 `R_max` 权重设得更大，是因为瞬时高风险通常比整体均值更值得被优化器关注。

---

## 5. 动力学与环境建模

### 5.1 Nomoto 一阶转向响应

当前转向响应建模为：

\[
\dot r = \frac{K r_c - r}{T}
\]

其中：

- `r` 是实际转艏角速度
- `r_c` 是控制器给出的期望转艏角速度
- `K` 是增益
- `T` 是 Nomoto 时间常数

这个模型的优点是简单、稳定、参数少，适合 MVP 阶段快速构建轨迹规划验证平台。

### 5.2 速度滞后

速度响应为：

\[
\dot u = \frac{u_c - u}{\tau_u}
\]

意味着船速不会瞬间从一个值跳到另一个值，而是有一阶滞后过程。

### 5.3 环境模型

当前环境只包含恒定流和恒定风：

- 海流给地速叠加一个固定漂移分量
- 风影响燃油消耗估计

后续如果升级成网格流场或时变风场，主要替换的是 `environment.py` 和相关消耗模型，
不需要推翻 `problem.py` 的接口。

---

## 6. 风险模型 `core/collision_risk.py`

当前风险不是简单“最近距离小于阈值”这种二元规则，而是连续风险值。这样对多目标优化
更友好，因为优化器能感知“更安全一点”和“更危险一点”的差别。

风险评估会返回：

- `max_risk`
- `mean_risk`
- `intrusion_time`
- `risk_series`

其中：

- `max_risk` 更敏感于瞬时高危交会
- `mean_risk` 更反映整体航程中的平均危险程度
- `intrusion_time` 有助于判断是否长时间侵入高风险域
- `risk_series` 主要服务于可视化和报告分析

---

## 7. KEMM 在 ship 主线中的适配方式

`ship_simulation/optimizer/kemm_solver.py` 做的不是重新实现一个新算法，而是把通用 KEMM
安全地适配到静态轨迹规划问题中。

主要适配点包括：

1. 用 ship 问题的 `n_var / bounds / evaluate_population` 替代 benchmark 测试函数接口
2. 初始化时注入直线初始解，避免纯随机种群过慢进入可行区域
3. 在静态 ship 问题中关闭 benchmark-aware prior
4. 把演化过程中的历史目标值和 Pareto 解保存下来，供报告出图使用

这说明当前架构已经实现了“算法核心”和“应用问题”之间的解耦。

---

## 8. 可视化层说明

### 8.1 `animator.py`

适合看单次轨迹的动态过程，重点回答：

- 船是怎么转向的
- 避碰发生在哪个时间段
- 本船和目标船的空间关系如何变化

### 8.2 `report_plots.py`

适合正式汇报和批量实验，重点输出：

- 轨迹对比图
- 风险时间历程图
- 速度剖面图
- Pareto 散点图
- KEMM 收敛图
- 汇总 dashboard

统一采用 `ExperimentSeries` 作为输入，是为了让画图函数不依赖求解器内部字段。

---

## 9. 目前的工程取舍与局限

### 9.1 有意保留的简化

- 目标船使用恒向恒速，而不是交互式避碰
- 当前没有完整 COLREG 规则机理
- 当前没有静态障碍、航道边界、海图约束
- 当前没有 MMG 三自由度动力学

### 9.2 这些简化为什么仍然有价值

因为它们保留了轨迹规划最关键的三类冲突：

- 路径几何形状
- 时间与速度分配
- 安全与经济性的权衡

所以这个系统虽然不是工程部署级仿真器，但已经足够承担“算法有效性验证平台”的角色。

---

## 10. 推荐阅读顺序

如果要系统读 ship 主线，建议顺序如下：

1. `ship_simulation/config.py`
2. `ship_simulation/scenario/encounter.py`
3. `ship_simulation/scenario/generator.py`
4. `ship_simulation/optimizer/problem.py`
5. `ship_simulation/optimizer/interface.py`
6. `ship_simulation/optimizer/kemm_solver.py`
7. `ship_simulation/core/ship_model.py`
8. `ship_simulation/core/collision_risk.py`
9. `ship_simulation/visualization/report_plots.py`
10. `ship_simulation/run_report.py`

---

## 11. 建议的下一步增强方向

从研究价值和工程投入比看，后续最值得继续强化的方向是：

1. 把 Nomoto 一阶模型升级到 MMG 三自由度模型
2. 把风险目标拆成更细的可约束项，如 `max_risk / intrusion_time / min_distance`
3. 加入多船交叉和更复杂的组合会遇场景
4. 加入网格流场和海图约束
5. 增加 ship 主线的批量统计检验与消融实验
