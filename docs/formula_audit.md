# KEMM 公式与实现审查记录

本文档的目的不是重复写一遍论文，而是明确区分下面三类内容：

1. 文献公式与代码实现基本一致
2. 工程启发式实现，但自洽且可运行
3. 当前已发现的不一致、边界条件或仍需继续审查的地方

---

## 1. 总体结论

当前项目的主要数学结构总体上是可运行且基本自洽的，但并不是每个部分都能严格等价为“标准文献原式”。

最重要的区分是：

- `UCB1 + softmax 比例分配` 属于工程启发式，不是标准单臂 UCB1 决策
- `benchmark-aware candidate generation` 属于 benchmark 强化逻辑，不应被表述成通用理论模块
- `ship_simulation` 的风险模型与动力学模型当前都是工程型 MVP，不是完整航海规则或高保真动力学模型

---

## 2. 审查范围

当前已覆盖的模块：

- `adaptive_operator.py`
- `compressed_memory.py`
- `pareto_drift.py`
- `geodesic_flow.py`
- `kemm/algorithms/kemm.py`
- `kemm/benchmark/problems.py`
- `kemm/benchmark/metrics.py`
- `ship_simulation` 中与目标函数、动力学和风险有关的核心模块

---

## 3. 模块级审查

### 3.1 `adaptive_operator.py`

状态：`工程启发式，自洽，可用`

#### 已确认正确的部分

UCB1 主体公式与标准形式一致：

\[
\mathrm{UCB}_i = Q_i + c \sqrt{\frac{\ln T}{N_i}}
\]

代码中关键量的对应关系：

- `Q_i`：滑动窗口中的平均 reward
- `T`：`total_count`
- `N_i`：`counts[i]`
- `c`：探索系数

#### 需要明确说明的部分

当前实现不是“选一个 arm”，而是把 UCB 值做 softmax，转成各策略的样本比例：

\[
p_i = \frac{\exp(\mathrm{UCB}_i / \tau)}{\sum_j \exp(\mathrm{UCB}_j / \tau)}
\]

因此更准确的说法应该是：

- `采用 UCB1 评分，并通过 softmax 形成多策略样本分配比例`
- 而不是：`严格使用标准 UCB1 做单臂决策`

#### 额外发现

reward 分润逻辑已经从“多策略平均领功”改成“主导策略主奖励，其他策略少量分成”，这更符合策略归因的实际含义。

---

### 3.2 `compressed_memory.py`

状态：`主公式基本正确，属于轻量 NumPy 工程实现`

#### 已确认正确的部分

ELBO 形式：

\[
\mathcal{L}_{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta D_{KL}(q(z|x)\|p(z))
\]

实现通常按最小化 loss 的形式写成：

\[
\mathrm{Loss} = \mathrm{ReconLoss} + \beta \cdot \mathrm{KL}
\]

当前 KL 项符号方向是正确的。

#### 需要保留的审慎意见

- 当前实现更偏“依赖少、可运行、便于集成”
- 不是高性能深度学习版本
- 如果以后要做更大规模实验，建议考虑 PyTorch 重写版并增加更多数值稳定性验证

---

### 3.3 `pareto_drift.py`

状态：`GPR 主体公式正确，工程实现已补齐`

#### 已确认正确的部分

RBF 核：

\[
k(x, x') = \sigma_f^2 \exp\left(- \frac{\|x - x'\|^2}{2l^2}\right)
\]

预测均值：

\[
\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y
\]

预测方差：

\[
\sigma_*^2 = k_{**} - k_*^T (K + \sigma_n^2 I)^{-1} k_*
\]

#### 已修复的问题

- 文件中曾存在两个 `predict_next()` 定义
- 当前已保留旧实现为 `_predict_next_raw()`，外部使用的是修正后的 `predict_next()`

#### 仍需持续审查的点

- `_compute_feature()` 属于工程特征设计，不是标准理论公式
- 它对 GP 预测上限影响很大，后续需要继续做消融分析

---

### 3.4 `geodesic_flow.py`

状态：`主方向正确，优于旧线性插值版`

#### 已确认正确的部分

Grassmann geodesic 使用的核心形式为：

\[
\Phi(t) = P_S U \cos(t\Theta) + R_S V \sin(t\Theta)
\]

#### 当前结论

- 这比旧的线性插值方式更接近理论原式
- fallback 里仍保留线性插值，但现在明确只作为数值回退路径存在

---

### 3.5 `kemm/algorithms/kemm.py`

状态：`算法流程清晰，但包含 benchmark 强化逻辑`

#### 当前已确认的结构

环境变化后，主流程现在被拆成以下步骤函数：

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

这次拆分的意义是：

- 更容易做单元测试
- 更容易做日志记录和可视化
- 更容易对每一步单独做公式审查

#### benchmark-aware 先验

`_problem_aware_candidates()` 是 benchmark 强化逻辑。

结论：

- 对 benchmark 很有效
- 但不应被描述为通用理论模块
- `ship_simulation` 中默认关闭这部分是正确的

#### 本轮发现并修复的问题

在流程拆分过程中发现一个隐藏 bug：

- memory probe 在小变量维度下会从 12 维退化到 10 维
- 这会导致 memory 检索阶段的维度不匹配
- 当前已修复为固定补齐到 12 维

---

### 3.6 `kemm/benchmark/problems.py`

状态：`问题定义大体可用，但存在一处表达需统一`

#### 已确认可用的部分

- `FDA1`
- `FDA3`
- `dMOP1`
- `dMOP2`
- `JY1`
- `JY4`

#### 需要统一的点

`FDA2` 的文字描述与当前实现的参数表达不是完全一致。

这不一定会导致程序错误，因为：

- 目标函数和 POF 生成函数当前使用的是同一套实现

但如果要在论文里写“严格使用某文献原定义”，这里必须继续统一说明和代码。

#### 已修复的问题

- `dMOP3` 曾经被误写成与 `FDA1` 数学等价，导致完整 benchmark 报表中 `FDA1` 与 `dMOP3`
  的 MIGD / SP / MS 行完全相同
- 当前已按文献里的 `f1(x_I) = x_r` 结构修复
- 由于原文只写 `r = S(1, 2, ..., n)` 而未给出可直接编码的确定性调度，本项目采用“按环境
  索引循环切换 active coordinate”的可复现实例化方式，使 `dMOP3` 不再退化为 `FDA1`

---

### 3.7 `kemm/benchmark/metrics.py`

状态：`主要指标实现基本正确`

#### 当前结论

- `IGD / MIGD / SP / MS` 的实现基本正确
- `HV` 当前只对 2 目标情况有直接支持

这不是错误，而是功能边界，需要在文档里说清楚。

---

### 3.8 `ship_simulation`

状态：`物理逻辑已闭环，但仍是 MVP`

#### 已完成的关键修正

- Nomoto 控制链已修正，直线基线可以到达终点
- 终点残差惩罚已加入目标函数，避免“未到达终点的伪优解”
- 风险模型已支持连续风险输出与时间序列分析

#### 当前边界

- 风险模型是工程连续风险，不是完整 COLREG 规则模型
- 动力学是 Nomoto 一阶，不是 MMG

---

## 4. 当前最关键的待改项

### 高优先级

1. 继续把 `adaptive_operator.py / compressed_memory.py / geodesic_flow.py / pareto_drift.py` 压成兼容层
2. 统一 `FDA2` 的文档描述与代码表达
3. 为 adaptive / drift / transfer / memory 增加更细的单元测试

### 中优先级

1. 继续拆分 `apps/benchmark_runner.py`
2. 为 ship 主线加入更明确的约束输出接口
3. 继续统一 benchmark 与 ship 的图表风格和报告结构

---

## 5. 推荐表述模板

### 对老师或论文中较稳妥的说法

- `KEMM 的核心由自适应策略分配、压缩记忆、漂移预测和几何迁移组成`
- `其中 UCB1 与 GP 部分基于标准统计学习公式实现`
- `部分候选生成与比例修正逻辑属于工程启发式`
- `当前 benchmark 强化版还包含面向标准测试问题的结构先验增强`

### 不建议直接使用的说法

- `完全严格按照文献原式实现了所有模块`
- `当前 KEMM 是对任意动态多目标问题都完全通用的纯理论版本`
