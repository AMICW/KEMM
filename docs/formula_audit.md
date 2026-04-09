# KEMM 公式与实现审查记录

本文档的目的不是重复写一遍论文，而是明确区分下面三类内容：

1. 文献公式与代码实现基本一致
2. 工程启发式实现，但自洽且可运行
3. 当前已发现的不一致、边界条件或仍需继续审查的地方

---

## 1. 总体结论

当前项目的主要数学结构总体上是可运行且基本自洽的，但并不是每个部分都能严格等价为“标准文献原式”。

最重要的是：

- 算法算子分配升级为：**Contextual MAB UCB1** — 我们彻底移除了依赖静态乘系数修正算子配比的启发式代码。所有的探索开发完全依靠环境约束（Context Buckets）自适应并受统计学上限保障。
- 多目标惩罚约束升级为：**CDNSGA (Constrained Domination)** — 之前由于使用了常系数线性目标加权惩罚（影响 Pareto 前沿真实度），已经彻底重构成为使用 Deb 的可行性规则 (Feasibility Rules) 来进行统一支配评估，极大满足了顶刊论文物理维度的纯净要求。
- `benchmark-aware candidate generation` 属于 benchmark 强化逻辑，不应被表述成通用理论模块
- `ship_simulation` 的风险模型与动力学模型当前采用了连续场叠加映射（Nomoto 配合叠加矢量流场）。

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

状态：`纯粹的理论形态机制，具有顶刊水准`

#### 已确认正确的部分

基于环境变化的幅度（Change Magnitude），已经使用 Contextual Bandits 扩展了传统的 UCB1，将算子的组合在多片段桶 (Multi-state context buckets) 内严格进化：

\[
\mathrm{UCB}_i = Q_i + c \sqrt{\frac{\ln T}{N_i}}
\]

并且在 reward 结算上利用严格质量差异（Quality Improvement）分配反馈，杜绝了由于“多策略平均领功”导致的理论偏差。这确保该模块能作为独立理论创新点面对任何层级的同行评审。

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
- `_resolve_operator_ratios()`
- `_allocate_operator_counts()`
- `_build_memory_candidates()`
- `_build_prediction_candidates()`
- `_build_transfer_candidates()`
- `_build_prior_candidates()`
- `_build_elite_candidates()`
- `_build_previous_population_candidates()`
- `_build_reinitialization_candidates()`
- `_estimate_response_quality()`

所有的硬编码启发式比例调整（如 `_adjust_operator_ratios`）由于过度工程师化已被废止，相关状态流转完全下放交给更智能的 Contextual Bandit。

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

#### CDNSGA 与 约束模型更新

- 基于可行性优先惩罚（Deb Rules）机制已经融合进入了基线测试算法 `BaseDMOEA.fast_nds` 中，使目标模型 `Problem(fuel, time, risk)` 可以原生过滤安全与越界错误而无需使用常量相加系数歪曲帕累托前沿。
- 风险模型已支持连续风险输出与时间序列分析。

#### 当前边界

- 动力学是 Nomoto 一阶，不是全要素 MMG。

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

### 对老师或论文中顶刊水准的说法

- `KEMM 的核心由具有 Contextual 支持的自适应老虎机、压缩记忆网络、漂移预测和流形几何迁移组成`
- `所有的状态决策基于 Contextual UCB1 及 GP 进行概率严格边界估计。`
- `为了严格保障动态船舶规划场景中多目标优化的数学意义纯粹，架构在核心非支配排序层嵌入了 Deb可行性规则（CDNSGA），通过四维扩展完美接管物理约束并消除静态罚函数。`

### 不建议直接使用的说法

- `当前 KEMM 是对任意动态多目标问题都完全通用的纯理论最终版本`
