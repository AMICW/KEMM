# KEMM 主线详细说明

本文档专门解释 `kemm/` 与 benchmark 主线。目标是把 KEMM 主线补到和
`docs/ship_simulation_reference.md` 同样细的程度，让读者不必先翻完全部源码，也能
理解下面几件事：

1. KEMM 主线到底解决什么问题
2. `kemm/algorithms/`、`kemm/core/`、`kemm/benchmark/`、`kemm/reporting/` 各自负责什么
3. 一次 benchmark 实验从入口到图表是怎么跑完的
4. KEMM 的每个子模块在代码里如何协同
5. 哪些部分是标准公式，哪些部分是工程启发式或 benchmark-aware 增强

---

## 1. 主线定位

KEMM 主线负责解决动态多目标优化问题。相比静态多目标问题，动态多目标问题的关键困难
在于：

- 目标函数会随时间变化
- 当前环境的最优解分布不一定能直接复用到下一环境
- 只靠随机重启通常代价太高

KEMM 的基本思想是：环境变化后，不是把种群全部丢掉重来，而是同时利用四类信息源：

1. 历史记忆 `memory`
2. 漂移预测 `drift prediction`
3. 几何迁移 `transfer`
4. 随机重启 `reinitialization`

然后通过自适应策略分配模块，根据环境变化强度与历史表现，动态决定这几类信息源各应
该贡献多少候选样本。

从工程角度看，KEMM 主线承担三个任务：

- 给 benchmark 主线提供主要研究对象
- 给 ship 主线提供可复用的优化核心
- 给后续消融实验和模块替换提供稳定骨架

---

## 2. KEMM 主线目录结构

```text
kemm/
├── adapters/
│   ├── __init__.py
│   └── benchmark.py
├── algorithms/
│   ├── base.py
│   ├── baselines.py
│   └── kemm.py
├── benchmark/
│   ├── problems.py
│   └── metrics.py
├── core/
│   ├── adaptive.py
│   ├── drift.py
│   ├── memory.py
│   ├── transfer.py
│   └── types.py
└── reporting/
    ├── __init__.py
    └── benchmark_report.py
```

此外，benchmark 主线的真实应用入口在：

- `apps/benchmark_runner.py`

兼容入口在：

- `run_experiments.py`
- `benchmark_algorithms.py`
- `visualization.py`

---

## 3. 各目录职责

### 3.1 `kemm/algorithms/`

这一层负责可直接运行的算法对象。

- `base.py`
  定义所有动态多目标算法共享的基础能力，例如初始化、评价、非支配排序、拥挤距离、
  环境选择和单代演化骨架。
- `baselines.py`
  保存用于 benchmark 对比的经典算法实现。
- `kemm.py`
  保存 KEMM 主体，是整个项目里最关键的算法编排文件。

### 3.2 `kemm/core/`

这一层负责 KEMM 的子模块，每个文件理论上只做一个方面的工作。

- `adaptive.py`
  策略选择与 reward 更新。
- `drift.py`
  前沿漂移建模与预测。
- `memory.py`
  历史环境压缩记忆与相似环境检索。
- `transfer.py`
  几何迁移与多源迁移。
- `types.py`
  用 dataclass 和类型别名承接配置、结果和结构化状态。

需要明确的一点是：当前 `adaptive.py / drift.py / memory.py / transfer.py` 仍然是
新架构下的真实实现区；根目录旧文件现在只保留兼容导出：

- `adaptive_operator.py`
- `pareto_drift.py`
- `compressed_memory.py`
- `geodesic_flow.py`

因此，后续如果你要真正修改 KEMM 的核心子模块，应优先直接改：

- `kemm/core/adaptive.py`
- `kemm/core/drift.py`
- `kemm/core/memory.py`
- `kemm/core/transfer.py`

### 3.3 `kemm/adapters/`

这一层负责把“问题专用增强逻辑”从通用 KEMM 核心里拆出去。

- `benchmark.py`
  保存 benchmark-only 的结构先验候选生成器。

这一层的意义很直接：

- 以后你改 KEMM 主体时，不需要碰 benchmark 测试函数专用逻辑
- 以后你改 benchmark prior 时，也不需要回到主流程类里找硬编码
- ship 主线可以天然不依赖这一层

### 3.4 `kemm/benchmark/`

这一层负责 benchmark 问题与指标，不直接承担算法流程。

- `problems.py`
  定义动态测试问题、真实前沿生成和时间变量控制逻辑。
- `metrics.py`
  定义 `IGD / MIGD / SP / MS / HV` 等指标。

### 3.5 `kemm/reporting/`

这一层负责把 benchmark 结果转换成报告产物，而不是直接在控制台里丢表格。

- `benchmark_report.py`
  输出 `metrics.csv / ranks.csv / summary.json / summary.md`

---

## 4. 入口层如何驱动 KEMM

### 4.1 命令行入口

最常见的 benchmark 入口是：

```bash
python run_experiments.py --quick
python -m apps.benchmark_runner --quick
```

调用链大致是：

1. `run_experiments.py`
2. `apps.benchmark_runner.main()`
3. `run_benchmark()`
4. `ExperimentRunner.run_all()`
5. `_run_single()` 中创建算法实例
6. 算法对象在每次环境变化时调用 `respond_to_change()`
7. 实验结束后交给 `ResultPresenter` 与 `export_benchmark_report()`

### 4.2 输出目录

benchmark 主线的标准输出结构是：

```text
benchmark_outputs/
└── benchmark_YYYYMMDD_HHMMSS/
    ├── figures/
    ├── raw/
    └── reports/
```

这和 ship 主线保持一致，便于统一归档与阅读。

---

## 5. `apps/benchmark_runner.py` 的职责

这个文件不是算法实现本身，但它决定了 benchmark 主线是如何被组织成完整实验的。

### 5.1 `ExperimentConfig`

负责集中描述：

- 使用哪些问题
- 使用哪些算法
- 每个问题重复多少次
- 每次环境变化跑多少代
- 是否启用 `JY` 问题
- 输出目录放在哪里

它的作用类似于实验计划表。

### 5.2 `ExperimentRunner`

负责真正执行实验。核心循环是三层：

1. 算法
2. 问题
3. 随机重复

在每个 `(算法, 问题, 运行编号)` 内部，又会继续遍历每次环境变化。

### 5.3 `ResultPresenter`

负责三类事情：

1. 打印终端表格
2. 画图
3. 导出结构化报告

因此 `apps/benchmark_runner.py` 在工程角色上更像实验编排器。

这里有一个现在很重要的边界：

- `ResultPresenter` 不再直接扒 KEMM 私有属性画图
- 它会先整理出 `BenchmarkFigurePayload`
- 图表层再消费 `KEMMChangeDiagnostics` 这类结构化对象

这样后续你重构 `respond_to_change()` 时，只要结构化输出不变，可视化层通常不需要改。

---

## 6. `kemm/algorithms/base.py` 负责什么

`BaseDMOEA` 是所有动态多目标算法共享的基础层。它不关心某个算法是否有 memory 或
transfer，只关心一个动态多目标进化算法最基本要会什么。

### 6.1 它通常承担的能力

- 初始化种群
- 统一调用目标函数
- 非支配排序
- 拥挤距离计算
- 环境选择
- 单代演化
- 得到当前 Pareto front

### 6.2 为什么这一层必须存在

如果没有 `BaseDMOEA`：

- 每个算法都要重复实现一遍非支配排序和环境选择
- benchmark 对比不公平，因为不同算法可能隐式使用了不同基础操作
- ship 主线适配时会更难复用

所以它的价值是统一演化框架底盘。

---

## 7. `kemm/algorithms/baselines.py` 负责什么

这个文件保存 benchmark 对比算法。当前包含：

- `RI_DMOEA`
- `PPS_DMOEA`
- `KF_DMOEA`
- `SVR_DMOEA`
- `Tr_DMOEA`
- `MMTL_DMOEA`

这些算法的作用不是陪跑，而是定义 KEMM 的对照基线。也就是说，benchmark 主线能不能
支撑论文，关键取决于：

1. KEMM 是否强于这些基线
2. KEMM 的收益是否稳定
3. 这些基线是否覆盖了合理的策略类型

从结构上看，这个文件的意义是把对手算法集中管理，而不是分散在实验脚本里。

---

## 8. `kemm/benchmark/problems.py` 负责什么

这个文件定义动态多目标 benchmark 问题。它回答的是：

- 在时间 `t` 下，某个问题的目标函数是什么
- 在时间 `t` 下，理论真实前沿 `PF_t^*` 是什么

### 8.1 当前问题族

当前项目中主要包括：

- `FDA1`
- `FDA2`
- `FDA3`
- `dMOP1`
- `dMOP2`
- `dMOP3`
- `JY1`
- `JY4`

### 8.2 为什么必须同时提供目标函数和真实前沿

因为 benchmark 指标例如 `IGD` 和 `MIGD` 需要知道真实前沿。

如果没有真实前沿，就只能比较算法之间谁更好，却很难说谁更接近真实最优分布。

### 8.3 代码层面的职责分工

通常一个问题对象要负责：

- 输入维度和变量边界
- 给定 `X, t` 返回 `F(X, t)`
- 给定 `t` 生成参考前沿样本

因此 `problems.py` 本质上是实验标准答案生成器。

---

## 9. `kemm/benchmark/metrics.py` 负责什么

这个文件定义 benchmark 评估指标，用来把“看起来不错”变成可量化比较。

### 9.1 `IGD / MIGD`

`IGD` 衡量算法得到的前沿相对真实前沿的接近程度。

\[
\mathrm{IGD}(P, PF^*) = \frac{1}{|PF^*|} \sum_{y \in PF^*} d(y, P)
\]

`MIGD` 是时间上的平均：

\[
\mathrm{MIGD} = \frac{1}{T} \sum_{t=1}^{T} \mathrm{IGD}(P_t, PF_t^*)
\]

### 9.2 `Spacing`

`SP` 衡量解集均匀性，越小越好。

\[
\mathrm{SP} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar d)^2}
\]

### 9.3 `Maximum Spread`

`MS` 衡量覆盖范围，越大越好。

\[
\mathrm{MS} = \sqrt{\frac{1}{m}\sum_{j=1}^{m}
\left(
\frac{\min(f_j^{max}, F_j^{max}) - \max(f_j^{min}, F_j^{min})}{F_j^{max} - F_j^{min}}
\right)^2}
\]

### 9.4 为什么指标层要独立出来

因为指标不是某个算法私有的，而是整个 benchmark 体系共享的评价标准。

---

## 10. `kemm/core/adaptive.py` 负责什么

这一层负责变化后应该把多少资源分给 `memory`、`prediction`、`transfer`、`reinit`。

### 10.1 当前核心对象

- `UCB1Bandit`
- `AdaptiveOperatorSelector`
- `ParetoFrontDriftDetector`

### 10.2 数学核心

UCB1 主体评分：

\[
\mathrm{UCB}_i = Q_i + c \sqrt{\frac{\ln T}{N_i}}
\]

其中：

- `Q_i`：第 `i` 个策略的平均 reward
- `T`：累计选择总次数
- `N_i`：该策略被选中的次数
- `c`：探索系数

### 10.3 与标准 UCB1 的差异

标准 UCB1 是选一个 arm，而当前工程实现会进一步做 softmax：

\[
p_i = \frac{\exp(\mathrm{UCB}_i / \tau)}{\sum_j \exp(\mathrm{UCB}_j / \tau)}
\]

它更准确地说应被称为：

- `UCB-guided allocation`

而不是标准单臂 UCB1 决策。

### 10.4 在 KEMM 主流程中的作用

它的输入主要来自两方面：

- 历史 reward
- 当前变化强度的辅助信息

它的输出是：

- 四类策略的样本比例

---

## 11. `kemm/core/memory.py` 负责什么

这一层负责记住过去见过的环境，并在未来相似环境出现时复用历史精英。

### 11.1 当前核心对象

- `LightweightVAE`
- `VAECompressedMemory`

### 11.2 为什么要做压缩记忆

如果直接把所有历史环境和所有历史解都原样存起来：

- 空间成本很高
- 检索也会越来越慢
- 不容易建立当前环境与过去哪一类环境更接近的映射

所以当前方案是：

1. 为每个环境构造一个固定维度的环境指纹
2. 用轻量 VAE 压缩环境与精英解信息
3. 在变化后按相似度检索若干历史环境
4. 把历史精英拿出来作为候选池的一部分

### 11.3 数学核心

VAE 的 ELBO：

\[
\mathcal{L}_{ELBO}
=
\mathbb{E}_{q(z|x)}[\log p(x|z)]
- \beta D_{KL}(q(z|x)\|p(z))
\]

在实现里通常按最小化损失写成：

\[
\mathrm{Loss} = \mathrm{ReconLoss} + \beta \cdot \mathrm{KL}
\]

### 11.4 在主流程中的作用

`respond_to_change()` 中的 `_build_memory_candidates()` 会：

1. 用当前环境探针向量做检索
2. 取回最相似的若干历史环境
3. 按相似度分配各环境可贡献的候选数量
4. 用当前目标函数快速筛一遍，避免过时解直接进入下一代

---

## 12. `kemm/core/drift.py` 负责什么

这一层负责环境变化后，前沿大概会往哪里漂移。

### 12.1 当前核心对象

- `LightweightGPR`
- `ParetoFrontDriftPredictor`

### 12.2 数学核心

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

### 12.3 代码层面的任务

这个模块要做三件事：

1. 从每一时刻的 Pareto front 提取可比较的特征
2. 用时间序列方式累积训练样本
3. 在下一次变化时预测新前沿的形状或中心趋势

### 12.4 在主流程中的作用

`_build_prediction_candidates()` 当前采用混合策略：

- 线性质心外推
- GP 生成候选
- 以 elite 中心为核心的扰动补样

这样做的原因是：

- 线性外推稳定、便宜
- GP 更有表达力
- elite 中心补样可以避免预测候选过于发散

---

## 13. `kemm/core/transfer.py` 负责什么

这一层负责当当前环境和历史环境结构相似时，能否把旧环境里的结构知识迁到新环境。

### 13.1 当前核心对象

- `GrassmannGeodesicFlow`
- `ManifoldTransferLearning`
- `MultiSourceTransfer`

### 13.2 数学核心

Grassmann geodesic 的核心形式：

\[
\Phi(t) = P_S U \cos(t\Theta) + R_S V \sin(t\Theta), \quad t \in [0,1]
\]

这里的直觉是：不直接在原空间粗暴线性插值，而是在子空间流形上更平滑地连接源域和目标域。

### 13.3 在主流程中的作用

`_build_transfer_candidates()` 的逻辑是：

1. 从记忆里取回若干相似源域
2. 从当前种群中抽目标域参考样本
3. 用多源迁移生成候选
4. 在迁移结果周围再加一层小扰动，补充多样性

这一步的作用主要是结构复用，与 memory 直接回放不同。

---

## 14. `kemm/core/types.py` 负责什么

这一层不直接贡献优化性能，但它对工程稳定性很重要。

当前它的价值主要体现在：

- 用 dataclass 收口配置
- 用结构化对象表达结果
- 让 IDE 和测试更容易发现接口不一致

从长期维护角度看，它是让代码不继续散掉的基础设施。

当前已经显式提供：

- `KEMMConfig`
  统一承接 reward window、探索系数、memory/drift/transfer 超参数、候选池启发式系数等
- `KEMMChangeDiagnostics`
  统一承接一次变化响应后的 `operator_ratios / requested_counts / actual_counts /
  candidate_pool_size / prediction_confidence / response_quality` 等信息

---

## 15. `kemm/algorithms/kemm.py` 负责什么

这是 KEMM 主线里最核心的编排器。它本身不是某个单独数学模块，而是把 adaptive、
memory、drift、transfer 和演化底盘组织成一个完整动态响应流程。

### 15.1 类的角色

`KEMM_DMOEA_Improved` 的职责是：

1. 在初始化时组装所有核心子模块
2. 在环境变化时生成新候选池
3. 在平稳阶段执行常规进化
4. 维护 KEMM 的状态变量和历史信息

### 15.2 初始化阶段

构造函数里主要组装：

- `_transfer_module`
- `_multi_src_transfer`
- `_vae_memory`
- `_operator_selector`
- `_drift_detector`
- `_drift_predictor`
- `_centroid_history`
- `_benchmark_adapter`

这意味着 `kemm.py` 本质上是依赖注入和流程编排文件。

### 15.3 变化响应总流程

`respond_to_change()` 当前可以分解为下面九类逻辑：

1. 归档上一环境信息
2. 计算基础策略比例
3. 结合当前变化强度修正比例
4. 把比例映射为各类样本数量
5. 分别生成各类候选
6. 合并为候选池
7. 统一评价候选池
8. 通过环境选择保留 `pop_size` 个样本
9. 估计响应质量并更新 reward

### 15.4 候选池来源

当前候选池可能包含以下来源：

- memory candidates
- prediction candidates
- transfer candidates
- benchmark-aware prior candidates
- elite keep
- previous population reuse
- random reinitialization

这些来源的设计意图不同：

- `memory`：复用过去的好结构
- `prediction`：根据时间趋势前推
- `transfer`：把相似域结构迁过来
- `prior`：benchmark 专用结构先验
- `elite keep`：保持连续性
- `previous reuse`：保留分布惯性
- `reinit`：防止陷入局部并补齐规模

### 15.5 benchmark-aware prior 的地位

现在 benchmark prior 已经通过 `kemm/adapters/benchmark.py` 注入，主流程里只保留
`_build_prior_candidates()` 和兼容旧接口的 `_problem_aware_candidates()` 包装。

它的特点是：

- 对标准测试问题很有效
- 对真实问题不一定成立
- 不应被写成通用理论模块

因此当前设计里：

- benchmark 主线可以显式开启
- ship 主线默认关闭

这是正确的架构边界。

---

## 16. 一次 `respond_to_change()` 的细分说明

为了更方便读代码，下面把关键私有函数逐个解释。

### 16.1 `_archive_current_environment()`

职责：

- 提取当前非支配精英
- 计算环境指纹
- 写入 VAE memory
- 更新漂移检测器
- 更新 GP 预测器

它的意义是给下一次变化响应准备历史上下文。

### 16.2 `_current_feature_probe()`

职责：

- 为 memory 检索构造固定维度探针向量

这不是论文标准公式，更像工程环境指纹。当前还包含过一次维度补齐 bug 修复。

### 16.3 `_adjust_operator_ratios()`

职责：

- 在 UCB 基础比例之上，再结合当前变化强度和可迁移性做二次修正

这部分应明确标注为 engineering heuristic。

### 16.4 `_allocate_operator_counts()`

职责：

- 把连续比例转成整数样本数
- 保证总样本数与 `pop_size` 同量级

### 16.5 `_build_memory_candidates()`

职责：

- 从记忆里取回相似环境的精英
- 用当前目标函数再筛一遍

### 16.6 `_build_prediction_candidates()`

职责：

- 混合线性外推、GP 候选和 elite 中心扰动

### 16.7 `_build_transfer_candidates()`

职责：

- 基于相似源域做多源迁移
- 再加小扰动补多样性

### 16.8 `_build_prior_candidates()`

职责：

- 只在 benchmark 模式下调用 adapter 生成结构先验候选

### 16.9 `_build_elite_candidates()`

职责：

- 保留上一环境精英
- 在精英附近继续补样

### 16.10 `_build_previous_population_candidates()`

职责：

- 保留上一代部分种群分布

### 16.11 `_build_reinitialization_candidates()`

职责：

- 在候选不足或变化很大时，用随机样本兜底

### 16.12 `_estimate_response_quality()`

职责：

- 用轻量 proxy 度量当前响应质量
- 反馈给 adaptive selector

这里不是严格 IGD，而是工程代理量。

### 16.13 `get_last_change_diagnostics()`

职责：

- 给 benchmark 可视化、日志和调试层提供结构化诊断接口
- 避免外部直接依赖私有属性

---

## 17. KEMM 主线和 ship 主线如何衔接

ship 主线并没有重新发明一套算法，而是通过 `ship_simulation/optimizer/kemm_solver.py`
复用同一个 `KEMM_DMOEA_Improved`。

适配时主要做了几件事：

1. 用 ship 问题的 `evaluate_population` 代替 benchmark 目标函数
2. 用 ship 问题的变量边界替代 benchmark 的边界
3. 注入直线初始解，帮助种群更快进入可行区域
4. 关闭 benchmark-aware prior

因此可以把当前项目理解成：

- `kemm/` 负责通用算法核心
- `ship_simulation/` 负责真实应用问题壳层

---

## 18. 当前工程边界

### 18.1 已经完成的

- KEMM 已被拆出为独立算法骨架
- benchmark 与 ship 共用一套核心
- 报告输出目录统一
- 关键步骤已可单独测试

### 18.2 仍需继续收口的

- `apps/benchmark_runner.py` 还可以继续拆 presenter 与 plotting
- KEMM 主线仍有 benchmark-aware 增强逻辑，需要在论文与文档里谨慎表述

---

## 19. 推荐阅读顺序

如果要从 KEMM 主线切入，建议按下面顺序阅读：

1. `README.md`
2. `docs/kemm_reference.md`
3. `docs/formula_audit.md`
4. `apps/benchmark_runner.py`
5. `kemm/algorithms/base.py`
6. `kemm/algorithms/kemm.py`
7. `kemm/core/adaptive.py`
8. `kemm/core/memory.py`
9. `kemm/core/drift.py`
10. `kemm/core/transfer.py`
11. `kemm/benchmark/problems.py`
12. `kemm/benchmark/metrics.py`

---

## 20. 如果给老师介绍 KEMM 主线，可以怎么说

可以用一句话概括：

`KEMM 是一个面向动态多目标问题的组合式响应框架，它在环境变化后同时利用历史记忆、漂移预测、几何迁移和随机探索，通过自适应策略分配生成新的候选池，再用统一环境选择形成下一环境种群。`

如果需要再压缩一点，可以强调三个关键词：

- 组合式
- 响应型
- 可迁移
