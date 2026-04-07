"""Typed configuration and diagnostics objects shared by KEMM workflows.

这一层不负责实现具体算法，而是负责把下面这些内容显式类型化：

- KEMM 主流程经常修改的超参数
- 一次环境变化后的结构化诊断结果
- benchmark 主线共用的实验配置

这样做的直接好处是：后续如果你要调整算法结构，优先修改配置对象和诊断对象即可，
而不是继续把魔法数字和隐式字典散落在主流程代码中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class KEMMConfig:
    """KEMM 主算法运行配置。

    这份配置把当前 KEMM 主流程里最常改的参数集中收口，主要分成几类：

    - 算法规模参数
    - 自适应策略分配参数
    - 记忆模块参数
    - 漂移预测参数
    - 几何迁移参数
    - 候选池构造启发式参数

    以后如果你要：

    - 调整候选池构成比例
    - 关闭或弱化 benchmark prior
    - 修改记忆/预测/迁移的预算
    - 做消融实验

    都应优先从这个对象入手，而不是直接在 `kemm/algorithms/kemm.py` 里改硬编码常量。
    """

    pop_size: int = 100
    n_var: int = 10
    n_obj: int = 2
    benchmark_aware_prior: bool = True
    enable_memory: bool = True
    enable_prediction: bool = True
    enable_transfer: bool = True
    enable_adaptive: bool = True

    # Adaptive operator selection.
    exploration_c: float = 0.5
    reward_window: int = 10
    temperature: float = 0.8
    min_operator_ratio: float = 0.05

    # Memory module.
    memory_capacity: int = 50
    memory_hidden_dim: int = 64
    memory_beta: float = 0.1
    memory_online_epochs: int = 15
    memory_top_k: int = 3
    memory_min_keep: int = 4
    latent_dim_cap: int = 8

    # Drift module.
    drift_window: int = 6
    drift_feature_dim: int = 10
    drift_history: int = 8
    gp_lengthscale: float = 1.0
    gp_noise_var: float = 0.05
    prediction_confidence_threshold: float = 0.3
    prediction_pool_multiplier: int = 2
    prediction_min_pool: int = 8
    probe_dim: int = 12

    # Transfer module.
    transfer_n_clusters: int = 4
    transfer_n_subspaces: int = 5
    transfer_var_threshold: float = 0.95
    transfer_jitter: float = 1e-6
    transfer_target_sample_size: int = 30
    transfer_jitter_ratio: float = 0.02

    # Candidate-pool heuristics.
    memory_transferability_bonus: float = 0.12
    transfer_transferability_bonus: float = 0.10
    prediction_stability_bonus: float = 0.04
    prediction_change_penalty: float = 0.10
    transfer_change_penalty: float = 0.06
    reinit_change_bonus: float = 0.12
    elite_keep_min: int = 6
    elite_keep_fraction: float = 0.10
    elite_jitter_ratio: float = 0.03
    previous_keep_min: int = 10
    previous_keep_fraction: float = 0.20
    elite_spread_floor_ratio: float = 0.02
    prior_sample_fraction: float = 0.20
    prior_min_samples: int = 8


@dataclass
class KEMMChangeDiagnostics:
    """一次环境变化响应后的结构化诊断结果。

    这个对象是 KEMM 主流程和图表层之间的稳定边界。它的设计目的有三点：

    1. 让 benchmark 图表不需要直接访问算法私有属性。
    2. 让你改动 `respond_to_change()` 内部实现时，外层报告接口保持稳定。
    3. 让后续调参、调试、写论文时可以直接读取关键中间量。

    字段说明：

    - `time_step`：第几次环境变化。
    - `change_time`：当前动态问题对应的时间变量。
    - `operator_ratios`：memory / prediction / transfer / reinit 的最终比例。
    - `requested_counts`：按比例期望分配的候选数量。
    - `actual_counts`：各来源最终实际进入候选池的数量。
    - `candidate_pool_size`：变化响应完成后候选池总规模。
    - `prediction_confidence`：漂移预测模块给出的置信度代理值。
    - `change_magnitude`：环境变化幅度代理值。
    - `transferability`：当前环境与历史环境的可迁移性估计。
    - `response_quality`：变化响应质量代理值。
    - `selected_front_size`：环境选择后保留下来的前沿规模。
    """

    time_step: int
    change_time: float
    operator_ratios: dict[str, float]
    requested_counts: dict[str, int]
    actual_counts: dict[str, int]
    candidate_pool_size: int
    prediction_confidence: float
    change_magnitude: float
    transferability: float
    response_quality: float
    selected_front_size: int


@dataclass
class ExperimentConfig:
    """通用 benchmark 实验配置对象。

    这份 dataclass 用于承接 benchmark 主线里与算法无关的实验规模参数。
    当前 `apps/benchmark_runner.py` 还有自己的类式配置，但这份对象可以作为后续进一步
    统一 runner 配置接口的基础。
    """

    pop_size: int = 100
    n_var: int = 10
    n_obj: int = 2
    nt: int = 10
    tau_t: int = 10
    n_changes: int = 10
    gens_per_change: int = 20
    n_runs: int = 5
    significance: float = 0.05
    problems_standard: Sequence[str] = field(
        default_factory=lambda: ["FDA1", "FDA2", "FDA3", "dMOP1", "dMOP2", "dMOP3"]
    )
    problems_jy: Sequence[str] = field(default_factory=lambda: ["JY1", "JY4"])


__all__ = ["ExperimentConfig", "KEMMChangeDiagnostics", "KEMMConfig"]
