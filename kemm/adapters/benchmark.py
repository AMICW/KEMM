"""Benchmark-only structural priors for KEMM.

这个模块的存在是为了把“通用 KEMM 核心”和“针对标准测试函数的结构先验”明确分层。
以后你要改算法主体时，可以不碰这里；要改 benchmark 强化策略时，也不需要回到
`kemm/algorithms/kemm.py` 里找硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkPriorAdapterConfig:
    """benchmark 先验候选生成器配置。"""

    noise_std: float = 0.02


class BenchmarkPriorAdapter:
    """针对标准动态测试问题生成结构先验候选。

    这是 benchmark-aware enhancement，不是通用 KEMM 理论模块。
    对 ship 或其他真实问题，默认不应启用。
    """

    def __init__(self, config: BenchmarkPriorAdapterConfig | None = None):
        self.config = config or BenchmarkPriorAdapterConfig()

    def generate(
        self,
        *,
        obj_func,
        t: float,
        n_samples: int,
        lb: np.ndarray,
        ub: np.ndarray,
        n_var: int,
    ) -> np.ndarray | None:
        """根据 benchmark 问题名生成问题结构感知候选。"""

        name = getattr(obj_func, "__name__", "").lower()
        if not name:
            return None

        X = np.random.uniform(lb, ub, (n_samples, n_var))
        tradeoff = np.linspace(lb[0], ub[0], n_samples)
        np.random.shuffle(tradeoff)
        X[:, 0] = tradeoff

        if name in ("fda1", "dmop2", "dmop3", "jy1", "jy4"):
            shift = float(np.sin(0.5 * np.pi * t))
            X[:, 1:] = shift
        elif name == "fda2":
            H = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
            split = max(2, n_var // 2)
            X[:, 1:split] = 0.0
            X[:, split:] = H
        elif name == "fda3":
            G = abs(np.sin(0.5 * np.pi * t))
            half = max(1, n_var // 2)
            X[:, :half] = 0.0
            X[:, 0] = tradeoff
            X[:, half:] = G
        elif name == "dmop1":
            X[:, 1:] = 0.0
        else:
            return None

        noise = np.random.normal(0.0, self.config.noise_std, X.shape)
        noise[:, 0] = 0.0
        return np.clip(X + noise, lb, ub)
