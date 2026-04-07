"""Dynamic multi-objective benchmark problems.

This module contains the actual benchmark problem implementations that were
previously embedded in ``benchmark_algorithms.py``.
"""

from __future__ import annotations

import numpy as np


class DynamicTestProblems:
    """Dynamic benchmark problem set used by the KEMM experiments."""

    def __init__(self, nt: int = 10, tau_t: int = 10):
        self.nt = nt
        self.tau_t = tau_t

    def get_time(self, generation: int) -> float:
        """Return normalized dynamic time."""

        return (1.0 / self.nt) * np.floor(generation / self.tau_t)

    def _dmop3_active_index(self, t: float, n_var: int) -> int:
        """Return the active decision-variable index for dMOP3.

        Published dMOP3 uses ``f1(x_I) = x_r`` with ``r = S(1, 2, ..., n)``.
        The source notation does not fully specify a deterministic schedule for
        ``S`` in code form, so we use a reproducible cyclic selection keyed by
        the environment index. This preserves the intended benchmark behavior:
        the decision variable controlling POF spread changes over time.
        """

        if n_var <= 0:
            return 0
        env_index = int(np.floor(t * self.nt + 1e-12))
        return env_index % n_var

    @staticmethod
    def fda1(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        G = np.sin(0.5 * np.pi * t)
        f1 = x[:, 0]
        g = 1.0 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    @staticmethod
    def fda1_pof(n_points: int = 200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    @staticmethod
    def fda2(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        n = x.shape[1]
        H = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        f1 = x[:, 0]
        xII = x[:, 1 : max(2, n // 2)]
        xIII = x[:, max(2, n // 2) :]
        g = 1.0 + np.sum(xII**2, axis=1)
        exp_term = H + np.sum((xIII - H) ** 2, axis=1) if xIII.shape[1] > 0 else np.full(len(x), H)
        f2 = g * (1.0 - (f1 / g) ** exp_term)
        return np.column_stack([f1, f2])

    @staticmethod
    def fda2_pof(t: float, n_points: int = 200) -> np.ndarray:
        H = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        f1 = np.linspace(0.001, 1, n_points)
        return np.column_stack([f1, 1.0 - f1**H])

    @staticmethod
    def fda3(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        n = x.shape[1]
        F = 10 ** (2.0 * np.sin(0.5 * np.pi * t))
        G = np.abs(np.sin(0.5 * np.pi * t))
        half = max(1, n // 2)
        f1 = np.sum(np.abs(x[:, :half]) ** F, axis=1)
        g = 1.0 + G + np.sum((x[:, half:] - G) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    @staticmethod
    def fda3_pof(t: float = 0, n_points: int = 200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    @staticmethod
    def dmop1(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        n = x.shape[1]
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(np.abs(x[:, 1:]) ** H, axis=1) / max(1, n - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    @staticmethod
    def dmop1_pof(n_points: int = 200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    @staticmethod
    def dmop2(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        G = np.sin(0.5 * np.pi * t)
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = x[:, 0]
        g = 1.0 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])

    @staticmethod
    def dmop2_pof(t: float, n_points: int = 200) -> np.ndarray:
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0.001, 1, n_points)
        return np.column_stack([f1, 1.0 - f1**H])

    def dmop3(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        n_var = x.shape[1]
        G = np.sin(0.5 * np.pi * t)
        active_idx = self._dmop3_active_index(t, n_var)
        mask = np.ones(n_var, dtype=bool)
        mask[active_idx] = False
        g = 1.0 + 9.0 * np.sum((x[:, mask] - G) ** 2, axis=1)
        f1 = x[:, active_idx]
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / g, 0.0, None)))
        return np.column_stack([f1, f2])

    @staticmethod
    def dmop3_pof(n_points: int = 200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    @staticmethod
    def jy1(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        A = np.sin(0.5 * np.pi * t)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum((x[:, 1:] - A) ** 2, axis=1) / max(1, x.shape[1] - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])

    @staticmethod
    def jy1_pof(n_points: int = 200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])

    @staticmethod
    def jy4(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        A = np.sin(0.5 * np.pi * t)
        H = 1.5 + A
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum((x[:, 1:] - A) ** 2, axis=1) / max(1, x.shape[1] - 1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])

    @staticmethod
    def jy4_pof(t: float, n_points: int = 200) -> np.ndarray:
        H = 1.5 + np.sin(0.5 * np.pi * t)
        f1 = np.linspace(0.001, 1, n_points)
        return np.column_stack([f1, 1.0 - f1**H])

    def get_problem(self, name: str):
        """Return objective function and true Pareto front generator."""

        mapping = {
            "FDA1": (self.fda1, self.fda1_pof),
            "FDA2": (self.fda2, self.fda2_pof),
            "FDA3": (self.fda3, self.fda3_pof),
            "dMOP1": (self.dmop1, self.dmop1_pof),
            "dMOP2": (self.dmop2, self.dmop2_pof),
            "dMOP3": (self.dmop3, self.dmop3_pof),
            "JY1": (self.jy1, self.jy1_pof),
            "JY4": (self.jy4, self.jy4_pof),
        }
        if name not in mapping:
            raise ValueError(f"未知测试函数: {name}. 可用: {list(mapping.keys())}")
        return mapping[name]


__all__ = ["DynamicTestProblems"]
