"""Benchmark performance metrics."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial.distance import cdist


class PerformanceMetrics:
    """Performance metrics for dynamic multi-objective optimization."""

    @staticmethod
    def igd(obtained_pof: np.ndarray, true_pof: np.ndarray) -> float:
        if len(obtained_pof) == 0:
            return float("inf")
        distances = cdist(true_pof, obtained_pof, "euclidean")
        return float(np.mean(np.min(distances, axis=1)))

    @staticmethod
    def migd(igd_values: List[float]) -> float:
        return float(np.mean(igd_values))

    @staticmethod
    def spacing(obtained_pof: np.ndarray) -> float:
        if len(obtained_pof) <= 1:
            return float("inf")
        distances = cdist(obtained_pof, obtained_pof, "euclidean")
        np.fill_diagonal(distances, np.inf)
        d_i = np.min(distances, axis=1)
        d_mean = np.mean(d_i)
        return float(np.sqrt(np.sum((d_i - d_mean) ** 2) / max(1, len(d_i) - 1)))

    @staticmethod
    def maximum_spread(obtained_pof: np.ndarray, true_pof: np.ndarray) -> float:
        if len(obtained_pof) == 0:
            return 0.0
        m = true_pof.shape[1]
        p_max = np.max(true_pof, axis=0)
        p_min = np.min(true_pof, axis=0)
        ps_max = np.max(obtained_pof, axis=0)
        ps_min = np.min(obtained_pof, axis=0)
        overlap = np.minimum(p_max, ps_max) - np.maximum(p_min, ps_min)
        ms_sum = np.sum(np.maximum(overlap, 0) ** 2)
        return float(np.sqrt(ms_sum / m))

    @staticmethod
    def hypervolume(obtained_pof: np.ndarray, ref_point: np.ndarray) -> float:
        if len(obtained_pof) == 0:
            return 0.0

        m = obtained_pof.shape[1]
        if m != 2:
            return 0.0

        valid = np.all(obtained_pof < ref_point, axis=1)
        pof = obtained_pof[valid]
        if len(pof) == 0:
            return 0.0

        pof_sorted = pof[np.argsort(pof[:, 1])]
        hv = 0.0
        prev_f1 = ref_point[0]
        for i in range(len(pof_sorted) - 1, -1, -1):
            hv += (prev_f1 - pof_sorted[i, 0]) * (
                ref_point[1] - pof_sorted[i, 1]
                if i == len(pof_sorted) - 1
                else pof_sorted[i + 1][1] - pof_sorted[i][1]
            )
            prev_f1 = pof_sorted[i, 0]
        return max(0.0, float(hv))


__all__ = ["PerformanceMetrics"]
