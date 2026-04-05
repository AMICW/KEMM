"""ship 主线的代表解选择策略。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ship_simulation.optimizer.problem import EvaluationResult


def weighted_objective_scores(objectives: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """将多目标值压成一个可比较的归一化折中分数。"""

    array = np.atleast_2d(np.asarray(objectives, dtype=float))
    if array.size == 0:
        return np.zeros(0, dtype=float)
    span = np.ptp(array, axis=0)
    normalized = (array - array.min(axis=0)) / (span + 1e-9)
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or weight_array.size != array.shape[1]:
        raise ValueError("weights must match the objective dimension.")
    weight_array = weight_array / max(float(np.sum(weight_array)), 1e-9)
    return normalized @ weight_array


def select_representative_index(
    objectives: np.ndarray,
    evaluations: Sequence[EvaluationResult],
    weights: Sequence[float],
    *,
    safety_clearance: float,
) -> int:
    """优先选安全且推进更充分的代表解。"""

    if not evaluations:
        raise ValueError("At least one evaluation is required.")

    scores = weighted_objective_scores(objectives, weights)

    def ranking_key(index: int) -> tuple[float | bool, ...]:
        evaluation = evaluations[index]
        clearance = float(evaluation.analysis_metrics.get("minimum_clearance", np.inf))
        ship_distance = float(evaluation.analysis_metrics.get("minimum_ship_distance", np.inf))
        if np.isfinite(clearance):
            hard_intrusion = max(0.0, -clearance)
            clearance_shortfall = max(0.0, safety_clearance - clearance)
        else:
            hard_intrusion = 0.0
            clearance_shortfall = 0.0
        return (
            hard_intrusion > 0.0,
            clearance_shortfall > 0.0,
            clearance_shortfall,
            not bool(evaluation.reached_goal),
            float(evaluation.terminal_distance),
            -ship_distance if np.isfinite(ship_distance) else float("-inf"),
            float(scores[index]),
            float(evaluation.risk.max_risk),
        )

    return int(min(range(len(evaluations)), key=ranking_key))


__all__ = [
    "select_representative_index",
    "weighted_objective_scores",
]
