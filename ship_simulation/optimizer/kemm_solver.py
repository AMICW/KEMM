"""将通用 KEMM 适配到 ship 主线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import time

import numpy as np

from kemm.algorithms.kemm import KEMM_DMOEA_Improved
from ship_simulation.config import DemoConfig
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.optimizer.selection import select_representative_index


@dataclass
class KEMMOptimizationResult:
    """KEMM 求解结果。"""

    best_decision: np.ndarray
    best_evaluation: EvaluationResult
    pareto_decisions: np.ndarray
    pareto_objectives: np.ndarray
    population: np.ndarray
    fitness: np.ndarray
    history: List[Dict[str, float]]
    runtime_s: float


class ShipKEMMOptimizer:
    """ship 主线下的 KEMM 运行器。"""

    def __init__(self, interface: ShipOptimizerInterface, demo_config: DemoConfig):
        self.interface = interface
        self.demo_config = demo_config
        self.context = interface.build_context()

    def optimize(self) -> KEMMOptimizationResult:
        kemm_cfg = self.demo_config.kemm
        bounds = (
            self.context.var_bounds[:, 0].astype(float),
            self.context.var_bounds[:, 1].astype(float),
        )
        objective = self.interface.make_objective_function(self.context)

        old_state = np.random.get_state()
        np.random.seed(kemm_cfg.seed)
        t0 = time.perf_counter()
        try:
            algo = KEMM_DMOEA_Improved(
                pop_size=kemm_cfg.pop_size,
                n_var=self.context.n_var,
                n_obj=self.context.n_obj,
                var_bounds=bounds,
                benchmark_aware_prior=False,
            )
            algo.initialize()
            self._inject_initial_guesses(algo.population)
            algo.fitness = algo.evaluate(algo.population, objective, 0.0)

            history: List[Dict[str, float]] = [self._summarize_generation(algo, generation=0)]
            for generation in range(1, kemm_cfg.generations + 1):
                if kemm_cfg.use_change_response and kemm_cfg.refresh_interval > 0 and generation % kemm_cfg.refresh_interval == 0:
                    algo.respond_to_change(objective, float(generation))
                else:
                    algo.evolve_one_gen(objective, float(generation))
                history.append(self._summarize_generation(algo, generation=generation))

            fronts = algo.fast_nds(algo.fitness)
            pareto_idx = np.asarray(fronts[0] if fronts else np.arange(len(algo.population)), dtype=int)
            pareto_decisions = algo.population[pareto_idx].copy()
            pareto_objectives = algo.fitness[pareto_idx].copy()
            best_decision, best_evaluation = self._select_representative_solution(pareto_decisions, pareto_objectives)
            return KEMMOptimizationResult(
                best_decision=best_decision,
                best_evaluation=best_evaluation,
                pareto_decisions=pareto_decisions,
                pareto_objectives=pareto_objectives,
                population=algo.population.copy(),
                fitness=algo.fitness.copy(),
                history=history,
                runtime_s=time.perf_counter() - t0,
            )
        finally:
            np.random.set_state(old_state)

    def _inject_initial_guesses(self, population: np.ndarray) -> None:
        kemm_cfg = self.demo_config.kemm
        if not kemm_cfg.inject_initial_guess or len(population) == 0:
            return
        base = np.clip(
            self.context.initial_guess.astype(float),
            self.context.var_bounds[:, 0],
            self.context.var_bounds[:, 1],
        )
        copies = min(max(1, kemm_cfg.initial_guess_copies), len(population))
        population[0] = base
        if copies == 1:
            return
        scale = kemm_cfg.initial_guess_jitter_ratio * (self.context.var_bounds[:, 1] - self.context.var_bounds[:, 0])
        noise = np.random.normal(0.0, scale, size=(copies - 1, self.context.n_var))
        seeded = np.clip(base + noise, self.context.var_bounds[:, 0], self.context.var_bounds[:, 1])
        population[1:copies] = seeded

    def _summarize_generation(self, algo: KEMM_DMOEA_Improved, generation: int) -> Dict[str, float]:
        fronts = algo.fast_nds(algo.fitness)
        pareto_idx = fronts[0] if fronts else list(range(len(algo.population)))
        pf = algo.fitness[pareto_idx]
        mins = np.min(algo.fitness, axis=0)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-9)
        summary = {
            "generation": float(generation),
            "pareto_size": float(len(pareto_idx)),
            "best_fuel": float(mins[0]),
            "best_time": float(mins[1]),
            "best_risk": float(mins[2]),
            "best_weighted_score": float(np.min(algo.fitness @ weights)),
            "pareto_mean_fuel": float(np.mean(pf[:, 0])),
            "pareto_mean_time": float(np.mean(pf[:, 1])),
            "pareto_mean_risk": float(np.mean(pf[:, 2])),
        }
        diagnostics = getattr(algo, "last_change_diagnostics", None)
        if diagnostics is not None:
            summary["prediction_confidence"] = float(diagnostics.prediction_confidence)
            summary["response_quality"] = float(diagnostics.response_quality)
        return summary

    def _select_representative_solution(
        self,
        decisions: np.ndarray,
        objectives: np.ndarray,
    ) -> tuple[np.ndarray, EvaluationResult]:
        evaluations = [self.interface.simulate(ind) for ind in decisions]
        real_index = select_representative_index(
            objectives,
            evaluations,
            self.interface.config.objective_weights,
            safety_clearance=self.interface.config.safety_clearance,
        )
        return decisions[real_index].copy(), evaluations[real_index]

    def _pick_by_weighted_score(self, objectives: np.ndarray) -> int:
        if len(objectives) == 1:
            return 0
        spread = np.ptp(objectives, axis=0)
        normalized = (objectives - objectives.min(axis=0)) / (spread + 1e-9)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / np.sum(weights)
        scores = normalized @ weights
        return int(np.argmin(scores))
