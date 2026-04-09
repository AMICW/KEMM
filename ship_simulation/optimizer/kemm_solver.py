"""将通用 KEMM 适配到 ship 主线。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List
import time

import numpy as np

from kemm.algorithms.kemm import KEMM_DMOEA_Improved
from kemm.core.types import KEMMConfig as RuntimeKEMMConfig
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
        self._algo: KEMM_DMOEA_Improved | None = None
        self._internal_random_state = None
        self._solve_count = 0

    def reset(self) -> None:
        self._algo = None
        self._internal_random_state = None
        self._solve_count = 0

    def optimize(
        self,
        interface: ShipOptimizerInterface | None = None,
        *,
        change_time: float = 0.0,
        reset: bool = False,
    ) -> KEMMOptimizationResult:
        if interface is not None:
            self.interface = interface
            self.context = interface.build_context()
        if reset:
            self.reset()
        kemm_cfg = self.demo_config.kemm
        bounds = (
            self.context.var_bounds[:, 0].astype(float),
            self.context.var_bounds[:, 1].astype(float),
        )
        objective = self.interface.make_objective_function(self.context)

        caller_state = np.random.get_state()
        if self._internal_random_state is None:
            np.random.seed(kemm_cfg.seed)
        else:
            np.random.set_state(self._internal_random_state)
        t0 = time.perf_counter()
        try:
            algo = self._ensure_algorithm(bounds, objective)

            if self._solve_count > 0:
                algo.population = np.clip(algo.population, self.context.var_bounds[:, 0], self.context.var_bounds[:, 1])
                if kemm_cfg.use_change_response:
                    algo.respond_to_change(objective, float(change_time))
                else:
                    algo.fitness = algo.evaluate(algo.population, objective, float(change_time))
            self._blend_initial_guess_candidates(algo, objective, float(change_time))

            history: List[Dict[str, float]] = [self._summarize_generation(algo, generation=0)]
            for generation in range(1, kemm_cfg.generations + 1):
                algo.evolve_one_gen(objective, float(change_time))
                history.append(self._summarize_generation(algo, generation=generation))

            fronts = algo.fast_nds(algo.fitness)
            pareto_idx = np.asarray(fronts[0] if fronts else np.arange(len(algo.population)), dtype=int)
            pareto_decisions = algo.population[pareto_idx].copy()
            pareto_objectives = algo.fitness[pareto_idx, :3].copy()
            best_decision, best_evaluation = self._select_representative_solution(pareto_decisions, pareto_objectives)
            self._solve_count += 1
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
            self._internal_random_state = np.random.get_state()
            np.random.set_state(caller_state)

    def _build_runtime_config(self) -> RuntimeKEMMConfig:
        kemm_cfg = self.demo_config.kemm
        return replace(
            kemm_cfg.runtime,
            pop_size=kemm_cfg.pop_size,
            n_var=self.context.n_var,
            n_obj=self.context.n_obj,
            benchmark_aware_prior=False,
        )

    def _ensure_algorithm(self, bounds, objective) -> KEMM_DMOEA_Improved:
        if self._algo is not None and self._algo.n_var == self.context.n_var and self._algo.n_obj == self.context.n_obj:
            if np.allclose(self._algo.lb, bounds[0]) and np.allclose(self._algo.ub, bounds[1]):
                return self._algo

        algo = KEMM_DMOEA_Improved(
            pop_size=self.demo_config.kemm.pop_size,
            n_var=self.context.n_var,
            n_obj=self.context.n_obj,
            var_bounds=bounds,
            benchmark_aware_prior=False,
            config=self._build_runtime_config(),
        )
        algo.initialize()
        self._inject_initial_guesses(algo.population)
        algo.fitness = algo.evaluate(algo.population, objective, 0.0)
        self._algo = algo
        self._solve_count = 0
        return algo

    def _inject_initial_guesses(self, population: np.ndarray) -> None:
        samples = self._initial_guess_samples(len(population))
        if samples is None or len(samples) == 0:
            return
        population[: len(samples)] = samples

    def _initial_guess_samples(self, max_count: int) -> np.ndarray | None:
        kemm_cfg = self.demo_config.kemm
        if not kemm_cfg.inject_initial_guess or max_count <= 0:
            return None
        base = np.clip(
            self.context.initial_guess.astype(float),
            self.context.var_bounds[:, 0],
            self.context.var_bounds[:, 1],
        )
        copies = min(max(1, kemm_cfg.initial_guess_copies), max_count)
        samples = np.repeat(base[None, :], copies, axis=0)
        if copies == 1:
            return samples
        scale = kemm_cfg.initial_guess_jitter_ratio * (self.context.var_bounds[:, 1] - self.context.var_bounds[:, 0])
        noise = np.random.normal(0.0, scale, size=(copies - 1, self.context.n_var))
        seeded = np.clip(base + noise, self.context.var_bounds[:, 0], self.context.var_bounds[:, 1])
        samples[1:copies] = seeded
        return samples

    def _blend_initial_guess_candidates(self, algo: KEMM_DMOEA_Improved, objective, change_time: float) -> None:
        samples = self._initial_guess_samples(max(1, min(self.demo_config.kemm.initial_guess_copies, algo.pop_size // 4)))
        if samples is None or len(samples) == 0:
            return
        candidate_pop = np.vstack([algo.population, samples])
        candidate_fit = algo.evaluate(candidate_pop, objective, change_time)
        algo.population, algo.fitness = algo.env_selection(candidate_pop, candidate_fit, algo.pop_size)

    def _summarize_generation(self, algo: KEMM_DMOEA_Improved, generation: int) -> Dict[str, float]:
        fronts = algo.fast_nds(algo.fitness)
        pareto_idx = fronts[0] if fronts else list(range(len(algo.population)))
        pf = algo.fitness[pareto_idx, :3]
        mins = np.min(algo.fitness[:, :3], axis=0)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-9)
        summary = {
            "generation": float(generation),
            "pareto_size": float(len(pareto_idx)),
            "best_fuel": float(mins[0]),
            "best_time": float(mins[1]),
            "best_risk": float(mins[2]),
            "best_weighted_score": float(np.min(algo.fitness[:, :3] @ weights)),
            "pareto_mean_fuel": float(np.mean(pf[:, 0])),
            "pareto_mean_time": float(np.mean(pf[:, 1])),
            "pareto_mean_risk": float(np.mean(pf[:, 2])),
        }
        diagnostics = getattr(algo, "last_change_diagnostics", None)
        if diagnostics is not None:
            summary["prediction_confidence"] = float(diagnostics.prediction_confidence)
            summary["response_quality"] = float(diagnostics.response_quality)
            
        ratios = getattr(algo, "last_operator_ratios", None)
        if ratios is not None and len(ratios) == 4:
            summary["ratio_memory"] = float(ratios[0])
            summary["ratio_predict"] = float(ratios[1])
            summary["ratio_transfer"] = float(ratios[2])
            summary["ratio_reinit"] = float(ratios[3])
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
        obj_only = objectives[:, :3]
        spread = np.ptp(obj_only, axis=0)
        normalized = (obj_only - obj_only.min(axis=0)) / (spread + 1e-9)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / np.sum(weights)
        scores = normalized @ weights
        return int(np.argmin(scores))
