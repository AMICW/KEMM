"""ship 主线的 NSGA-II 风格基线优化器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import time

import numpy as np

from ship_simulation.config import DemoConfig
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.problem import EvaluationResult
from ship_simulation.optimizer.selection import select_representative_index


@dataclass
class EvolutionaryOptimizationResult:
    best_decision: np.ndarray
    best_evaluation: EvaluationResult
    pareto_decisions: np.ndarray
    pareto_objectives: np.ndarray
    population: np.ndarray
    fitness: np.ndarray
    history: List[Dict[str, float]]
    runtime_s: float


class ShipNSGAStyleOptimizer:
    """一个轻量、可扩展的 NSGA-II 风格基线。"""

    def __init__(self, interface: ShipOptimizerInterface, demo_config: DemoConfig):
        self.interface = interface
        self.demo_config = demo_config
        self.context = interface.build_context()
        self.rng = np.random.default_rng(demo_config.random_search_seed + 17)

    def optimize(self) -> EvolutionaryOptimizationResult:
        pop_size = self.demo_config.evolutionary_baseline_pop_size
        generations = self.demo_config.evolutionary_baseline_generations
        lower = self.context.var_bounds[:, 0]
        upper = self.context.var_bounds[:, 1]
        population = self.rng.uniform(lower, upper, size=(pop_size, self.context.n_var))
        population[0] = self.context.initial_guess
        fitness = self.context.evaluate_population(population)
        history = [self._summarize_generation(fitness, 0)]
        t0 = time.perf_counter()

        for generation in range(1, generations + 1):
            offspring = self._generate_offspring(population, fitness)
            offspring_fitness = self.context.evaluate_population(offspring)
            merged_pop = np.vstack([population, offspring])
            merged_fit = np.vstack([fitness, offspring_fitness])
            population, fitness = self._environmental_selection(merged_pop, merged_fit, pop_size)
            history.append(self._summarize_generation(fitness, generation))

        fronts = self._fast_nondominated_sort(fitness)
        pareto_idx = np.asarray(fronts[0], dtype=int)
        pareto_decisions = population[pareto_idx].copy()
        pareto_objectives = fitness[pareto_idx, :3].copy()
        best_decision, best_evaluation = self._select_representative_solution(pareto_decisions, pareto_objectives)
        return EvolutionaryOptimizationResult(
            best_decision=best_decision,
            best_evaluation=best_evaluation,
            pareto_decisions=pareto_decisions,
            pareto_objectives=pareto_objectives,
            population=population.copy(),
            fitness=fitness.copy(),
            history=history,
            runtime_s=time.perf_counter() - t0,
        )

    def _generate_offspring(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        lower = self.context.var_bounds[:, 0]
        upper = self.context.var_bounds[:, 1]
        fronts = self._fast_nondominated_sort(fitness)
        ranks = np.zeros(len(population), dtype=int)
        crowding = np.zeros(len(population), dtype=float)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            crowding[front] = self._crowding_distance(fitness[front])

        offspring = np.zeros_like(population)
        for idx in range(len(population)):
            p1 = population[self._tournament(ranks, crowding)]
            p2 = population[self._tournament(ranks, crowding)]
            child = 0.5 * (p1 + p2)
            if self.rng.random() < 0.75:
                child += self.rng.normal(0.0, 0.08 * (upper - lower), size=self.context.n_var)
            child = np.clip(child, lower, upper)
            offspring[idx] = child
        offspring[0] = self.context.initial_guess
        return offspring

    def _environmental_selection(self, population: np.ndarray, fitness: np.ndarray, pop_size: int) -> tuple[np.ndarray, np.ndarray]:
        fronts = self._fast_nondominated_sort(fitness)
        selected_indices: list[int] = []
        for front in fronts:
            if len(selected_indices) + len(front) <= pop_size:
                selected_indices.extend(front)
                continue
            front_fit = fitness[front]
            distances = self._crowding_distance(front_fit)
            order = np.argsort(-distances)
            selected_indices.extend([front[idx] for idx in order[: pop_size - len(selected_indices)]])
            break
        selected = np.asarray(selected_indices, dtype=int)
        return population[selected], fitness[selected]

    def _dominates(self, p: np.ndarray, q: np.ndarray) -> bool:
        if len(p) > 3:
            p_obj, p_cv, q_obj, q_cv = p[:3], sum(p[3:]), q[:3], sum(q[3:])
            if p_cv < q_cv: return True
            if p_cv > q_cv: return False
            if abs(p_cv - q_cv) < 1e-9:
                return np.all(p_obj <= q_obj) and np.any(p_obj < q_obj)
        return np.all(p <= q) and np.any(p < q)

    def _fast_nondominated_sort(self, fitness: np.ndarray) -> list[list[int]]:
        dominates = [set() for _ in range(len(fitness))]
        domination_count = np.zeros(len(fitness), dtype=int)
        fronts: list[list[int]] = [[]]
        for i in range(len(fitness)):
            for j in range(len(fitness)):
                if i == j:
                    continue
                if self._dominates(fitness[i], fitness[j]):
                    dominates[i].add(j)
                elif self._dominates(fitness[j], fitness[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)
        rank = 0
        while rank < len(fronts) and fronts[rank]:
            next_front: list[int] = []
            for i in fronts[rank]:
                for j in dominates[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            rank += 1
        return fronts

    def _crowding_distance(self, front_fitness: np.ndarray) -> np.ndarray:
        if len(front_fitness) == 0:
            return np.zeros(0, dtype=float)
        if len(front_fitness) <= 2:
            return np.full(len(front_fitness), np.inf, dtype=float)
        distances = np.zeros(len(front_fitness), dtype=float)
        for obj in range(min(3, front_fitness.shape[1])):
            order = np.argsort(front_fitness[:, obj])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            span = front_fitness[order[-1], obj] - front_fitness[order[0], obj]
            if span <= 1e-12:
                continue
            for idx in range(1, len(order) - 1):
                distances[order[idx]] += (front_fitness[order[idx + 1], obj] - front_fitness[order[idx - 1], obj]) / span
        return distances

    def _tournament(self, ranks: np.ndarray, crowding: np.ndarray) -> int:
        i, j = self.rng.integers(0, len(ranks), size=2)
        if ranks[i] < ranks[j]:
            return int(i)
        if ranks[j] < ranks[i]:
            return int(j)
        return int(i if crowding[i] >= crowding[j] else j)

    def _pick_by_weighted_score(self, objectives: np.ndarray) -> int:
        if len(objectives) == 1:
            return 0
        obj_only = objectives[:, :3]
        spread = np.ptp(obj_only, axis=0)
        normalized = (obj_only - obj_only.min(axis=0)) / (spread + 1e-9)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / np.sum(weights)
        return int(np.argmin(normalized @ weights))

    def _select_representative_solution(self, decisions: np.ndarray, objectives: np.ndarray) -> tuple[np.ndarray, EvaluationResult]:
        evaluations = [self.interface.simulate(ind) for ind in decisions]
        chosen = select_representative_index(
            objectives,
            evaluations,
            self.interface.config.objective_weights,
            safety_clearance=self.interface.config.safety_clearance,
        )
        return decisions[chosen].copy(), evaluations[chosen]

    def _summarize_generation(self, fitness: np.ndarray, generation: int) -> Dict[str, float]:
        mins = np.min(fitness[:, :3], axis=0)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-9)
        return {
            "generation": float(generation),
            "best_fuel": float(mins[0]),
            "best_time": float(mins[1]),
            "best_risk": float(mins[2]),
            "best_weighted_score": float(np.min(fitness[:, :3] @ weights)),
        }
