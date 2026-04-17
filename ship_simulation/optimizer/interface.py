"""ship 主线的优化接口适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from ship_simulation.config import ProblemConfig
from ship_simulation.optimizer.problem import EvaluationResult, ShipTrajectoryProblem
from ship_simulation.scenario.encounter import EncounterScenario


@dataclass
class OptimizerContext:
    """暴露给外部优化算法的上下文对象。"""

    n_var: int
    n_obj: int
    var_bounds: np.ndarray
    evaluate_population: Callable[[np.ndarray], np.ndarray]
    evaluate_single: Callable[[np.ndarray], np.ndarray]
    initial_guess: np.ndarray
    problem: ShipTrajectoryProblem


class ShipOptimizerInterface:
    """船舶轨迹规划问题的算法适配器。"""

    def __init__(self, scenario: EncounterScenario, config: Optional[ProblemConfig] = None):
        self.config = config or ProblemConfig()
        self.problem = ShipTrajectoryProblem(scenario=scenario, config=self.config)

    def build_context(self) -> OptimizerContext:
        return OptimizerContext(
            n_var=self.problem.n_var,
            n_obj=self.problem.n_obj,
            var_bounds=self.problem.var_bounds.copy(),
            evaluate_population=self.problem.evaluate_population,
            evaluate_single=self.problem.evaluate,
            initial_guess=self.problem.initial_guess(),
            problem=self.problem,
        )

    def evaluate(self, decision_vector: np.ndarray) -> np.ndarray:
        return self.problem.evaluate(decision_vector)

    def simulate(self, decision_vector: np.ndarray) -> EvaluationResult:
        return self.problem.simulate(decision_vector)

    def simulate_population(self, population: np.ndarray, *, copy_results: bool = True) -> list[EvaluationResult]:
        return self.problem.simulate_population(population, copy_results=copy_results)

    @staticmethod
    def make_objective_function(context: OptimizerContext) -> Callable[[np.ndarray, float], np.ndarray]:
        """包装成 benchmark 主线常用的 `obj_func(pop, t)` 风格接口。"""

        def objective(population: np.ndarray, t: float) -> np.ndarray:
            _ = t
            return context.evaluate_population(population)

        return objective

    @staticmethod
    def make_algorithm_kwargs(context: OptimizerContext) -> Dict[str, object]:
        bounds_as_tuples = [tuple(row) for row in context.var_bounds.tolist()]
        return {
            "n_var": context.n_var,
            "n_obj": context.n_obj,
            "var_bounds": bounds_as_tuples,
        }
