"""ship_simulation/optimizer/interface.py

本文件负责对外暴露优化算法接入接口。

它的核心目标是：让现有优化算法几乎不用改结构，就能把船舶仿真问题当成一个普通多目标问题来调用。

当前已经兼容两种典型调用方式：
1. `evaluate_single(x)` / `evaluate_population(pop)` 直接评估
2. `obj_func(pop, t)` 风格，兼容你当前 benchmark 框架中的算法实现
"""

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
    """船舶优化问题适配器。

    这个类本身不实现优化算法，只负责：
    - 创建问题对象
    - 暴露统一评估接口
    - 把问题包装成你现有代码更容易消费的形式
    """

    def __init__(self, scenario: EncounterScenario, config: Optional[ProblemConfig] = None):
        self.config = config or ProblemConfig()
        self.problem = ShipTrajectoryProblem(scenario=scenario, config=self.config)

    def build_context(self) -> OptimizerContext:
        """构造适合外部算法直接使用的上下文。"""

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
        """评估单个候选解。"""

        return self.problem.evaluate(decision_vector)

    def simulate(self, decision_vector: np.ndarray) -> EvaluationResult:
        """返回带轨迹细节的完整仿真结果。"""

        return self.problem.simulate(decision_vector)

    @staticmethod
    def make_objective_function(context: OptimizerContext) -> Callable[[np.ndarray, float], np.ndarray]:
        """包装为 `obj_func(pop, t)` 风格接口。

        你的现有 DMOEA 框架里，很多算法采用这种目标函数签名：
        `fitness = obj_func(population, t)`

        这里保留参数 `t`，即使当前静态场景下并不使用它。
        这样以后若把场景做成动态环境，也不用改算法外层接口。
        """

        def objective(population: np.ndarray, t: float) -> np.ndarray:
            _ = t
            return context.evaluate_population(population)

        return objective

    @staticmethod
    def make_algorithm_kwargs(context: OptimizerContext) -> Dict[str, object]:
        """返回与现有算法构造函数风格接近的参数字典。"""

        # 这里把 numpy 边界数组转成更通用的 Python tuple 列表
        bounds_as_tuples = [tuple(row) for row in context.var_bounds.tolist()]
        return {
            "n_var": context.n_var,
            "n_obj": context.n_obj,
            "var_bounds": bounds_as_tuples,
        }
