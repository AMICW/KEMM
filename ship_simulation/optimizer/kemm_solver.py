"""ship_simulation/optimizer/kemm_solver.py

本文件负责把外部 benchmark 中的最强版 KEMM 算法，
包装成适合 ship_simulation 直接调用的求解器。

设计原则：
1. 不改 benchmark 侧 KEMM 主体实现，避免破坏你现有实验代码。
2. 在船舶问题侧补足工程化能力：初值注入、运行历史、结果筛选。
3. 对静态轨迹规划问题做轻量适配：周期性调用 respond_to_change()，
   让 KEMM 的 memory/predict/transfer 机制也能参与搜索，而不是退化成
   单纯的 NSGA-II 迭代器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from benchmark_algorithms import KEMM_DMOEA_Improved
from ship_simulation.config import DemoConfig
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.problem import EvaluationResult


@dataclass
class KEMMOptimizationResult:
    """KEMM 求解完成后的结果对象。"""

    best_decision: np.ndarray
    best_evaluation: EvaluationResult
    pareto_decisions: np.ndarray
    pareto_objectives: np.ndarray
    population: np.ndarray
    fitness: np.ndarray
    history: List[Dict[str, float]]


class ShipKEMMOptimizer:
    """船舶轨迹规划问题上的 KEMM 运行器。"""

    def __init__(self, interface: ShipOptimizerInterface, demo_config: DemoConfig):
        self.interface = interface
        self.demo_config = demo_config
        self.context = interface.build_context()

    def optimize(self) -> KEMMOptimizationResult:
        """运行一次完整 KEMM 优化。"""

        kemm_cfg = self.demo_config.kemm
        bounds = (
            self.context.var_bounds[:, 0].astype(float),
            self.context.var_bounds[:, 1].astype(float),
        )
        objective = self.interface.make_objective_function(self.context)

        # benchmark 代码大量使用 numpy 全局随机数，这里显式保存与恢复状态，
        # 避免 ship_simulation 的调用把外部全局随机流污染掉。
        old_state = np.random.get_state()
        np.random.seed(kemm_cfg.seed)
        try:
            algo = KEMM_DMOEA_Improved(
                pop_size=kemm_cfg.pop_size,
                n_var=self.context.n_var,
                n_obj=self.context.n_obj,
                var_bounds=bounds,
            )
            algo.initialize()
            self._inject_initial_guesses(algo.population)
            algo.fitness = algo.evaluate(algo.population, objective, 0.0)

            history: List[Dict[str, float]] = [self._summarize_generation(algo, generation=0)]
            for generation in range(1, kemm_cfg.generations + 1):
                if kemm_cfg.refresh_interval > 0 and generation % kemm_cfg.refresh_interval == 0:
                    algo.respond_to_change(objective, float(generation))
                else:
                    algo.evolve_one_gen(objective, float(generation))
                history.append(self._summarize_generation(algo, generation=generation))

            fronts = algo.fast_nds(algo.fitness)
            pareto_idx = np.asarray(fronts[0] if fronts else np.arange(len(algo.population)), dtype=int)
            pareto_decisions = algo.population[pareto_idx].copy()
            pareto_objectives = algo.fitness[pareto_idx].copy()
            best_decision, best_evaluation = self._select_representative_solution(
                pareto_decisions,
                pareto_objectives,
            )

            return KEMMOptimizationResult(
                best_decision=best_decision,
                best_evaluation=best_evaluation,
                pareto_decisions=pareto_decisions,
                pareto_objectives=pareto_objectives,
                population=algo.population.copy(),
                fitness=algo.fitness.copy(),
                history=history,
            )
        finally:
            np.random.set_state(old_state)

    def _inject_initial_guesses(self, population: np.ndarray) -> None:
        """把直线初始解及其邻域样本注入初代。"""

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

        scale = kemm_cfg.initial_guess_jitter_ratio * (
            self.context.var_bounds[:, 1] - self.context.var_bounds[:, 0]
        )
        noise = np.random.normal(0.0, scale, size=(copies - 1, self.context.n_var))
        seeded = np.clip(base + noise, self.context.var_bounds[:, 0], self.context.var_bounds[:, 1])
        population[1:copies] = seeded

    def _summarize_generation(self, algo: KEMM_DMOEA_Improved, generation: int) -> Dict[str, float]:
        """提取每一代的简要统计量，便于后续分析收敛行为。"""

        fronts = algo.fast_nds(algo.fitness)
        pareto_idx = fronts[0] if fronts else list(range(len(algo.population)))
        pf = algo.fitness[pareto_idx]
        mins = np.min(algo.fitness, axis=0)
        return {
            "generation": float(generation),
            "pareto_size": float(len(pareto_idx)),
            "best_fuel": float(mins[0]),
            "best_time": float(mins[1]),
            "best_risk": float(mins[2]),
            "pareto_mean_fuel": float(np.mean(pf[:, 0])),
            "pareto_mean_time": float(np.mean(pf[:, 1])),
            "pareto_mean_risk": float(np.mean(pf[:, 2])),
        }

    def _select_representative_solution(
        self,
        decisions: np.ndarray,
        objectives: np.ndarray,
    ) -> tuple[np.ndarray, EvaluationResult]:
        """从 Pareto 解集中选一个适合演示的代表方案。

        策略：
        1. 优先在“到达终点”的方案中选。
        2. 再按目标归一化后的加权分数挑折中解。
        3. 若前沿里都不满足到达条件，则退化为全体中分数最好的方案。
        """

        evaluations = [self.interface.simulate(ind) for ind in decisions]
        feasible_indices = [idx for idx, item in enumerate(evaluations) if item.reached_goal]

        if feasible_indices:
            chosen = self._pick_by_weighted_score(decisions[feasible_indices], objectives[feasible_indices])
            real_index = feasible_indices[chosen]
        else:
            chosen = self._pick_by_weighted_score(decisions, objectives)
            real_index = chosen

        return decisions[real_index].copy(), evaluations[real_index]

    def _pick_by_weighted_score(self, decisions: np.ndarray, objectives: np.ndarray) -> int:
        """用配置中的权重从一组目标值里选分数最低的候选。"""

        if len(decisions) == 1:
            return 0

        spread = np.ptp(objectives, axis=0)
        normalized = (objectives - objectives.min(axis=0)) / (spread + 1e-9)
        weights = np.asarray(self.interface.config.objective_weights, dtype=float)
        weights = weights / np.sum(weights)
        scores = normalized @ weights
        return int(np.argmin(scores))
