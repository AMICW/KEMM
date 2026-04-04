"""ship_simulation/main_demo.py

本文件是整个框架的最小演示入口。

它串起了完整流程：
1. 读取默认配置
2. 生成一个会遇场景
3. 构造优化问题接口
4. 用一个简单随机搜索做 baseline
5. 选择一个较平衡的解进行仿真与动画展示

注意：
这里的 `random_search()` 不是正式优化算法，只是为了验证框架闭环是否能跑通。
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from ship_simulation.config import (
    DemoConfig,
    ProblemConfig,
    build_default_config,
    build_default_demo_config,
)
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.kemm_solver import ShipKEMMOptimizer
from ship_simulation.scenario.generator import ScenarioGenerator
from ship_simulation.visualization.animator import TrajectoryAnimator


def random_search(interface: ShipOptimizerInterface, n_samples: int = 80, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """简单随机搜索基线。

    作用：
    - 验证接口是否可运行
    - 生成一些候选解供可视化
    - 作为后续正式算法对照基线
    """

    context = interface.build_context()
    rng = np.random.default_rng(seed)
    lower = context.var_bounds[:, 0]
    upper = context.var_bounds[:, 1]
    samples = rng.uniform(lower, upper, size=(n_samples, context.n_var))
    # 把第一条样本替换成直线初始解，保证至少有一个较规则的候选轨迹
    samples[0] = context.initial_guess
    objectives = context.evaluate_population(samples)
    return samples, objectives


def select_demo_solution(decisions: np.ndarray, objectives: np.ndarray) -> np.ndarray:
    """从若干候选解中挑一个较平衡的演示解。

    这里不是严格 Pareto 决策，只是做一个简单归一化加权评分，
    以便选择一个可视化上比较“像样”的轨迹。
    """

    spread = np.ptp(objectives, axis=0)
    normalized = (objectives - objectives.min(axis=0)) / (spread + 1e-9)
    scores = 0.4 * normalized[:, 0] + 0.25 * normalized[:, 1] + 0.35 * normalized[:, 2]
    return decisions[int(np.argmin(scores))]


def _legacy_run_demo(scenario_type: str = "crossing", show_animation: bool = True) -> dict:
    """运行完整演示流程。"""

    config: ProblemConfig = build_default_config()
    scenario = ScenarioGenerator(config).generate(scenario_type)
    interface = ShipOptimizerInterface(scenario=scenario, config=config)

    # 先用随机搜索生成一批候选解
    decisions, objectives = random_search(interface)
    spread = np.ptp(objectives, axis=0)
    normalized = (objectives - objectives.min(axis=0)) / (spread + 1e-9)
    scores = 0.4 * normalized[:, 0] + 0.25 * normalized[:, 1] + 0.35 * normalized[:, 2]
    ranked_indices = np.argsort(scores)

    selected = None
    result = None
    for idx in ranked_indices:
        candidate = decisions[int(idx)]
        candidate_result = interface.simulate(candidate)
        if candidate_result.reached_goal:
            selected = candidate
            result = candidate_result
            break

    if result is None:
        selected = select_demo_solution(decisions, objectives)

    # 对被选中的方案做完整仿真，拿到轨迹和风险细节
    if result is None:
        result = interface.simulate(selected)

    summary = {
        "scenario": scenario.name,
        "best_objectives": result.objectives.tolist(),
        "risk": {
            "max_risk": result.risk.max_risk,
            "mean_risk": result.risk.mean_risk,
            "intrusion_time": result.risk.intrusion_time,
        },
        "reached_goal": result.reached_goal,
        "terminal_distance": result.terminal_distance,
        "n_samples": int(len(decisions)),
    }

    print("Ship simulation demo summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if show_animation:
        # 如果开启动画，则展示本船和目标船的动态轨迹
        animator = TrajectoryAnimator(scenario=scenario, config=config)
        animator.show(result)
    else:
        plt.close("all")

    return summary

def run_demo(
    scenario_type: str = "crossing",
    optimizer_name: str = "kemm",
    show_animation: bool = True,
    demo_config: DemoConfig | None = None,
) -> dict:
    """运行完整演示流程。

    这里重新定义 `run_demo`，默认切换到最强版 KEMM。
    保留 `optimizer_name="random"` 入口，便于和基线快速对照。
    """

    config: ProblemConfig = build_default_config()
    runtime = demo_config or build_default_demo_config()
    runtime.optimizer_name = optimizer_name
    scenario = ScenarioGenerator(config).generate(scenario_type)
    interface = ShipOptimizerInterface(scenario=scenario, config=config)

    if runtime.optimizer_name.lower() == "kemm":
        kemm_result = ShipKEMMOptimizer(interface=interface, demo_config=runtime).optimize()
        selected = kemm_result.best_decision
        result = kemm_result.best_evaluation
        decisions = kemm_result.population
        pareto_size = int(len(kemm_result.pareto_decisions))
        history_tail = kemm_result.history[-3:]
    else:
        decisions, objectives = random_search(
            interface=interface,
            n_samples=runtime.random_search_samples,
            seed=runtime.random_search_seed,
        )
        spread = np.ptp(objectives, axis=0)
        normalized = (objectives - objectives.min(axis=0)) / (spread + 1e-9)
        scores = 0.4 * normalized[:, 0] + 0.25 * normalized[:, 1] + 0.35 * normalized[:, 2]
        ranked_indices = np.argsort(scores)

        selected = None
        result = None
        for idx in ranked_indices:
            candidate = decisions[int(idx)]
            candidate_result = interface.simulate(candidate)
            if candidate_result.reached_goal:
                selected = candidate
                result = candidate_result
                break

        if result is None:
            selected = select_demo_solution(decisions, objectives)
            result = interface.simulate(selected)

        pareto_size = 0
        history_tail = []

    summary = {
        "scenario": scenario.name,
        "optimizer": runtime.optimizer_name,
        "best_objectives": result.objectives.tolist(),
        "risk": {
            "max_risk": result.risk.max_risk,
            "mean_risk": result.risk.mean_risk,
            "intrusion_time": result.risk.intrusion_time,
        },
        "reached_goal": result.reached_goal,
        "terminal_distance": result.terminal_distance,
        "n_samples": int(len(decisions)),
        "pareto_size": pareto_size,
        "history_tail": history_tail,
        "selected_decision": selected.tolist(),
    }

    print("Ship simulation demo summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if show_animation:
        animator = TrajectoryAnimator(scenario=scenario, config=config)
        animator.show(result)
    else:
        plt.close("all")

    return summary


if __name__ == "__main__":
    run_demo()
