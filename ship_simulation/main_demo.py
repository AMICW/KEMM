"""ship 主线最小演示入口。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ship_simulation.config import DemoConfig, ProblemConfig, build_default_config, build_default_demo_config
from ship_simulation.optimizer.episode import RollingHorizonPlanner
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.scenario.generator import ScenarioGenerator
from ship_simulation.visualization.animator import TrajectoryAnimator


def random_search(interface: ShipOptimizerInterface, n_samples: int = 80, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """保留给外部调用的随机搜索基线。"""

    context = interface.build_context()
    rng = np.random.default_rng(seed)
    lower = context.var_bounds[:, 0]
    upper = context.var_bounds[:, 1]
    samples = rng.uniform(lower, upper, size=(n_samples, context.n_var))
    samples[0] = context.initial_guess
    objectives = context.evaluate_population(samples)
    return samples, objectives


def select_demo_solution(
    decisions: np.ndarray,
    objectives: np.ndarray,
    objective_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """从候选解里选一个折中解。"""

    spread = np.ptp(objectives, axis=0)
    normalized = (objectives - objectives.min(axis=0)) / (spread + 1e-9)
    weights = np.asarray(objective_weights, dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-9)
    scores = normalized @ weights
    return decisions[int(np.argmin(scores))]


def run_demo(
    scenario_type: str = "crossing",
    optimizer_name: str = "kemm",
    show_animation: bool = True,
    demo_config: DemoConfig | None = None,
) -> dict:
    """运行一段完整的 ship episode 演示。"""

    config: ProblemConfig = build_default_config()
    runtime = demo_config or build_default_demo_config()
    runtime.optimizer_name = optimizer_name
    scenario = ScenarioGenerator(config).generate(scenario_type)
    planner = RollingHorizonPlanner(scenario=scenario, config=config, demo_config=runtime)
    episode = planner.run(runtime.optimizer_name)

    summary = {
        "scenario": scenario.name,
        "optimizer": runtime.optimizer_name,
        "best_objectives": episode.final_evaluation.objectives.tolist(),
        "risk": {
            "max_risk": episode.final_evaluation.risk.max_risk,
            "mean_risk": episode.final_evaluation.risk.mean_risk,
            "intrusion_time": episode.final_evaluation.risk.intrusion_time,
        },
        "reached_goal": episode.final_evaluation.reached_goal,
        "terminal_distance": episode.final_evaluation.terminal_distance,
        "planning_steps": len(episode.steps),
        "pareto_size": int(len(episode.pareto_objectives)),
        "knee_objectives": episode.knee_objectives.tolist() if episode.knee_objectives is not None else None,
        "terminated_reason": episode.terminated_reason,
        "analysis_metrics": episode.analysis_metrics,
    }

    print("Ship simulation demo summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if show_animation:
        animator = TrajectoryAnimator(scenario=scenario, config=config)
        animator.show_episode(episode)
    else:
        plt.close("all")

    return summary


if __name__ == "__main__":
    run_demo()
