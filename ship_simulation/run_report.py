"""批量运行船舶仿真实验并导出报告。

这个脚本对应 ship 主线的“正式实验入口”。和 `main_demo.py` 的区别是：

- `main_demo.py` 更偏向单次演示和交互观察
- `run_report.py` 更偏向批量生成结果、图表和可归档的实验报告

当前它会对内置的三类经典会遇场景逐个运行：

1. `head_on`
2. `crossing`
3. `overtaking`

每个场景同时运行：

- KEMM 求解器
- 随机搜索 baseline

然后把结果统一写入 `raw/`, `figures/`, `reports/` 三个目录，保持与 benchmark
主线一致的输出习惯。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# 兼容两种运行方式：
# 1. `python -m ship_simulation.run_report`
# 2. `python ship_simulation/run_report.py`
# 后者会把 `ship_simulation/` 当成脚本目录加入 sys.path，导致仓库根目录下的
# `reporting_config.py` 无法直接导入，因此这里在脚本直跑时补回仓库根路径。
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from reporting_config import ShipPlotConfig
from ship_simulation.config import DemoConfig, build_default_config, build_default_demo_config
from ship_simulation.main_demo import random_search, select_demo_solution
from ship_simulation.optimizer.interface import ShipOptimizerInterface
from ship_simulation.optimizer.kemm_solver import ShipKEMMOptimizer
from ship_simulation.scenario.generator import ScenarioGenerator
from ship_simulation.visualization import (
    ExperimentSeries,
    save_convergence_plot,
    save_normalized_objective_bars,
    save_pareto_scatter,
    save_risk_bars,
    save_risk_time_series,
    save_speed_profiles,
    save_summary_dashboard,
    save_trajectory_comparison,
)


def _build_quick_demo_config() -> DemoConfig:
    """构造一个用于快速冒烟验证的轻量配置。"""

    demo = build_default_demo_config()
    demo.random_search_samples = 24
    demo.kemm.pop_size = 40
    demo.kemm.generations = 16
    demo.kemm.refresh_interval = 4
    return demo


def _pick_random_baseline(
    interface: ShipOptimizerInterface,
    demo_config: DemoConfig,
) -> tuple[np.ndarray, np.ndarray, object]:
    """运行随机搜索 baseline，并挑出一个用于展示和对比的代表解。

    随机搜索会先生成很多候选解，再按加权分数排序。这里优先返回“已经到达终点”
    的方案，避免把物理上不完整的轨迹误当作示例结果。
    """

    decisions, objectives = random_search(
        interface=interface,
        n_samples=demo_config.random_search_samples,
        seed=demo_config.random_search_seed,
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

    return decisions, objectives, result


def _scenario_result_dict(
    scenario_key: str,
    optimizer_name: str,
    result,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """把单场景结果拍平成适合 CSV/JSON/Markdown 导出的字典。"""

    payload = {
        "scenario_key": scenario_key,
        "scenario_name": result.own_trajectory.__class__.__name__,
        "optimizer": optimizer_name,
        "fuel": float(result.objectives[0]),
        "time": float(result.objectives[1]),
        "risk_objective": float(result.objectives[2]),
        "max_risk": float(result.risk.max_risk),
        "mean_risk": float(result.risk.mean_risk),
        "intrusion_time": float(result.risk.intrusion_time),
        "reached_goal": bool(result.reached_goal),
        "terminal_distance": float(result.terminal_distance),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_summary_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """把结构化结果写成 CSV。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _write_summary_markdown(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """把结构化结果写成便于汇报的 Markdown 表格。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("# Ship Simulation Report\n\nNo rows generated.\n", encoding="utf-8")
        return

    headers = [
        "scenario_key",
        "optimizer",
        "fuel",
        "time",
        "risk_objective",
        "max_risk",
        "mean_risk",
        "intrusion_time",
        "reached_goal",
        "terminal_distance",
    ]
    lines = [
        "# Ship Simulation Report",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")



def generate_report(output_root: Path | None = None, plot_config: ShipPlotConfig | None = None) -> Path:
    """运行内置船舶场景并导出完整报告目录。

    目录结构固定为：

    - `raw/`：原始数值结果
    - `figures/`：图表
    - `reports/`：Markdown 摘要
    """

    config = build_default_config()
    return generate_report_with_config(
        config=config,
        demo_config=build_default_demo_config(),
        output_root=output_root,
        plot_config=plot_config,
        scenario_keys=None,
        verbose=True,
    )


def generate_report_with_config(
    config,
    demo_config: DemoConfig,
    output_root: Path | None = None,
    plot_config: ShipPlotConfig | None = None,
    scenario_keys: List[str] | None = None,
    verbose: bool = True,
) -> Path:
    """使用显式配置运行 ship 批量报告。"""

    plot_config = plot_config or ShipPlotConfig()
    generator = ScenarioGenerator(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_root or Path("ship_simulation/outputs") / f"report_{timestamp}"
    raw_dir = root / "raw"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scenario_keys = scenario_keys or ["head_on", "crossing", "overtaking"]
    summary_rows: List[Dict[str, object]] = []
    kemm_objectives = []
    random_objectives = []
    kemm_risks = []
    random_risks = []
    scenario_names = []
    t0 = time.time()

    if verbose:
        print("Ship simulation batch report started.", flush=True)
        print("This command exports figures to disk; it does not open interactive animation windows.", flush=True)
        print(f"Output directory: {root}", flush=True)
        print(f"Scenarios: {', '.join(scenario_keys)}", flush=True)

    for index, scenario_key in enumerate(scenario_keys, start=1):
        if verbose:
            print(f"[{index}/{len(scenario_keys)}] Building scenario `{scenario_key}`...", flush=True)
        scenario = generator.generate(scenario_key)
        interface = ShipOptimizerInterface(scenario=scenario, config=config)

        # KEMM 是主算法；随机搜索作为轻量 baseline，帮助判断问题是否真的存在优化空间。
        if verbose:
            print(f"[{index}/{len(scenario_keys)}] Running KEMM optimizer...", flush=True)
        kemm_result = ShipKEMMOptimizer(interface=interface, demo_config=demo_config).optimize()
        if verbose:
            print(f"[{index}/{len(scenario_keys)}] Running random baseline...", flush=True)
        _, _, random_eval = _pick_random_baseline(interface, demo_config)

        scenario_names.append(scenario.name)
        kemm_objectives.append(kemm_result.best_evaluation.objectives)
        random_objectives.append(random_eval.objectives)
        kemm_risks.append(
            [
                kemm_result.best_evaluation.risk.max_risk,
                kemm_result.best_evaluation.risk.mean_risk,
                kemm_result.best_evaluation.risk.intrusion_time,
            ]
        )
        random_risks.append(
            [
                random_eval.risk.max_risk,
                random_eval.risk.mean_risk,
                random_eval.risk.intrusion_time,
            ]
        )

        # 可视化层统一消费 ExperimentSeries，尽量不直接依赖求解器内部数据结构。
        series = [
            ExperimentSeries(
                label="KEMM",
                result=kemm_result.best_evaluation,
                color=plot_config.own_ship_color,
                history=kemm_result.history,
                pareto_objectives=kemm_result.pareto_objectives,
            ),
            ExperimentSeries(
                label="Random",
                result=random_eval,
                color=plot_config.baseline_color,
            ),
        ]

        prefix = figures_dir / scenario_key
        if verbose:
            print(f"[{index}/{len(scenario_keys)}] Saving figures for `{scenario_key}`...", flush=True)
        save_trajectory_comparison(prefix.with_name(f"{scenario_key}_trajectory.png"), scenario, series, plot_config=plot_config)
        save_risk_time_series(prefix.with_name(f"{scenario_key}_risk_series.png"), scenario.name, series, plot_config=plot_config)
        save_speed_profiles(prefix.with_name(f"{scenario_key}_speed_profile.png"), scenario.name, series, plot_config=plot_config)
        save_summary_dashboard(prefix.with_name(f"{scenario_key}_dashboard.png"), scenario, series, plot_config=plot_config)
        save_convergence_plot(
            prefix.with_name(f"{scenario_key}_kemm_convergence.png"),
            scenario.name,
            kemm_result.history,
            plot_config=plot_config,
        )
        save_pareto_scatter(
            prefix.with_name(f"{scenario_key}_kemm_pareto.png"),
            scenario.name,
            kemm_result.pareto_objectives,
            plot_config=plot_config,
        )

        summary_rows.append(
            _scenario_result_dict(
                scenario_key,
                "KEMM",
                kemm_result.best_evaluation,
                {
                    "scenario_name": scenario.name,
                    "pareto_size": int(len(kemm_result.pareto_decisions)),
                },
            )
        )
        summary_rows.append(
            _scenario_result_dict(
                scenario_key,
                "Random",
                random_eval,
                {
                    "scenario_name": scenario.name,
                    "pareto_size": 0,
                },
            )
        )

        if verbose:
            elapsed = time.time() - t0
            print(
                f"[{index}/{len(scenario_keys)}] Finished `{scenario_key}` in {elapsed:.1f}s "
                f"(best risk={kemm_result.best_evaluation.risk.max_risk:.4f}).",
                flush=True,
            )

    kemm_objectives_arr = np.asarray(kemm_objectives, dtype=float)
    random_objectives_arr = np.asarray(random_objectives, dtype=float)
    kemm_risks_arr = np.asarray(kemm_risks, dtype=float)
    random_risks_arr = np.asarray(random_risks, dtype=float)

    # 这两张图用于横向比较多个场景下的总体表现。
    save_normalized_objective_bars(
        figures_dir / "overall_normalized_objectives.png",
        scenario_names,
        kemm_objectives_arr,
        random_objectives_arr,
        plot_config=plot_config,
    )
    save_risk_bars(
        figures_dir / "overall_risk_bars.png",
        scenario_names,
        kemm_risks_arr,
        random_risks_arr,
        plot_config=plot_config,
    )

    _write_summary_csv(raw_dir / "summary.csv", summary_rows)
    _write_summary_markdown(reports_dir / "summary.md", summary_rows)
    (raw_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if verbose:
        print(f"Ship simulation report generated in {time.time() - t0:.1f}s: {root}", flush=True)

    return root


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run ship-simulation batch reports and export figures/tables.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller smoke-test configuration for faster feedback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root directory. Defaults to ship_simulation/outputs/report_<timestamp>.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["head_on", "crossing", "overtaking"],
        default=None,
        help="Optional subset of scenarios to run.",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = _parse_args()
    config = build_default_config()
    demo = _build_quick_demo_config() if args.quick else build_default_demo_config()
    report_dir = generate_report_with_config(
        config=config,
        demo_config=demo,
        output_root=args.output_dir,
        scenario_keys=args.scenarios,
        verbose=True,
    )
    print(f"Final report directory: {report_dir}", flush=True)


if __name__ == "__main__":
    main()
