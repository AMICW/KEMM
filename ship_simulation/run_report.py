"""批量运行 ship episode，并导出论文风格报告。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from reporting_config import ShipPlotConfig, build_ship_plot_config
from ship_simulation.config import DemoConfig, apply_experiment_profile, build_default_config, build_default_demo_config
from ship_simulation.optimizer.episode import PlanningEpisodeResult, RollingHorizonPlanner
from ship_simulation.scenario.generator import ScenarioGenerator
from ship_simulation.visualization import (
    ExperimentSeries,
    save_change_timeline_panel,
    save_control_time_series,
    save_convergence_statistics,
    save_distribution_violin,
    save_dynamic_avoidance_snapshots,
    save_environment_overlay,
    save_normalized_objective_bars,
    save_parallel_coordinates,
    save_pareto_3d_with_knee,
    save_pareto_projection_panel,
    save_radar_chart,
    save_risk_breakdown_time_series,
    save_risk_bars,
    save_route_planning_panel,
    save_route_bundle_gallery,
    save_run_statistics_panel,
    save_runtime_tradeoff,
    save_decision_space_projection,
    save_safety_envelope_plot,
    save_scenario_gallery,
    save_spatiotemporal_plot,
    save_summary_dashboard,
    save_operator_allocation_history,
)


@dataclass(frozen=True)
class ScenarioFigureSpec:
    suffix: str
    meaning: str
    renderer_name: str


@dataclass(frozen=True)
class GlobalFigureSpec:
    file_name: str
    meaning: str
    renderer_name: str


@dataclass(frozen=True)
class AlgorithmReportSpec:
    key: str
    label: str
    color_attr: str


@dataclass
class ScenarioFigureContext:
    scenario_key: str
    scenario: object
    best_series: Sequence[ExperimentSeries]
    histories_by_label: Mapping[str, Sequence[Sequence[dict[str, float]]]]
    metrics_by_label: Mapping[str, Sequence[dict[str, float]]]


@dataclass
class GlobalFigureContext:
    scenario_map: Mapping[str, object]
    aggregate_payload: Mapping[str, Mapping[str, Sequence[PlanningEpisodeResult]]]
    algorithm_specs: Sequence[AlgorithmReportSpec]


def _build_quick_demo_config() -> DemoConfig:
    demo = build_default_demo_config()
    demo.random_search_samples = 20
    demo.evolutionary_baseline_pop_size = 22
    demo.evolutionary_baseline_generations = 10
    demo.n_runs = 1
    demo.kemm.pop_size = 28
    demo.kemm.generations = 12
    demo.kemm.initial_guess_copies = 4
    demo.episode.local_horizon = 320.0
    demo.episode.execution_horizon = 160.0
    demo.episode.max_replans = 8
    return demo


def _recommended_worker_count(limit: int = 4) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return max(1, min(limit, cpu_count - 1))


def _ship_report_task(task: dict[str, object]) -> dict[str, object]:
    scenario_key = str(task["scenario_key"])
    algorithm = str(task["algorithm"])
    run_index = int(task["run_index"])
    scenario_index = int(task["scenario_index"])
    algorithm_index = int(task["algorithm_index"])
    config = task["config"]
    demo_config = task["demo_config"]

    scenario = ScenarioGenerator(config).generate(scenario_key)
    run_demo = replace(
        demo_config,
        random_search_seed=demo_config.random_search_seed + run_index,
        kemm=replace(demo_config.kemm, seed=demo_config.kemm.seed + run_index),
    )
    planner = RollingHorizonPlanner(scenario=scenario, config=config, demo_config=run_demo)
    episode = planner.run(optimizer_name=algorithm)
    return {
        "scenario_key": scenario_key,
        "algorithm": algorithm,
        "run_index": run_index,
        "scenario_index": scenario_index,
        "algorithm_index": algorithm_index,
        "episode": episode,
    }


def _normalized_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    total = float(np.sum(arr))
    if total <= 1e-12:
        return np.full(len(arr), 1.0 / max(len(arr), 1), dtype=float)
    return arr / total


def _weighted_score(objectives: np.ndarray, weights: Sequence[float]) -> float:
    return float(np.dot(objectives, _normalized_weights(weights)))


def _optimizer_display_name(name: str) -> str:
    return next((spec.label for spec in DEFAULT_ALGORITHM_SPECS if spec.key == name), name)


DEFAULT_ALGORITHM_SPECS: tuple[AlgorithmReportSpec, ...] = (
    AlgorithmReportSpec("kemm", "KEMM", "own_ship_color"),
    AlgorithmReportSpec("nsga_style", "NSGA-style", "third_algo_color"),
    AlgorithmReportSpec("random", "Random", "baseline_color"),
)


def _resolve_algorithm_specs(demo_config: DemoConfig) -> list[AlgorithmReportSpec]:
    registry = {spec.key: spec for spec in DEFAULT_ALGORITHM_SPECS}
    specs: list[AlgorithmReportSpec] = []
    for key in demo_config.report_algorithms:
        spec = registry.get(key)
        if spec is None:
            spec = AlgorithmReportSpec(key=key, label=key, color_attr="baseline_color")
        specs.append(spec)
    return specs


def _algorithm_color(spec: AlgorithmReportSpec, plot_config: ShipPlotConfig, index: int) -> str:
    color = getattr(plot_config, spec.color_attr, None)
    if color is not None:
        return str(color)
    fallback = [plot_config.own_ship_color, plot_config.third_algo_color, plot_config.baseline_color]
    return str(fallback[index % len(fallback)])


def _episode_row(scenario_key: str, optimizer_name: str, run_index: int, episode: PlanningEpisodeResult) -> dict[str, object]:
    metrics = episode.analysis_metrics
    knee = episode.knee_objectives if episode.knee_objectives is not None else np.array([np.nan, np.nan, np.nan], dtype=float)
    return {
        "scenario_key": scenario_key,
        "scenario_name": episode.scenario_name,
        "optimizer": _optimizer_display_name(optimizer_name),
        "run": int(run_index),
        "experiment_profile": episode.experiment_profile,
        "fuel": float(episode.final_evaluation.objectives[0]),
        "time": float(episode.final_evaluation.objectives[1]),
        "risk": float(episode.final_evaluation.objectives[2]),
        "max_risk": float(episode.final_evaluation.risk.max_risk),
        "mean_risk": float(episode.final_evaluation.risk.mean_risk),
        "intrusion_time": float(episode.final_evaluation.risk.intrusion_time),
        "reached_goal": bool(episode.final_evaluation.reached_goal),
        "terminal_distance": float(episode.final_evaluation.terminal_distance),
        "minimum_clearance": float(metrics.get("minimum_clearance", 0.0)),
        "minimum_static_clearance": float(metrics.get("minimum_static_clearance", 0.0)),
        "minimum_ship_distance": float(metrics.get("minimum_ship_distance", 0.0)),
        "minimum_dcpa": float(metrics.get("minimum_dcpa", 0.0)),
        "minimum_tcpa": float(metrics.get("minimum_tcpa", 0.0)),
        "smoothness": float(metrics.get("smoothness", 0.0)),
        "control_effort": float(metrics.get("control_effort", 0.0)),
        "heading_variation": float(metrics.get("heading_variation", 0.0)),
        "max_yaw_rate": float(metrics.get("max_yaw_rate", 0.0)),
        "max_commanded_yaw_rate": float(metrics.get("max_commanded_yaw_rate", 0.0)),
        "runtime": float(metrics.get("runtime", 0.0)),
        "planning_steps": float(metrics.get("planning_steps", 0.0)),
        "scheduled_change_count": float(metrics.get("scheduled_change_count", 0.0)),
        "pareto_size": float(len(episode.pareto_objectives)),
        "snapshot_count": float(len(episode.snapshots)),
        "knee_index": float(episode.knee_index) if episode.knee_index is not None else np.nan,
        "knee_fuel": float(knee[0]),
        "knee_time": float(knee[1]),
        "knee_risk": float(knee[2]),
        "terminated_reason": episode.terminated_reason,
    }


def _aggregate_rows(rows: List[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario_key"]), str(row["optimizer"]))].append(row)
    aggregates = []
    numeric_keys = [
        "fuel",
        "time",
        "risk",
        "max_risk",
        "mean_risk",
        "intrusion_time",
        "terminal_distance",
        "minimum_clearance",
        "minimum_static_clearance",
        "minimum_ship_distance",
        "minimum_dcpa",
        "minimum_tcpa",
        "smoothness",
        "control_effort",
        "heading_variation",
        "max_yaw_rate",
        "max_commanded_yaw_rate",
        "runtime",
        "planning_steps",
        "scheduled_change_count",
        "pareto_size",
        "snapshot_count",
        "knee_index",
        "knee_fuel",
        "knee_time",
        "knee_risk",
    ]
    for (scenario_key, optimizer), items in grouped.items():
        payload = {
            "scenario_key": scenario_key,
            "optimizer": optimizer,
            "n_runs": len(items),
            "success_rate": float(np.mean([1.0 if item["reached_goal"] else 0.0 for item in items])),
        }
        for key in numeric_keys:
            values = np.asarray([float(item[key]) for item in items], dtype=float)
            payload[f"{key}_mean"] = float(np.mean(values))
            payload[f"{key}_std"] = float(np.std(values))
        aggregates.append(payload)
    return aggregates


def _representative_episode_index(episodes: Iterable[PlanningEpisodeResult], objective_weights: Sequence[float]) -> int:
    episode_list = list(episodes)
    if not episode_list:
        raise ValueError("At least one episode is required.")

    successful = [episode for episode in episode_list if episode.final_evaluation.reached_goal]
    candidates = successful or episode_list
    objective_matrix = np.asarray([episode.final_evaluation.objectives for episode in candidates], dtype=float)
    center = np.median(objective_matrix, axis=0)
    span = np.ptp(objective_matrix, axis=0)
    weighted_scores = np.asarray([_weighted_score(episode.final_evaluation.objectives, objective_weights) for episode in candidates], dtype=float)
    target_score = float(np.median(weighted_scores))

    def ranking_key(index: int) -> tuple[float, float, float]:
        normalized_distance = float(np.linalg.norm((objective_matrix[index] - center) / (span + 1e-9)))
        weighted_gap = abs(float(weighted_scores[index]) - target_score)
        runtime = float(candidates[index].analysis_metrics.get("runtime", 0.0))
        return normalized_distance, weighted_gap, runtime

    chosen = min(range(len(candidates)), key=ranking_key)
    return episode_list.index(candidates[chosen])


def _representative_episode(episodes: Iterable[PlanningEpisodeResult], objective_weights: Sequence[float]) -> PlanningEpisodeResult:
    episode_list = list(episodes)
    return episode_list[_representative_episode_index(episode_list, objective_weights)]


def _repeated_statistics(episodes: Iterable[PlanningEpisodeResult]) -> dict[str, float]:
    episode_list = list(episodes)
    if not episode_list:
        return {}
    keys = [
        "fuel",
        "time",
        "risk",
        "minimum_clearance",
        "minimum_static_clearance",
        "minimum_ship_distance",
        "minimum_dcpa",
        "minimum_tcpa",
        "runtime",
        "planning_steps",
    ]
    stats: dict[str, float] = {"n_runs": float(len(episode_list))}
    success = np.asarray([1.0 if episode.final_evaluation.reached_goal else 0.0 for episode in episode_list], dtype=float)
    stats["success_rate"] = float(np.mean(success))
    stats["success_rate_std"] = float(np.std(success))
    objective_index = {"fuel": 0, "time": 1, "risk": 2}
    for key in keys:
        if key in objective_index:
            idx = objective_index[key]
            values = np.asarray([float(episode.final_evaluation.objectives[idx]) for episode in episode_list], dtype=float)
        else:
            values = np.asarray([float(episode.analysis_metrics.get(key, 0.0)) for episode in episode_list], dtype=float)
        stats[f"{key}_mean"] = float(np.mean(values))
        stats[f"{key}_std"] = float(np.std(values))
    return stats


def _write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(output_path: Path, payload: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(output_path: Path, aggregates: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ship Simulation Report",
        "",
        "> Tables below aggregate all repeated runs. Trajectory-like figures use representative runs recorded in `raw/representative_runs.json`.",
        "",
        "| Scenario | Optimizer | Fuel | Time | Risk | Clearance | Ship Dist | Runtime | Knee (F/T/R) | Success Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {scenario_key} | {optimizer} | {fuel_mean:.3f} | {time_mean:.3f} | {risk_mean:.3f} | {minimum_clearance_mean:.3f} | {minimum_ship_distance_mean:.3f} | {runtime_mean:.3f} | ({knee_fuel_mean:.2f}, {knee_time_mean:.2f}, {knee_risk_mean:.2f}) | {success_rate:.2%} |".format(**row)
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scenario_catalog_entry(scenario, generation_config: object) -> dict[str, object]:
    circular_count = sum(hasattr(obstacle, "radius") for obstacle in scenario.static_obstacles)
    polygon_count = len(scenario.static_obstacles) - circular_count
    return {
        "name": scenario.name,
        "area": list(scenario.area),
        "metadata": asdict(scenario.metadata),
        "counts": {
            "targets": len(scenario.target_ships),
            "static_obstacles": len(scenario.static_obstacles),
            "circular_obstacles": circular_count,
            "polygon_obstacles": polygon_count,
            "scalar_fields": len(scenario.scalar_fields),
            "vector_fields": len(scenario.vector_fields),
        },
        "generation_config": asdict(generation_config),
    }


def _write_figure_inventory(output_path: Path, scenario_keys: Iterable[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure Inventory",
        "",
        "## Global Figures",
        "",
        "| File | Meaning |",
        "| --- | --- |",
    ]
    for spec in GLOBAL_FIGURE_SPECS:
        lines.append(f"| {spec.file_name} | {spec.meaning} |")
    lines.extend([
        "",
        "## Per-Scenario Figures",
        "",
        "| File Pattern | Meaning |",
        "| --- | --- |",
    ])
    for scenario_key in scenario_keys:
        for spec in SCENARIO_FIGURE_SPECS:
            lines.append(f"| {scenario_key}_{spec.suffix}.png | {spec.meaning} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _science_style_tuple(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _figure_manifest(scenario_keys: Sequence[str]) -> dict[str, object]:
    return {
        "global_figures": [
            {"file_name": spec.file_name, "meaning": spec.meaning, "renderer": spec.renderer_name}
            for spec in GLOBAL_FIGURE_SPECS
        ],
        "scenario_figures": [
            {
                "suffix": spec.suffix,
                "meaning": spec.meaning,
                "renderer": spec.renderer_name,
                "files": [f"{scenario_key}_{spec.suffix}.png" for scenario_key in scenario_keys],
            }
            for spec in SCENARIO_FIGURE_SPECS
        ],
    }


def _render_environment_overlay(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_environment_overlay(output_path, context.scenario, context.best_series, plot_config=plot_config)


def _render_route_planning_panel(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_route_planning_panel(output_path, context.scenario, context.best_series, plot_config=plot_config)


def _render_change_timeline(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_change_timeline_panel(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_snapshots(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_dynamic_avoidance_snapshots(output_path, context.scenario, context.best_series[0].episode, plot_config=plot_config)


def _render_spatiotemporal(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_spatiotemporal_plot(output_path, context.scenario, context.best_series[0].episode, plot_config=plot_config)


def _render_control_timeseries(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_control_time_series(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_pareto3d(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_pareto_3d_with_knee(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_pareto_projection(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_pareto_projection_panel(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_risk_breakdown(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_risk_breakdown_time_series(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_safety_envelope(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_safety_envelope_plot(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_parallel(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_parallel_coordinates(output_path, context.scenario.name, context.best_series, plot_config=plot_config)


def _render_radar(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_radar_chart(output_path, context.scenario.name, context.best_series, plot_config=plot_config)


def _render_convergence(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    objective_weights = {
        series.label: getattr(series.problem_config, "objective_weights", (1.0, 1.0, 1.0))
        for series in context.best_series
    }
    series_colors = {series.label: series.color for series in context.best_series}
    save_convergence_statistics(
        output_path,
        context.scenario.name,
        context.histories_by_label,
        objective_weights=objective_weights,
        series_colors=series_colors,
        plot_config=plot_config,
    )


def _render_distribution(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_distribution_violin(output_path, context.scenario.name, context.metrics_by_label, plot_config=plot_config)


def _render_run_statistics(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_run_statistics_panel(output_path, context.scenario.name, context.best_series, plot_config=plot_config)


def _render_dashboard(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_summary_dashboard(
        output_path,
        context.scenario,
        context.best_series,
        histories_by_label=context.histories_by_label,
        metrics_by_label=context.metrics_by_label,
        plot_config=plot_config,
    )


def _render_runtime_tradeoff(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    series_colors = {series.label: series.color for series in context.best_series}
    save_runtime_tradeoff(
        output_path, 
        context.scenario.name, 
        context.metrics_by_label, 
        series_colors=series_colors, 
        plot_config=plot_config
    )


def _render_decision_projection(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_decision_space_projection(output_path, context.scenario.name, context.best_series[0].episode, plot_config=plot_config)


def _render_operator_allocation(context: ScenarioFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    kemm_series = next((s for s in context.best_series if s.label == "KEMM"), None)
    if kemm_series:
        save_operator_allocation_history(output_path, context.scenario.name, kemm_series, plot_config=plot_config)


def _render_scenario_gallery(context: GlobalFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    save_scenario_gallery(output_path, context.scenario_map, plot_config=plot_config)


def _render_route_bundle_gallery(context: GlobalFigureContext, output_path: Path, plot_config: ShipPlotConfig) -> None:
    algorithm_styles = [
        (spec.key, spec.label, _algorithm_color(spec, plot_config, index))
        for index, spec in enumerate(context.algorithm_specs)
    ]
    save_route_bundle_gallery(
        output_path,
        context.scenario_map,
        context.aggregate_payload,
        algorithm_styles=algorithm_styles,
        plot_config=plot_config,
    )


SCENARIO_FIGURE_SPECS = (
    ScenarioFigureSpec("environment_overlay", "环境场叠加轨迹图", "_render_environment_overlay"),
    ScenarioFigureSpec("route_planning_panel", "复杂海域路线规划主图", "_render_route_planning_panel"),
    ScenarioFigureSpec("change_timeline", "重规划变化时间轴图", "_render_change_timeline"),
    ScenarioFigureSpec("snapshots", "动态避碰时序快照", "_render_snapshots"),
    ScenarioFigureSpec("spatiotemporal", "三维时空轨迹图", "_render_spatiotemporal"),
    ScenarioFigureSpec("control_timeseries", "控制与动力学时序", "_render_control_timeseries"),
    ScenarioFigureSpec("pareto3d", "3D Pareto 前沿图", "_render_pareto3d"),
    ScenarioFigureSpec("pareto_projection", "二维 Pareto 投影视图", "_render_pareto_projection"),
    ScenarioFigureSpec("risk_breakdown", "风险分解时间序列", "_render_risk_breakdown"),
    ScenarioFigureSpec("safety_envelope", "安全包络图", "_render_safety_envelope"),
    ScenarioFigureSpec("parallel", "Parallel coordinates", "_render_parallel"),
    ScenarioFigureSpec("radar", "Radar 图", "_render_radar"),
    ScenarioFigureSpec("convergence", "收敛统计图", "_render_convergence"),
    ScenarioFigureSpec("distribution", "分布 violin 图", "_render_distribution"),
    ScenarioFigureSpec("run_statistics", "重复运行统计图", "_render_run_statistics"),
    ScenarioFigureSpec("dashboard", "摘要 dashboard", "_render_dashboard"),
    ScenarioFigureSpec("runtime_tradeoff", "求解时间与控制能耗折中图", "_render_runtime_tradeoff"),
    ScenarioFigureSpec("decision_projection", "决策空间PCA聚类视图", "_render_decision_projection"),
    ScenarioFigureSpec("operator_allocation", "Contextual MAB 算子动态演化堆叠图", "_render_operator_allocation"),
)


GLOBAL_FIGURE_SPECS = (
    GlobalFigureSpec("scenario_gallery.png", "多场景环境总览图", "_render_scenario_gallery"),
    GlobalFigureSpec("route_bundle_gallery.png", "多场景最终轨迹带对比图", "_render_route_bundle_gallery"),
)


def generate_report(output_root: Path | None = None, plot_config: ShipPlotConfig | None = None) -> Path:
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
    scenario_keys: list[str] | None = None,
    algorithm_keys: Sequence[str] | None = None,
    verbose: bool = True,
    n_runs: int | None = None,
    max_workers: int = 1,
    render_figures: bool = True,
) -> Path:
    plot_config = plot_config or build_ship_plot_config(demo_config.plot_preset, appendix_plots=demo_config.appendix_plots)
    scenario_keys = scenario_keys or ["head_on", "crossing", "overtaking", "harbor_clutter"]
    if algorithm_keys:
        allowed_algorithms = {spec.key for spec in DEFAULT_ALGORITHM_SPECS}
        unknown_algorithms = [name for name in algorithm_keys if name not in allowed_algorithms]
        if unknown_algorithms:
            raise ValueError(f"Unknown ship algorithms: {unknown_algorithms}. Available: {sorted(allowed_algorithms)}")
        demo_config = replace(demo_config, report_algorithms=tuple(algorithm_keys))
    generator = ScenarioGenerator(config)
    algorithm_specs = _resolve_algorithm_specs(demo_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_root or Path("ship_simulation/outputs") / f"report_{timestamp}"
    raw_dir = root / "raw"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    algorithms = [spec.key for spec in algorithm_specs]
    run_count = int(n_runs or demo_config.n_runs)
    scenario_rows: list[dict[str, object]] = []
    aggregate_payload: dict[str, dict[str, list[PlanningEpisodeResult]]] = defaultdict(dict)
    step_payload: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    representative_payload: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    scenario_map: dict[str, object] = {key: generator.generate(key) for key in scenario_keys}
    t0 = time.time()
    task_specs = [
        {
            "scenario_key": scenario_key,
            "algorithm": algorithm,
            "run_index": run_index,
            "scenario_index": scenario_index,
            "algorithm_index": algorithm_index,
            "config": config,
            "demo_config": demo_config,
        }
        for scenario_index, scenario_key in enumerate(scenario_keys, start=1)
        for algorithm_index, algorithm in enumerate(algorithms)
        for run_index in range(run_count)
    ]
    task_total = len(task_specs)
    actual_workers = max(1, min(int(max_workers), task_total))
    effective_workers = actual_workers
    episode_lookup: dict[tuple[str, str, int], PlanningEpisodeResult] = {}

    if verbose:
        print("Ship simulation batch report started.", flush=True)
        print("This command exports figures to disk; it does not open interactive animation windows.", flush=True)
        print(f"Output directory: {root}", flush=True)
        print(f"Scenarios: {', '.join(scenario_keys)}", flush=True)
        print(f"Experiment profile: {config.experiment.profile_name}", flush=True)
        print(f"Algorithms: {', '.join(spec.label for spec in algorithm_specs)} | Runs per algorithm: {run_count}", flush=True)
        print(f"Episode workers: {actual_workers}", flush=True)
        print(f"Render figures: {'yes' if render_figures else 'no'}", flush=True)

    if actual_workers > 1:
        if verbose:
            print(f"Parallel episode execution started for {task_total} tasks.", flush=True)
        try:
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                future_map = {executor.submit(_ship_report_task, task): task for task in task_specs}
                for completed, future in enumerate(as_completed(future_map), start=1):
                    payload = future.result()
                    scenario_key = str(payload["scenario_key"])
                    algorithm = str(payload["algorithm"])
                    run_index = int(payload["run_index"])
                    episode = payload["episode"]
                    episode_lookup[(scenario_key, algorithm, run_index)] = episode
                    if verbose:
                        print(
                            f"[RUN {completed:>3d}/{task_total}] {scenario_key} | {_optimizer_display_name(algorithm):>10s} | "
                            f"run {run_index + 1}/{run_count}: fuel={episode.final_evaluation.objectives[0]:.2f}, "
                            f"time={episode.final_evaluation.objectives[1]:.2f}, risk={episode.final_evaluation.objectives[2]:.3f}",
                            flush=True,
                        )
        except (PermissionError, OSError) as exc:
            if verbose:
                print(f"[WARN] Parallel episode execution unavailable; falling back to serial: {exc}", flush=True)
            effective_workers = 1
            episode_lookup.clear()
            for completed, task in enumerate(task_specs, start=1):
                payload = _ship_report_task(task)
                scenario_key = str(payload["scenario_key"])
                algorithm = str(payload["algorithm"])
                run_index = int(payload["run_index"])
                episode = payload["episode"]
                episode_lookup[(scenario_key, algorithm, run_index)] = episode
                if verbose:
                    print(
                        f"[RUN {completed:>3d}/{task_total}] {scenario_key} | {_optimizer_display_name(algorithm):>10s} | "
                        f"run {run_index + 1}/{run_count}: fuel={episode.final_evaluation.objectives[0]:.2f}, "
                        f"time={episode.final_evaluation.objectives[1]:.2f}, risk={episode.final_evaluation.objectives[2]:.3f}",
                        flush=True,
                    )
    else:
        for completed, task in enumerate(task_specs, start=1):
            payload = _ship_report_task(task)
            scenario_key = str(payload["scenario_key"])
            algorithm = str(payload["algorithm"])
            run_index = int(payload["run_index"])
            episode = payload["episode"]
            episode_lookup[(scenario_key, algorithm, run_index)] = episode
            if verbose:
                print(
                    f"[RUN {completed:>3d}/{task_total}] {scenario_key} | {_optimizer_display_name(algorithm):>10s} | "
                    f"run {run_index + 1}/{run_count}: fuel={episode.final_evaluation.objectives[0]:.2f}, "
                    f"time={episode.final_evaluation.objectives[1]:.2f}, risk={episode.final_evaluation.objectives[2]:.3f}",
                    flush=True,
                )

    for scenario_index, scenario_key in enumerate(scenario_keys, start=1):
        scenario = scenario_map[scenario_key]
        if verbose:
            print(f"[{scenario_index}/{len(scenario_keys)}] Render scenario `{scenario_key}`", flush=True)
        aggregate_payload[scenario_key] = {}
        step_payload[scenario_key] = {}
        for algorithm in algorithms:
            episodes: list[PlanningEpisodeResult] = []
            for run_index in range(run_count):
                episode = episode_lookup[(scenario_key, algorithm, run_index)]
                episodes.append(episode)
                scenario_rows.append(_episode_row(scenario_key, algorithm, run_index, episode))
                step_payload[scenario_key].setdefault(algorithm, []).append(
                    {
                        "run": run_index,
                        "optimizer": _optimizer_display_name(algorithm),
                        "terminated_reason": episode.terminated_reason,
                        "pareto_size": int(len(episode.pareto_objectives)),
                        "knee_index": episode.knee_index,
                        "knee_objectives": episode.knee_objectives.tolist() if episode.knee_objectives is not None else None,
                        "analysis_metrics": dict(episode.analysis_metrics),
                        "snapshots": [
                            {
                                "time_s": snapshot.time_s,
                                "own_position": snapshot.own_position.tolist(),
                                "target_positions": [position.tolist() for position in snapshot.target_positions],
                                "risk": snapshot.risk,
                                "minimum_clearance": snapshot.minimum_clearance,
                            }
                            for snapshot in episode.snapshots
                        ],
                        "steps": [
                            {
                                "step_index": step.step_index,
                                "start_time": step.start_time,
                                "runtime": step.runtime_s,
                                "applied_changes": list(step.applied_changes),
                                "objectives": step.selected_evaluation.objectives.tolist(),
                                "terminal_distance": step.selected_evaluation.terminal_distance,
                                "risk_max": step.selected_evaluation.risk.max_risk,
                                "pareto_size": int(len(step.pareto_objectives)),
                            }
                            for step in episode.steps
                        ],
                    }
                )
            aggregate_payload[scenario_key][algorithm] = episodes

        best_series: list[ExperimentSeries] = []
        for algo_index, spec in enumerate(algorithm_specs):
            episodes = list(aggregate_payload[scenario_key].get(spec.key, []))
            if not episodes:
                continue
            representative_index = _representative_episode_index(episodes, config.objective_weights)
            representative = episodes[representative_index]
            representative_payload[scenario_key][spec.key] = {
                "label": spec.label,
                "run_index": int(representative_index),
                "selection_method": "median-objective-medoid",
                "reached_goal": bool(representative.final_evaluation.reached_goal),
                "objectives": representative.final_evaluation.objectives.tolist(),
                "success_rate": float(np.mean([1.0 if episode.final_evaluation.reached_goal else 0.0 for episode in episodes])),
            }
            best_series.append(
                ExperimentSeries(
                    label=spec.label,
                    episode=representative,
                    color=_algorithm_color(spec, plot_config, algo_index),
                    histories=[episode.convergence_history for episode in episodes],
                    distribution_metrics=[episode.analysis_metrics for episode in episodes],
                    repeated_statistics=_repeated_statistics(episodes),
                    problem_config=config,
                )
            )
        if render_figures:
            histories_by_label = {series.label: series.histories for series in best_series}
            metrics_by_label = {series.label: series.distribution_metrics for series in best_series}
            scenario_context = ScenarioFigureContext(
                scenario_key=scenario_key,
                scenario=scenario,
                best_series=best_series,
                histories_by_label=histories_by_label,
                metrics_by_label=metrics_by_label,
            )
            for spec in SCENARIO_FIGURE_SPECS:
                output_path = figures_dir / f"{scenario_key}_{spec.suffix}.png"
                renderer = globals()[spec.renderer_name]
                renderer(scenario_context, output_path, plot_config)

    aggregates = _aggregate_rows(scenario_rows)
    if render_figures:
        global_context = GlobalFigureContext(
            scenario_map=scenario_map,
            aggregate_payload=aggregate_payload,
            algorithm_specs=algorithm_specs,
        )
        for spec in GLOBAL_FIGURE_SPECS:
            output_path = figures_dir / spec.file_name
            renderer = globals()[spec.renderer_name]
            renderer(global_context, output_path, plot_config)
    _write_csv(raw_dir / "summary.csv", scenario_rows)
    _write_csv(raw_dir / "aggregate_summary.csv", aggregates)
    _write_json(raw_dir / "summary.json", {"runs": scenario_rows, "aggregates": aggregates})
    _write_json(raw_dir / "planning_steps.json", step_payload)
    _write_json(raw_dir / "representative_runs.json", representative_payload)
    if render_figures:
        _write_json(raw_dir / "figure_manifest.json", _figure_manifest(list(scenario_keys)))
    _write_json(
        raw_dir / "scenario_catalog.json",
        {
            key: _scenario_catalog_entry(scenario, getattr(config.scenario_generation, key))
            for key, scenario in scenario_map.items()
        },
    )
    _write_json(
        raw_dir / "report_metadata.json",
        {
            "scenario_keys": list(scenario_keys),
            "algorithms": [{"key": spec.key, "label": spec.label} for spec in algorithm_specs],
            "n_runs": run_count,
            "workers": effective_workers,
            "plot_preset": demo_config.plot_preset,
            "experiment_profile": config.experiment.profile_name,
            "experiment_enabled": bool(config.experiment.enabled),
            "experiment_config": asdict(config.experiment),
            "render_figures": bool(render_figures),
            "appendix_plots": bool(plot_config.appendix_plots),
            "interactive_figures": bool(plot_config.interactive_figures),
            "interactive_html": bool(plot_config.interactive_html),
            "elapsed_seconds": float(time.time() - t0),
        },
    )
    _write_markdown(reports_dir / "summary.md", aggregates)
    if render_figures:
        _write_figure_inventory(reports_dir / "figure_inventory.md", scenario_keys)

    # 附录图只保留旧柱状对比图，不再作为默认主图。
    if render_figures and plot_config.appendix_plots:
        aggregate_lookup = {(row["scenario_key"], row["optimizer"]): row for row in aggregates}
        comparison_pair = None
        if "kemm" in algorithms and "random" in algorithms:
            comparison_pair = ("KEMM", "Random")
        scenario_names = scenario_keys
        if comparison_pair is not None:
            left_label, right_label = comparison_pair
            left_rows = [aggregate_lookup[(key, left_label)] for key in scenario_keys]
            right_rows = [aggregate_lookup[(key, right_label)] for key in scenario_keys]
            left_objectives = np.asarray([[row["fuel_mean"], row["time_mean"], row["risk_mean"]] for row in left_rows], dtype=float)
            right_objectives = np.asarray([[row["fuel_mean"], row["time_mean"], row["risk_mean"]] for row in right_rows], dtype=float)
            left_risks = np.asarray([[row["max_risk_mean"], row["mean_risk_mean"], row["intrusion_time_mean"]] for row in left_rows], dtype=float)
            right_risks = np.asarray([[row["max_risk_mean"], row["mean_risk_mean"], row["intrusion_time_mean"]] for row in right_rows], dtype=float)
            save_normalized_objective_bars(figures_dir / "appendix_objective_bars.png", scenario_names, left_objectives, right_objectives, plot_config=plot_config)
            save_risk_bars(figures_dir / "appendix_risk_bars.png", scenario_names, left_risks, right_risks, plot_config=plot_config)

    if verbose:
        print(f"Finished in {time.time() - t0:.1f}s. Report directory: {root}", flush=True)
    return root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ship simulation report.")
    parser.add_argument("--quick", action="store_true", help="Run a lightweight smoke configuration.")
    parser.add_argument("--workers", type=int, default=None, help="Process workers for scenario/algorithm/run parallelism; default is auto for full mode and 1 for quick mode.")
    parser.add_argument("--summary-only", action="store_true", help="Skip figure rendering and export only raw results plus Markdown summary.")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Optional ship algorithms to run, e.g. kemm random.")
    parser.add_argument("--plot-preset", default="paper", help="Plot preset: default/paper/ieee/nature/thesis.")
    parser.add_argument("--experiment-profile", default="baseline", help="Ship experiment profile: baseline/drift/shock/recurring_harbor.")
    parser.add_argument("--science-style", default="", help="Comma-separated SciencePlots style tuple, e.g. science,ieee,no-latex.")
    parser.add_argument("--n-runs", type=int, default=None, help="Number of repeated runs per algorithm and scenario.")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Scenario keys to run.")
    parser.add_argument("--appendix-plots", action="store_true", help="Also export legacy appendix-style plots.")
    parser.add_argument("--interactive-figures", action="store_true", help="Also export interactive matplotlib figure bundles (.fig.pickle).")
    parser.add_argument("--interactive-html", action="store_true", help="Also export interactive HTML for supported 3D plots.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = build_default_config()
    apply_experiment_profile(config, args.experiment_profile)
    demo_config = _build_quick_demo_config() if args.quick else build_default_demo_config()
    demo_config.plot_preset = args.plot_preset
    demo_config.appendix_plots = bool(args.appendix_plots)
    style_overrides = {}
    science_tuple = _science_style_tuple(args.science_style)
    if science_tuple is not None:
        style_overrides["use_scienceplots"] = True
        style_overrides["science_styles"] = science_tuple
    plot_config = build_ship_plot_config(
        args.plot_preset,
        style_overrides=style_overrides,
        appendix_plots=args.appendix_plots,
        interactive_figures=args.interactive_figures,
        interactive_html=args.interactive_html,
    )
    workers = args.workers
    if workers is None:
        workers = 1 if args.quick else _recommended_worker_count()
    generate_report_with_config(
        config=config,
        demo_config=demo_config,
        plot_config=plot_config,
        scenario_keys=args.scenarios,
        algorithm_keys=args.algorithms,
        verbose=True,
        n_runs=args.n_runs,
        max_workers=workers,
        render_figures=not args.summary_only,
    )


if __name__ == "__main__":
    main()
