"""批量运行 ship episode，并导出论文风格报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from reporting_config import ShipPlotConfig, build_ship_plot_config, plot_style_context
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

EPISODE_CACHE_SCHEMA_VERSION = "ship-episode-cache-v2"
STAT_TEST_ALPHA = 0.05
DEFAULT_ROBUSTNESS_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_ROBUSTNESS_SCENARIOS: tuple[str, ...] = ("crossing", "overtaking", "harbor_clutter")
MATCHED_ALGORITHM_BASE: dict[str, str] = {
    "nsga_style_matched": "nsga_style",
    "random_matched": "random",
}
STRICT_COMPARABLE_COMPANIONS: dict[str, str] = {
    "nsga_style": "nsga_style_matched",
    "random": "random_matched",
}

try:  # pragma: no cover - optional dependency in local runtime
    from scipy import stats as scipy_stats

    HAS_SCIPY_STATS = True
except Exception:  # pragma: no cover
    scipy_stats = None
    HAS_SCIPY_STATS = False


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
    demo.scenario_profiles.active_profile_name = "legacy_uniform"
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
    demo.render_workers = 1
    return demo


def _recommended_worker_count(limit: int = 4) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return max(1, min(limit, cpu_count - 1))


def _json_ready(value):
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _episode_cache_signature(
    *,
    scenario_key: str,
    algorithm: str,
    run_index: int,
    config,
    demo_config: DemoConfig,
) -> str:
    payload = {
        "schema_version": EPISODE_CACHE_SCHEMA_VERSION,
        "scenario_key": str(scenario_key),
        "algorithm": str(algorithm),
        "run_index": int(run_index),
        "config": _json_ready(asdict(config)),
        "demo": {
            "random_search_samples": int(demo_config.random_search_samples),
            "random_search_seed": int(demo_config.random_search_seed),
            "evolutionary_baseline_pop_size": int(demo_config.evolutionary_baseline_pop_size),
            "evolutionary_baseline_generations": int(demo_config.evolutionary_baseline_generations),
            "kemm": _json_ready(asdict(demo_config.kemm)),
            "episode": _json_ready(asdict(demo_config.episode)),
            "scenario_profiles": _json_ready(asdict(demo_config.scenario_profiles)),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _episode_cache_path(cache_dir: Path, scenario_key: str, algorithm: str, run_index: int, cache_signature: str) -> Path:
    return cache_dir / str(scenario_key) / f"{algorithm}_run{int(run_index) + 1}_{cache_signature[:12]}.pkl"


def _save_episode_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_episode_cache(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("schema_version")) != EPISODE_CACHE_SCHEMA_VERSION:
        return None
    return payload


def _render_ship_figure_task(task: dict[str, object]) -> dict[str, object]:
    renderer = globals()[str(task["renderer_name"])]
    output_path = Path(str(task["output_path"]))
    renderer(task["context"], output_path, task["plot_config"])
    return {
        "kind": str(task["kind"]),
        "name": str(task["name"]),
        "output_path": str(output_path),
    }


def _ship_report_task(task: dict[str, object]) -> dict[str, object]:
    scenario_key = str(task["scenario_key"])
    requested_algorithm = str(task["algorithm"])
    run_index = int(task["run_index"])
    scenario_index = int(task["scenario_index"])
    algorithm_index = int(task["algorithm_index"])
    config = task["config"]
    demo_config = task["demo_config"]
    cache_enabled = bool(task.get("episode_cache_enabled", False))
    cache_dir = Path(str(task.get("episode_cache_dir") or ""))
    cache_signature = str(task.get("cache_signature") or "")
    cache_path = _episode_cache_path(cache_dir, scenario_key, requested_algorithm, run_index, cache_signature) if cache_enabled else None

    if cache_enabled and cache_path is not None:
        cached = _load_episode_cache(cache_path)
        if cached is not None and str(cached.get("cache_signature")) == cache_signature:
            return {
                "scenario_key": scenario_key,
                "algorithm": requested_algorithm,
                "run_index": run_index,
                "scenario_index": scenario_index,
                "algorithm_index": algorithm_index,
                "episode": cached["episode"],
                "cache_hit": True,
            }

    scenario = ScenarioGenerator(config).generate(scenario_key)
    run_demo = replace(
        demo_config,
        random_search_seed=demo_config.random_search_seed + run_index,
        kemm=replace(demo_config.kemm, seed=demo_config.kemm.seed + run_index),
    )
    solver_algorithm = requested_algorithm
    if requested_algorithm in MATCHED_ALGORITHM_BASE:
        run_demo = _strict_comparable_demo_config(run_demo)
        solver_algorithm = MATCHED_ALGORITHM_BASE[requested_algorithm]
    planner = RollingHorizonPlanner(scenario=scenario, config=config, demo_config=run_demo)
    episode = planner.run(optimizer_name=solver_algorithm)
    if requested_algorithm in MATCHED_ALGORITHM_BASE:
        episode.analysis_metrics["strict_comparable_budget"] = 1.0
        episode.analysis_metrics["requested_algorithm"] = requested_algorithm
        episode.analysis_metrics["solver_algorithm"] = solver_algorithm
    if cache_enabled and cache_path is not None:
        _save_episode_cache(
            cache_path,
            {
                "schema_version": EPISODE_CACHE_SCHEMA_VERSION,
                "cache_signature": cache_signature,
                "scenario_key": scenario_key,
                "algorithm": requested_algorithm,
                "run_index": run_index,
                "episode": episode,
            },
        )
    return {
        "scenario_key": scenario_key,
        "algorithm": requested_algorithm,
        "run_index": run_index,
        "scenario_index": scenario_index,
        "algorithm_index": algorithm_index,
        "episode": episode,
        "cache_hit": False,
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
    AlgorithmReportSpec("nsga_style_matched", "NSGA-style (Budget-Matched)", "third_algo_color"),
    AlgorithmReportSpec("random_matched", "Random (Budget-Matched)", "baseline_color"),
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


def _ci_bounds(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return mean, mean
    stderr = std / math.sqrt(arr.size)
    if HAS_SCIPY_STATS:
        alpha = max(1e-6, min(1.0 - confidence, 0.999999))
        critical = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, arr.size - 1))
    else:  # pragma: no cover - fallback when scipy is unavailable
        critical = 1.96
    delta = critical * stderr
    return mean - delta, mean + delta


def _budget_matched_profile(profile):
    eval_budget = max(int(profile.kemm_pop_size) * int(profile.kemm_generations), 1)
    return replace(
        profile,
        random_search_samples=eval_budget,
        evolutionary_baseline_pop_size=int(profile.kemm_pop_size),
        evolutionary_baseline_generations=int(profile.kemm_generations),
    )


def _strict_comparable_demo_config(demo_config: DemoConfig) -> DemoConfig:
    profiles = demo_config.scenario_profiles
    matched_profiles = replace(
        profiles,
        legacy_uniform={key: _budget_matched_profile(profile) for key, profile in profiles.legacy_uniform.items()},
        full_tuned={key: _budget_matched_profile(profile) for key, profile in profiles.full_tuned.items()},
    )
    budget = max(int(demo_config.kemm.pop_size) * int(demo_config.kemm.generations), 1)
    return replace(
        demo_config,
        random_search_samples=budget,
        evolutionary_baseline_pop_size=int(demo_config.kemm.pop_size),
        evolutionary_baseline_generations=int(demo_config.kemm.generations),
        scenario_profiles=matched_profiles,
    )


def _inject_strict_comparable_specs(algorithm_specs: Sequence[AlgorithmReportSpec]) -> list[AlgorithmReportSpec]:
    registry = {spec.key: spec for spec in DEFAULT_ALGORITHM_SPECS}
    selected = list(algorithm_specs)
    seen = {spec.key for spec in selected}
    for base_key, companion_key in STRICT_COMPARABLE_COMPANIONS.items():
        if base_key in seen and companion_key not in seen and companion_key in registry:
            selected.append(registry[companion_key])
            seen.add(companion_key)
    return selected


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
        success_values = np.asarray([1.0 if item["reached_goal"] else 0.0 for item in items], dtype=float)
        success_ci_low, success_ci_high = _ci_bounds(success_values)
        payload = {
            "scenario_key": scenario_key,
            "optimizer": optimizer,
            "n_runs": len(items),
            "success_rate": float(np.mean(success_values)),
            "success_rate_std": float(np.std(success_values)),
            "success_rate_ci_low": float(success_ci_low),
            "success_rate_ci_high": float(success_ci_high),
        }
        for key in numeric_keys:
            values = np.asarray([float(item[key]) for item in items], dtype=float)
            payload[f"{key}_mean"] = float(np.mean(values))
            payload[f"{key}_std"] = float(np.std(values))
            ci_low, ci_high = _ci_bounds(values)
            payload[f"{key}_ci_low"] = float(ci_low)
            payload[f"{key}_ci_high"] = float(ci_high)
        aggregates.append(payload)
    return aggregates


def _two_sample_test(left: np.ndarray, right: np.ndarray, *, alpha: float = STAT_TEST_ALPHA) -> dict[str, object]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return {"test": "insufficient_data", "p_value": float("nan"), "significant": False}
    if left.size == 1 and right.size == 1:
        return {"test": "insufficient_data", "p_value": float("nan"), "significant": False}
    if not HAS_SCIPY_STATS:
        return {"test": "scipy_unavailable", "p_value": float("nan"), "significant": False}

    use_welch = False
    if left.size >= 8 and right.size >= 8:
        try:
            left_normal_p = float(scipy_stats.shapiro(left).pvalue)
            right_normal_p = float(scipy_stats.shapiro(right).pvalue)
            use_welch = left_normal_p > alpha and right_normal_p > alpha
        except Exception:
            use_welch = False

    try:
        if use_welch:
            p_value = float(scipy_stats.ttest_ind(left, right, equal_var=False, nan_policy="omit").pvalue)
            test_name = "welch_t"
        else:
            try:
                p_value = float(scipy_stats.mannwhitneyu(left, right, alternative="two-sided", method="auto").pvalue)
            except TypeError:  # scipy<1.7 compatibility
                p_value = float(scipy_stats.mannwhitneyu(left, right, alternative="two-sided").pvalue)
            test_name = "mann_whitney_u"
    except Exception:
        p_value = float("nan")
        test_name = "test_failed"

    significant = bool(np.isfinite(p_value) and p_value < alpha)
    return {"test": test_name, "p_value": p_value, "significant": significant}


def _build_statistical_tests(
    rows: Sequence[dict[str, object]],
    *,
    alpha: float = STAT_TEST_ALPHA,
    anchor_optimizer: str = "KEMM",
) -> list[dict[str, object]]:
    metrics = {
        "fuel": "minimize",
        "time": "minimize",
        "risk": "minimize",
        "runtime": "minimize",
        "minimum_clearance": "maximize",
        "success_rate": "maximize",
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario_key"]), str(row["optimizer"]))].append(row)

    tests: list[dict[str, object]] = []
    scenarios = sorted({str(row["scenario_key"]) for row in rows})
    for scenario_key in scenarios:
        anchor_rows = grouped.get((scenario_key, anchor_optimizer), [])
        if not anchor_rows:
            continue
        scenario_optimizers = sorted(
            {
                optimizer
                for (scenario, optimizer) in grouped.keys()
                if scenario == scenario_key and optimizer != anchor_optimizer
            }
        )
        for optimizer_name in scenario_optimizers:
            optimizer_rows = grouped.get((scenario_key, optimizer_name), [])
            if not optimizer_rows:
                continue
            for metric_key, direction in metrics.items():
                if metric_key == "success_rate":
                    left_values = np.asarray([1.0 if bool(item["reached_goal"]) else 0.0 for item in anchor_rows], dtype=float)
                    right_values = np.asarray([1.0 if bool(item["reached_goal"]) else 0.0 for item in optimizer_rows], dtype=float)
                else:
                    left_values = np.asarray([float(item[metric_key]) for item in anchor_rows], dtype=float)
                    right_values = np.asarray([float(item[metric_key]) for item in optimizer_rows], dtype=float)
                if left_values.size == 0 or right_values.size == 0:
                    continue
                left_mean = float(np.mean(left_values))
                right_mean = float(np.mean(right_values))
                left_ci_low, left_ci_high = _ci_bounds(left_values)
                right_ci_low, right_ci_high = _ci_bounds(right_values)
                denom = max(abs(right_mean), 1e-9)
                if direction == "minimize":
                    improvement_pct = float((right_mean - left_mean) / denom * 100.0)
                else:
                    improvement_pct = float((left_mean - right_mean) / denom * 100.0)
                test_result = _two_sample_test(left_values, right_values, alpha=alpha)
                tests.append(
                    {
                        "scenario_key": scenario_key,
                        "metric": metric_key,
                        "direction": direction,
                        "anchor_optimizer": anchor_optimizer,
                        "comparison_optimizer": str(optimizer_name),
                        "anchor_mean": left_mean,
                        "anchor_ci_low": float(left_ci_low),
                        "anchor_ci_high": float(left_ci_high),
                        "comparison_mean": right_mean,
                        "comparison_ci_low": float(right_ci_low),
                        "comparison_ci_high": float(right_ci_high),
                        "delta_anchor_minus_comparison": float(left_mean - right_mean),
                        "improvement_pct_vs_comparison": improvement_pct,
                        "test": str(test_result["test"]),
                        "p_value": float(test_result["p_value"]),
                        "significant": bool(test_result["significant"]),
                        "alpha": float(alpha),
                        "n_anchor": int(left_values.size),
                        "n_comparison": int(right_values.size),
                    }
                )
    return tests


def _write_statistical_markdown(output_path: Path, tests: Sequence[dict[str, object]], *, alpha: float = STAT_TEST_ALPHA) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical Significance Summary",
        "",
        f"> Pairwise tests compare `KEMM` against each baseline per scenario and metric. Significance threshold: alpha={alpha:.2f}.",
        "",
        "| Scenario | Metric | Baseline | KEMM Mean (95% CI) | Baseline Mean (95% CI) | Improvement | Test | p-value | Significant |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in tests:
        p_value = row["p_value"]
        p_text = "nan" if not np.isfinite(p_value) else f"{p_value:.4f}"
        lines.append(
            "| {scenario_key} | {metric} | {comparison_optimizer} | "
            "{anchor_mean:.4f} [{anchor_ci_low:.4f}, {anchor_ci_high:.4f}] | "
            "{comparison_mean:.4f} [{comparison_ci_low:.4f}, {comparison_ci_high:.4f}] | "
            "{improvement_pct_vs_comparison:+.2f}% | {test} | "
            f"{p_text} | "
            "{significant} |".format(**row)
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _robustness_config_for_level(config, scenario_key: str, level: float):
    level = max(0.0, float(level))
    tuning = getattr(config.scenario_generation, scenario_key)
    tuned_tuning = replace(
        tuning,
        difficulty_scale=float(tuning.difficulty_scale * (1.0 + 0.35 * level)),
        geometry_jitter_m=float(tuning.geometry_jitter_m + 220.0 * level),
        traffic_heading_jitter_deg=float(tuning.traffic_heading_jitter_deg + 14.0 * level),
        current_direction_jitter_deg=float(tuning.current_direction_jitter_deg + 16.0 * level),
        target_speed_scale=float(tuning.target_speed_scale * (1.0 + 0.12 * level)),
        scalar_amplitude_scale=float(tuning.scalar_amplitude_scale * (1.0 + 0.30 * level)),
        vector_speed_scale=float(tuning.vector_speed_scale * (1.0 + 0.25 * level)),
    )
    tuned_generation = replace(config.scenario_generation, **{scenario_key: tuned_tuning})
    tuned_environment = replace(
        config.environment,
        current_speed=float(config.environment.current_speed * (1.0 + 0.28 * level)),
        wind_speed=float(config.environment.wind_speed * (1.0 + 0.22 * level)),
    )
    return replace(config, scenario_generation=tuned_generation, environment=tuned_environment)


def _run_robustness_sweep(
    *,
    config,
    demo_config: DemoConfig,
    scenario_keys: Sequence[str],
    algorithms: Sequence[str],
    run_count: int,
    levels: Sequence[float],
    verbose: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    t0 = time.time()
    total_tasks = max(1, len(scenario_keys) * len(levels) * len(algorithms) * run_count)
    completed = 0
    for scenario_key in scenario_keys:
        for level in levels:
            tuned_config = _robustness_config_for_level(config, scenario_key, level)
            for algorithm in algorithms:
                for run_index in range(run_count):
                    payload = _ship_report_task(
                        {
                            "scenario_key": scenario_key,
                            "algorithm": algorithm,
                            "run_index": run_index,
                            "scenario_index": 0,
                            "algorithm_index": 0,
                            "config": tuned_config,
                            "demo_config": demo_config,
                            "episode_cache_enabled": False,
                            "episode_cache_dir": "",
                            "cache_signature": "",
                        }
                    )
                    episode = payload["episode"]
                    completed += 1
                    if verbose:
                        print(
                            f"[ROB {completed:>3d}/{total_tasks}] {scenario_key} | level={float(level):.2f} | "
                            f"{_optimizer_display_name(algorithm):>24s} | run {run_index + 1}/{run_count} | "
                            f"success={bool(episode.final_evaluation.reached_goal)}",
                            flush=True,
                        )
                    rows.append(
                        {
                            "scenario_key": str(scenario_key),
                            "disturbance_level": float(level),
                            "optimizer": _optimizer_display_name(algorithm),
                            "run": int(run_index),
                            "reached_goal": bool(episode.final_evaluation.reached_goal),
                            "fuel": float(episode.final_evaluation.objectives[0]),
                            "time": float(episode.final_evaluation.objectives[1]),
                            "risk": float(episode.final_evaluation.objectives[2]),
                            "runtime": float(episode.analysis_metrics.get("runtime", 0.0)),
                        }
                    )

    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario_key"]), str(row["optimizer"]), float(row["disturbance_level"]))].append(row)

    curves: list[dict[str, object]] = []
    for (scenario_key, optimizer, level), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        success = np.asarray([1.0 if bool(item["reached_goal"]) else 0.0 for item in items], dtype=float)
        ci_low, ci_high = _ci_bounds(success)
        curves.append(
            {
                "scenario_key": scenario_key,
                "optimizer": optimizer,
                "disturbance_level": float(level),
                "n_runs": int(len(items)),
                "success_rate": float(np.mean(success)),
                "success_rate_std": float(np.std(success)),
                "success_rate_ci_low": float(ci_low),
                "success_rate_ci_high": float(ci_high),
            }
        )
    return rows, curves, float(time.time() - t0)


def _save_robustness_curve(
    output_path: Path,
    curves: Sequence[dict[str, object]],
    *,
    algorithm_specs: Sequence[AlgorithmReportSpec],
    plot_config: ShipPlotConfig,
) -> None:
    if not curves:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({str(row["scenario_key"]) for row in curves})
    levels = np.asarray(sorted({float(row["disturbance_level"]) for row in curves}), dtype=float)
    if not scenarios:
        return
    ncols = min(3, len(scenarios))
    nrows = int(math.ceil(len(scenarios) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    color_by_label = {
        spec.label: _algorithm_color(spec, plot_config, index)
        for index, spec in enumerate(algorithm_specs)
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in curves:
        grouped[(str(row["scenario_key"]), str(row["optimizer"]))].append(dict(row))

    for idx, scenario_key in enumerate(scenarios):
        ax = axes[idx // ncols][idx % ncols]
        for spec in algorithm_specs:
            label = spec.label
            points = sorted(
                grouped.get((scenario_key, label), []),
                key=lambda item: float(item["disturbance_level"]),
            )
            if not points:
                continue
            x = np.asarray([float(item["disturbance_level"]) for item in points], dtype=float)
            y = np.asarray([float(item["success_rate"]) for item in points], dtype=float)
            y_std = np.asarray([float(item["success_rate_std"]) for item in points], dtype=float)
            color = color_by_label.get(label, plot_config.baseline_color)
            ax.plot(x, y, marker="o", linewidth=plot_config.style.line_width, color=color, label=label)
            ax.fill_between(
                x,
                np.clip(y - y_std, 0.0, 1.0),
                np.clip(y + y_std, 0.0, 1.0),
                color=color,
                alpha=plot_config.style.band_alpha,
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(float(np.min(levels)) - 0.02, float(np.max(levels)) + 0.02)
        ax.set_title(f"{scenario_key}: success vs disturbance")
        ax.set_xlabel("Disturbance level")
        ax.set_ylabel("Success rate")
        ax.grid(True, alpha=plot_config.style.grid_alpha)
        ax.legend(loc="best", frameon=True, framealpha=plot_config.legend_alpha)

    for idx in range(len(scenarios), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=plot_config.style.dpi, bbox_inches="tight")
    plt.close(fig)


def _write_robustness_markdown(output_path: Path, curves: Sequence[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Robustness Sweep",
        "",
        "> Disturbance sweep reports success-rate curves under progressively stronger environment and scenario perturbations.",
        "",
        "| Scenario | Optimizer | Disturbance | Success Rate | 95% CI | Runs |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in curves:
        lines.append(
            "| {scenario_key} | {optimizer} | {disturbance_level:.2f} | {success_rate:.2%} | "
            "[{success_rate_ci_low:.2%}, {success_rate_ci_high:.2%}] | {n_runs} |".format(**row)
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _write_markdown(
    output_path: Path,
    aggregates: list[dict[str, object]],
    *,
    active_solve_profile_name: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ship Simulation Report",
        "",
        "> Tables below aggregate all repeated runs. Trajectory-like figures use representative runs recorded in `raw/representative_runs.json`.",
        "",
    ]
    if active_solve_profile_name:
        lines.extend(
            [
                f"> Active solve profile: `{active_solve_profile_name}`. `full_tuned` is the default complete-run profile; `legacy_uniform` remains available for regression comparison.",
                "",
            ]
        )
    lines.extend(
        [
        "| Scenario | Optimizer | Fuel | Time | Risk | Clearance | Ship Dist | Runtime | Knee (F/T/R) | Success Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
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
    strict_comparable: bool = False,
    robustness_sweep: bool = False,
    robustness_levels: Sequence[float] | None = None,
    robustness_scenarios: Sequence[str] | None = None,
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
    if strict_comparable:
        algorithm_specs = _inject_strict_comparable_specs(algorithm_specs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_root or Path("ship_simulation/outputs") / f"report_{timestamp}"
    raw_dir = root / "raw"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    episode_cache_dir = raw_dir / "episode_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if demo_config.episode_cache_enabled:
        episode_cache_dir.mkdir(parents=True, exist_ok=True)

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
            "episode_cache_enabled": bool(demo_config.episode_cache_enabled),
            "episode_cache_dir": episode_cache_dir,
            "cache_signature": _episode_cache_signature(
                scenario_key=scenario_key,
                algorithm=algorithm,
                run_index=run_index,
                config=config,
                demo_config=demo_config,
            ),
        }
        for scenario_index, scenario_key in enumerate(scenario_keys, start=1)
        for algorithm_index, algorithm in enumerate(algorithms)
        for run_index in range(run_count)
    ]
    task_total = len(task_specs)
    actual_workers = max(1, min(int(max_workers), task_total))
    effective_workers = actual_workers
    episode_lookup: dict[tuple[str, str, int], PlanningEpisodeResult] = {}
    episode_cache_hits = 0
    episode_cache_misses = 0
    episode_compute_seconds = 0.0
    figure_render_seconds = 0.0
    robustness_compute_seconds = 0.0
    effective_render_workers = 0

    if verbose:
        print("Ship simulation batch report started.", flush=True)
        print("This command exports figures to disk; it does not open interactive animation windows.", flush=True)
        print(f"Output directory: {root}", flush=True)
        print(f"Scenarios: {', '.join(scenario_keys)}", flush=True)
        print(f"Experiment profile: {config.experiment.profile_name}", flush=True)
        print(f"Scenario solve profile: {demo_config.scenario_profiles.active_profile_name}", flush=True)
        print(f"Strict comparable budgets: {'enabled' if strict_comparable else 'disabled'}", flush=True)
        print(f"Algorithms: {', '.join(spec.label for spec in algorithm_specs)} | Runs per algorithm: {run_count}", flush=True)
        print(f"Episode workers: {actual_workers}", flush=True)
        print(f"Render workers: {max(1, int(demo_config.render_workers)) if render_figures else 0}", flush=True)
        print(f"Episode cache: {'enabled' if demo_config.episode_cache_enabled else 'disabled'}", flush=True)
        print(f"Render figures: {'yes' if render_figures else 'no'}", flush=True)

    compute_t0 = time.time()
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
                    if bool(payload.get("cache_hit")):
                        episode_cache_hits += 1
                    else:
                        episode_cache_misses += 1
                    if verbose:
                        cache_label = "cache" if bool(payload.get("cache_hit")) else "solve"
                        print(
                            f"[RUN {completed:>3d}/{task_total}] {scenario_key} | {_optimizer_display_name(algorithm):>10s} | "
                            f"run {run_index + 1}/{run_count}: fuel={episode.final_evaluation.objectives[0]:.2f}, "
                            f"time={episode.final_evaluation.objectives[1]:.2f}, risk={episode.final_evaluation.objectives[2]:.3f} | {cache_label}",
                            flush=True,
                        )
        except (PermissionError, OSError) as exc:
            if verbose:
                print(f"[WARN] Parallel episode execution unavailable; falling back to serial: {exc}", flush=True)
            effective_workers = 1
            episode_lookup.clear()
            episode_cache_hits = 0
            episode_cache_misses = 0
            for completed, task in enumerate(task_specs, start=1):
                payload = _ship_report_task(task)
                scenario_key = str(payload["scenario_key"])
                algorithm = str(payload["algorithm"])
                run_index = int(payload["run_index"])
                episode = payload["episode"]
                episode_lookup[(scenario_key, algorithm, run_index)] = episode
                if bool(payload.get("cache_hit")):
                    episode_cache_hits += 1
                else:
                    episode_cache_misses += 1
                if verbose:
                    cache_label = "cache" if bool(payload.get("cache_hit")) else "solve"
                    print(
                        f"[RUN {completed:>3d}/{task_total}] {scenario_key} | {_optimizer_display_name(algorithm):>10s} | "
                        f"run {run_index + 1}/{run_count}: fuel={episode.final_evaluation.objectives[0]:.2f}, "
                        f"time={episode.final_evaluation.objectives[1]:.2f}, risk={episode.final_evaluation.objectives[2]:.3f} | {cache_label}",
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
            if bool(payload.get("cache_hit")):
                episode_cache_hits += 1
            else:
                episode_cache_misses += 1
            if verbose:
                cache_label = "cache" if bool(payload.get("cache_hit")) else "solve"
                print(
                    f"[RUN {completed:>3d}/{task_total}] {scenario_key} | {_optimizer_display_name(algorithm):>10s} | "
                    f"run {run_index + 1}/{run_count}: fuel={episode.final_evaluation.objectives[0]:.2f}, "
                    f"time={episode.final_evaluation.objectives[1]:.2f}, risk={episode.final_evaluation.objectives[2]:.3f} | {cache_label}",
                    flush=True,
                )
    episode_compute_seconds = time.time() - compute_t0

    render_tasks: list[dict[str, object]] = []
    for scenario_key in scenario_keys:
        scenario = scenario_map[scenario_key]
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
            representative_problem_config = episodes[0].problem_config or config
            representative_weights = getattr(representative_problem_config, "objective_weights", config.objective_weights)
            representative_index = _representative_episode_index(episodes, representative_weights)
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
                    problem_config=representative.problem_config or config,
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
                render_tasks.append(
                    {
                        "kind": "scenario",
                        "name": f"{scenario_key}_{spec.suffix}",
                        "renderer_name": spec.renderer_name,
                        "context": scenario_context,
                        "output_path": str(figures_dir / f"{scenario_key}_{spec.suffix}.png"),
                        "plot_config": plot_config,
                    }
                )

    aggregates = _aggregate_rows(scenario_rows)
    statistical_tests = _build_statistical_tests(scenario_rows, alpha=STAT_TEST_ALPHA)

    robustness_rows: list[dict[str, object]] = []
    robustness_curves: list[dict[str, object]] = []
    normalized_levels = sorted({float(level) for level in (robustness_levels or DEFAULT_ROBUSTNESS_LEVELS)})
    normalized_robustness_scenarios = [key for key in (robustness_scenarios or DEFAULT_ROBUSTNESS_SCENARIOS) if key in scenario_keys]
    if robustness_sweep and normalized_robustness_scenarios:
        if verbose:
            print(
                f"Robustness sweep started: scenarios={normalized_robustness_scenarios}, levels={normalized_levels}.",
                flush=True,
            )
        robustness_rows, robustness_curves, robustness_compute_seconds = _run_robustness_sweep(
            config=config,
            demo_config=demo_config,
            scenario_keys=normalized_robustness_scenarios,
            algorithms=algorithms,
            run_count=run_count,
            levels=normalized_levels,
            verbose=verbose,
        )
        if render_figures:
            with plot_style_context(plot_config.style):
                _save_robustness_curve(
                    figures_dir / "robustness_success_curve.png",
                    robustness_curves,
                    algorithm_specs=algorithm_specs,
                    plot_config=plot_config,
                )

    if render_figures:
        global_context = GlobalFigureContext(
            scenario_map=scenario_map,
            aggregate_payload=aggregate_payload,
            algorithm_specs=algorithm_specs,
        )
        for spec in GLOBAL_FIGURE_SPECS:
            render_tasks.append(
                {
                    "kind": "global",
                    "name": spec.file_name,
                    "renderer_name": spec.renderer_name,
                    "context": global_context,
                    "output_path": str(figures_dir / spec.file_name),
                    "plot_config": plot_config,
                }
            )

    if render_figures:
        render_t0 = time.time()
        effective_render_workers = max(1, min(int(demo_config.render_workers), len(render_tasks))) if render_tasks else 1
        if verbose:
            print(f"Figure render phase started for {len(render_tasks)} tasks.", flush=True)
        if effective_render_workers > 1 and len(render_tasks) > 1:
            try:
                with ProcessPoolExecutor(max_workers=effective_render_workers) as executor:
                    future_map = {executor.submit(_render_ship_figure_task, task): task for task in render_tasks}
                    for completed, future in enumerate(as_completed(future_map), start=1):
                        payload = future.result()
                        if verbose:
                            print(
                                f"[FIG {completed:>3d}/{len(render_tasks)}] {payload['name']}",
                                flush=True,
                            )
            except (PermissionError, OSError) as exc:
                if verbose:
                    print(f"[WARN] Parallel figure rendering unavailable; falling back to serial: {exc}", flush=True)
                effective_render_workers = 1
                for completed, task in enumerate(render_tasks, start=1):
                    _render_ship_figure_task(task)
                    if verbose:
                        print(f"[FIG {completed:>3d}/{len(render_tasks)}] {task['name']}", flush=True)
        else:
            for completed, task in enumerate(render_tasks, start=1):
                _render_ship_figure_task(task)
                if verbose:
                    print(f"[FIG {completed:>3d}/{len(render_tasks)}] {task['name']}", flush=True)
        figure_render_seconds = time.time() - render_t0

    _write_csv(raw_dir / "summary.csv", scenario_rows)
    _write_csv(raw_dir / "aggregate_summary.csv", aggregates)
    _write_csv(raw_dir / "statistical_tests.csv", statistical_tests)
    _write_json(raw_dir / "summary.json", {"runs": scenario_rows, "aggregates": aggregates})
    _write_json(raw_dir / "statistical_tests.json", {"alpha": STAT_TEST_ALPHA, "tests": statistical_tests})
    _write_json(raw_dir / "planning_steps.json", step_payload)
    _write_json(raw_dir / "representative_runs.json", representative_payload)
    if robustness_sweep and robustness_rows:
        _write_csv(raw_dir / "robustness_runs.csv", robustness_rows)
        _write_csv(raw_dir / "robustness_curve.csv", robustness_curves)
        _write_json(
            raw_dir / "robustness_summary.json",
            {
                "levels": normalized_levels,
                "scenarios": normalized_robustness_scenarios,
                "runs": robustness_rows,
                "curves": robustness_curves,
            },
        )
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
            "render_workers": effective_render_workers,
            "plot_preset": demo_config.plot_preset,
            "experiment_profile": config.experiment.profile_name,
            "experiment_enabled": bool(config.experiment.enabled),
            "experiment_config": asdict(config.experiment),
            "scenario_solve_profile": demo_config.scenario_profiles.active_profile_name,
            "scenario_profiles": asdict(demo_config.scenario_profiles),
            "render_figures": bool(render_figures),
            "appendix_plots": bool(plot_config.appendix_plots),
            "interactive_figures": bool(plot_config.interactive_figures),
            "interactive_html": bool(plot_config.interactive_html),
            "strict_comparable": bool(strict_comparable),
            "episode_cache_enabled": bool(demo_config.episode_cache_enabled),
            "episode_compute_seconds": float(episode_compute_seconds),
            "figure_render_seconds": float(figure_render_seconds),
            "robustness_sweep_enabled": bool(robustness_sweep),
            "robustness_scenarios": list(normalized_robustness_scenarios),
            "robustness_levels": list(normalized_levels),
            "robustness_compute_seconds": float(robustness_compute_seconds),
            "statistical_alpha": float(STAT_TEST_ALPHA),
            "statistical_test_count": int(len(statistical_tests)),
            "episode_cache_hits": int(episode_cache_hits),
            "episode_cache_misses": int(episode_cache_misses),
            "elapsed_seconds": float(time.time() - t0),
        },
    )
    _write_markdown(
        reports_dir / "summary.md",
        aggregates,
        active_solve_profile_name=demo_config.scenario_profiles.active_profile_name,
    )
    _write_statistical_markdown(
        reports_dir / "statistical_significance.md",
        statistical_tests,
        alpha=STAT_TEST_ALPHA,
    )
    if robustness_sweep and robustness_curves:
        _write_robustness_markdown(
            reports_dir / "robustness_sweep.md",
            robustness_curves,
        )
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
    parser.add_argument("--strict-comparable", action="store_true", help="Append budget-matched baseline groups (same budget and replanning cadence as KEMM).")
    parser.add_argument("--robustness-sweep", action="store_true", help="Run disturbance-level robustness sweep and export success-rate curves.")
    parser.add_argument("--robustness-levels", default="0,0.25,0.5,0.75,1.0", help="Comma-separated disturbance levels used by --robustness-sweep.")
    parser.add_argument("--robustness-scenarios", nargs="*", default=None, help="Scenario keys for robustness sweep. Default: crossing overtaking harbor_clutter.")
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
    robustness_levels = [
        float(item.strip())
        for item in str(args.robustness_levels).split(",")
        if item.strip()
    ]
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
        strict_comparable=bool(args.strict_comparable),
        robustness_sweep=bool(args.robustness_sweep),
        robustness_levels=robustness_levels,
        robustness_scenarios=args.robustness_scenarios,
    )


if __name__ == "__main__":
    main()
