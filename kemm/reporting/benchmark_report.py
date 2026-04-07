"""Benchmark reporting helpers.

This module converts in-memory benchmark results into a small report bundle
with machine-readable tables and a concise Markdown summary.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _float(value: Any) -> float:
    """Convert scalars to builtin float for JSON/CSV serialization."""

    return float(np.asarray(value).item())


def _collect_metric_rows(results: Dict[str, Dict[str, Dict[str, List[float]]]], cfg: Any) -> List[Dict[str, object]]:
    """Flatten benchmark metric statistics into row dictionaries."""

    rows: List[Dict[str, object]] = []
    for algorithm, problem_map in results.items():
        for problem in cfg.PROBLEMS:
            metrics = problem_map[problem]
            rows.append(
                {
                    "algorithm": algorithm,
                    "problem": problem,
                    "migd_mean": _float(np.mean(metrics["MIGD"])),
                    "migd_std": _float(np.std(metrics["MIGD"])),
                    "sp_mean": _float(np.mean(metrics["SP"])),
                    "sp_std": _float(np.std(metrics["SP"])),
                    "ms_mean": _float(np.mean(metrics["MS"])),
                    "ms_std": _float(np.std(metrics["MS"])),
                    "time_mean": _float(np.mean(metrics["TIME"])),
                    "time_std": _float(np.std(metrics["TIME"])),
                }
            )
    return rows


def _collect_setting_metric_rows(
    setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
    cfg: Any,
) -> List[Dict[str, object]]:
    """Flatten setting-aware benchmark statistics into row dictionaries."""

    rows: List[Dict[str, object]] = []
    for setting_key, results in setting_results.items():
        nt_str, tau_t_str = str(setting_key).split(",", maxsplit=1)
        nt = int(nt_str)
        tau_t = int(tau_t_str)
        for algorithm, problem_map in results.items():
            for problem in cfg.PROBLEMS:
                metrics = problem_map[problem]
                rows.append(
                    {
                        "setting": str(setting_key),
                        "nt": nt,
                        "tau_t": tau_t,
                        "algorithm": algorithm,
                        "problem": problem,
                        "migd_mean": _float(np.mean(metrics["MIGD"])),
                        "migd_std": _float(np.std(metrics["MIGD"])),
                    }
                )
    return rows


def _collect_ablation_delta_rows(
    ablation_setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
    cfg: Any,
    baseline_algorithm: str = "KEMM-Full",
) -> List[Dict[str, object]]:
    """Summarize ablation deltas relative to the full KEMM variant."""

    rows: List[Dict[str, object]] = []
    for setting_key, results in ablation_setting_results.items():
        nt_str, tau_t_str = str(setting_key).split(",", maxsplit=1)
        nt = int(nt_str)
        tau_t = int(tau_t_str)
        algorithms = list(results.keys())
        baseline_name = baseline_algorithm if baseline_algorithm in results else algorithms[0]
        baseline_by_problem = {
            problem: _float(np.mean(results[baseline_name][problem]["MIGD"]))
            for problem in cfg.PROBLEMS
        }
        for algorithm, problem_map in results.items():
            problem_means = np.asarray(
                [_float(np.mean(problem_map[problem]["MIGD"])) for problem in cfg.PROBLEMS],
                dtype=float,
            )
            baseline_values = np.asarray([baseline_by_problem[problem] for problem in cfg.PROBLEMS], dtype=float)
            delta_pct = np.zeros(len(cfg.PROBLEMS), dtype=float)
            safe = np.abs(baseline_values) > 1e-12
            delta_pct[safe] = (problem_means[safe] - baseline_values[safe]) / baseline_values[safe] * 100.0
            rows.append(
                {
                    "setting": str(setting_key),
                    "nt": nt,
                    "tau_t": tau_t,
                    "algorithm": algorithm,
                    "baseline_algorithm": baseline_name,
                    "migd_mean": _float(np.mean(problem_means)),
                    "migd_std_across_problems": _float(np.std(problem_means)),
                    "delta_pct_mean": _float(np.mean(delta_pct)),
                    "delta_pct_std": _float(np.std(delta_pct)),
                }
            )
    return rows


def _compute_average_ranks(results: Dict[str, Dict[str, Dict[str, List[float]]]], cfg: Any) -> Dict[str, float]:
    """Compute average rank over MIGD/SP/MS across all configured problems."""

    algorithms = list(results.keys())
    all_ranks = {algorithm: [] for algorithm in algorithms}
    for metric in ("MIGD", "SP", "MS"):
        direction = "smaller" if metric != "MS" else "larger"
        for problem in cfg.PROBLEMS:
            means = np.array([np.mean(results[algorithm][problem][metric]) for algorithm in algorithms], dtype=float)
            if direction == "larger":
                means = -means
            ranks = np.argsort(np.argsort(means)) + 1
            for index, algorithm in enumerate(algorithms):
                all_ranks[algorithm].append(int(ranks[index]))
    return {algorithm: _float(np.mean(rank_values)) for algorithm, rank_values in all_ranks.items()}


def _collect_rank_rows(results: Dict[str, Dict[str, Dict[str, List[float]]]], cfg: Any) -> List[Dict[str, object]]:
    """Flatten average rank statistics into row dictionaries."""

    avg_ranks = _compute_average_ranks(results, cfg)
    ordered = sorted(avg_ranks.items(), key=lambda item: item[1])
    return [
        {"rank": index + 1, "algorithm": algorithm, "avg_rank": avg_rank}
        for index, (algorithm, avg_rank) in enumerate(ordered)
    ]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(
    path: Path,
    metric_rows: List[Dict[str, object]],
    rank_rows: List[Dict[str, object]],
    cfg: Any,
    ablation_rows: List[Dict[str, object]] | None = None,
    paper_table_rows: List[Dict[str, object]] | None = None,
    ablation_delta_rows: List[Dict[str, object]] | None = None,
) -> None:
    """Write a compact Markdown summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Report",
        "",
        "## Configuration",
        "",
        f"- Population size: `{cfg.POP_SIZE}`",
        f"- Variables: `{cfg.N_VAR}`",
        f"- Objectives: `{cfg.N_OBJ}`",
        f"- Runs per problem: `{cfg.N_RUNS}`",
        f"- Problems: `{', '.join(cfg.PROBLEMS)}`",
        f"- Algorithms: `{', '.join(cfg.ALGORITHMS.keys())}`",
        "",
        "## Average Rank",
        "",
        "| Rank | Algorithm | AvgRank |",
        "| --- | --- | --- |",
    ]
    for row in rank_rows:
        lines.append(f"| {row['rank']} | {row['algorithm']} | {row['avg_rank']:.4f} |")

    lines.extend(
        [
            "",
            "## Metric Summary",
            "",
            "| Algorithm | Problem | MIGD | SP | MS | Time |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    str(row["problem"]),
                    f"{row['migd_mean']:.4f} +/- {row['migd_std']:.4f}",
                    f"{row['sp_mean']:.4f} +/- {row['sp_std']:.4f}",
                    f"{row['ms_mean']:.4f} +/- {row['ms_std']:.4f}",
                    f"{row['time_mean']:.4f} +/- {row['time_std']:.4f}",
                ]
            )
            + " |"
        )

    if ablation_rows:
        lines.extend(
            [
                "",
                "## Ablation Summary",
                "",
                "| Variant | Problem | MIGD | SP | MS | Time |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in ablation_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["algorithm"]),
                        str(row["problem"]),
                        f"{row['migd_mean']:.4f} +/- {row['migd_std']:.4f}",
                        f"{row['sp_mean']:.4f} +/- {row['sp_std']:.4f}",
                        f"{row['ms_mean']:.4f} +/- {row['ms_std']:.4f}",
                        f"{row['time_mean']:.4f} +/- {row['time_std']:.4f}",
                    ]
                )
                + " |"
            )

    if ablation_delta_rows:
        lines.extend(
            [
                "",
                "## Ablation Delta Summary",
                "",
                "| Setting | Variant | Mean MIGD | Delta vs Full (%) |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in ablation_delta_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["setting"]),
                        str(row["algorithm"]),
                        f"{row['migd_mean']:.4f}",
                        f"{row['delta_pct_mean']:+.2f} +/- {row['delta_pct_std']:.2f}",
                    ]
                )
                + " |"
            )

    if paper_table_rows:
        lines.extend(
            [
                "",
                "## Paper-Style MIGD Table",
                "",
            ]
        )
        algorithms = list(cfg.ALGORITHMS.keys())
        grouped: dict[tuple[str, str], dict[str, Dict[str, object]]] = {}
        for row in paper_table_rows:
            grouped.setdefault((str(row["problem"]), str(row["setting"])), {})[str(row["algorithm"])] = row

        lines.append("| Problem | n_t,τ_t | " + " | ".join(algorithms) + " |")
        lines.append("| --- | --- | " + " | ".join(["---"] * len(algorithms)) + " |")
        settings = [f"{nt},{tau_t}" for nt, tau_t in getattr(cfg, "SETTINGS", [])]
        for problem in cfg.PROBLEMS:
            for row_index, setting_key in enumerate(settings):
                cells = [problem if row_index == 0 else "", setting_key]
                algo_rows = grouped.get((problem, setting_key), {})
                if algo_rows:
                    best_algo = min(algorithms, key=lambda algo: algo_rows[algo]["migd_mean"])
                else:
                    best_algo = None
                for algorithm in algorithms:
                    row = algo_rows.get(algorithm)
                    if row is None:
                        cells.append("N/A")
                        continue
                    text = f"{row['migd_mean']:.4f} +/- {row['migd_std']:.4f}"
                    if algorithm == best_algo:
                        text = f"**{text}**"
                    cells.append(text)
                lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report_paths(output_root: Path | None = None, prefix: str = "report") -> Path:
    """Create a timestamped benchmark report root directory."""

    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("benchmark_outputs") / f"{prefix}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def export_benchmark_report(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    cfg: Any,
    output_root: Path | None = None,
    ablation_results: Dict[str, Dict[str, Dict[str, List[float]]]] | None = None,
    setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] | None = None,
    ablation_setting_results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] | None = None,
) -> Path:
    """Export benchmark results to raw tables and a Markdown report."""

    root = build_report_paths(output_root=output_root)
    raw_dir = root / "raw"
    reports_dir = root / "reports"

    metric_rows = _collect_metric_rows(results, cfg)
    rank_rows = _collect_rank_rows(results, cfg)
    ablation_rows = _collect_metric_rows(ablation_results, cfg) if ablation_results else []
    paper_table_rows = _collect_setting_metric_rows(setting_results, cfg) if setting_results else []
    ablation_setting_rows = _collect_setting_metric_rows(ablation_setting_results, cfg) if ablation_setting_results else []
    ablation_delta_rows = _collect_ablation_delta_rows(ablation_setting_results, cfg) if ablation_setting_results else []

    _write_csv(raw_dir / "metrics.csv", metric_rows)
    _write_csv(raw_dir / "ranks.csv", rank_rows)
    if ablation_rows:
        _write_csv(raw_dir / "ablation_metrics.csv", ablation_rows)
    if paper_table_rows:
        _write_csv(raw_dir / "paper_table_metrics.csv", paper_table_rows)
    if ablation_setting_rows:
        _write_csv(raw_dir / "ablation_setting_metrics.csv", ablation_setting_rows)
    if ablation_delta_rows:
        _write_csv(raw_dir / "ablation_delta_metrics.csv", ablation_delta_rows)
    _write_json(
        raw_dir / "summary.json",
        {
            "config": {
                "pop_size": cfg.POP_SIZE,
                "n_var": cfg.N_VAR,
                "n_obj": cfg.N_OBJ,
                "n_runs": cfg.N_RUNS,
                "problems": list(cfg.PROBLEMS),
                "algorithms": list(cfg.ALGORITHMS.keys()),
            },
            "metrics": metric_rows,
            "ranks": rank_rows,
            "ablation_metrics": ablation_rows,
            "paper_table_metrics": paper_table_rows,
            "ablation_setting_metrics": ablation_setting_rows,
            "ablation_delta_metrics": ablation_delta_rows,
        },
    )
    _write_markdown(
        reports_dir / "summary.md",
        metric_rows,
        rank_rows,
        cfg,
        ablation_rows=ablation_rows,
        paper_table_rows=paper_table_rows,
        ablation_delta_rows=ablation_delta_rows,
    )
    return root


__all__ = [
    "build_report_paths",
    "export_benchmark_report",
]
