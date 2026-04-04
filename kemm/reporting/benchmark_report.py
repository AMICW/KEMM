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
                    f"{row['migd_mean']:.4f} ± {row['migd_std']:.4f}",
                    f"{row['sp_mean']:.4f} ± {row['sp_std']:.4f}",
                    f"{row['ms_mean']:.4f} ± {row['ms_std']:.4f}",
                    f"{row['time_mean']:.4f} ± {row['time_std']:.4f}",
                ]
            )
            + " |"
        )

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
) -> Path:
    """Export benchmark results to raw tables and a Markdown report."""

    root = build_report_paths(output_root=output_root)
    raw_dir = root / "raw"
    reports_dir = root / "reports"

    metric_rows = _collect_metric_rows(results, cfg)
    rank_rows = _collect_rank_rows(results, cfg)

    _write_csv(raw_dir / "metrics.csv", metric_rows)
    _write_csv(raw_dir / "ranks.csv", rank_rows)
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
        },
    )
    _write_markdown(reports_dir / "summary.md", metric_rows, rank_rows, cfg)
    return root


__all__ = [
    "build_report_paths",
    "export_benchmark_report",
]
