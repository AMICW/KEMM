"""Structured benchmark reporting plots.

这个模块的目标不是塞满所有论文图，而是提供一套稳定、可扩展、面向结构化数据的
benchmark 可视化接口。以后你要改 KEMM 内部结构时，优先保持 payload 结构稳定，
图表层就不需要跟着重写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from kemm.core.types import KEMMChangeDiagnostics
from reporting_config import BenchmarkPlotConfig, plot_style_context

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.stats import ranksums

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class BenchmarkFigurePayload:
    """benchmark 图表生成所需的结构化输入。"""

    results: Dict[str, Dict[str, Dict[str, List[float]]]]
    problems: Sequence[str]
    igd_curves: Dict[str, Dict[str, List[List[float]]]] | None = None
    diagnostics: Dict[str, Dict[str, List[List[KEMMChangeDiagnostics]]]] | None = None
    ablation_results: Dict[str, Dict[str, Dict[str, List[float]]]] | None = None
    plot_config: BenchmarkPlotConfig = field(default_factory=BenchmarkPlotConfig)


def _resolve_plot_config(plot_config: BenchmarkPlotConfig | None) -> BenchmarkPlotConfig:
    return plot_config or BenchmarkPlotConfig()


def _color_for(algo: str, plot_config: BenchmarkPlotConfig | None = None) -> str:
    cfg = _resolve_plot_config(plot_config)
    return cfg.colors.get(algo, "#7f8c8d")


def _marker_for(algo: str, plot_config: BenchmarkPlotConfig | None = None) -> str:
    cfg = _resolve_plot_config(plot_config)
    return cfg.markers.get(algo, "o")



def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def _mean_metric(results, algo: str, problem: str, metric: str) -> float:
    return float(np.mean(results[algo][problem][metric]))



def _std_metric(results, algo: str, problem: str, metric: str) -> float:
    return float(np.std(results[algo][problem][metric]))



def _average_rank(results, problems: Sequence[str]) -> dict[str, float]:
    algos = list(results.keys())
    all_ranks = {algo: [] for algo in algos}
    for metric in ["MIGD", "SP", "MS"]:
        direction = "smaller" if metric != "MS" else "larger"
        for problem in problems:
            means = np.array([_mean_metric(results, algo, problem, metric) for algo in algos], dtype=float)
            if direction == "larger":
                means = -means
            ranks = np.argsort(np.argsort(means)) + 1
            for idx, algo in enumerate(algos):
                all_ranks[algo].append(float(ranks[idx]))
    return {algo: float(np.mean(rank_values)) for algo, rank_values in all_ranks.items()}



def _mean_curve(curves: List[List[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.zeros(0), np.zeros(0)
    max_len = max(len(curve) for curve in curves)
    padded = [curve + [curve[-1]] * (max_len - len(curve)) for curve in curves if curve]
    if not padded:
        return np.zeros(0), np.zeros(0)
    array = np.asarray(padded, dtype=float)
    return np.mean(array, axis=0), np.std(array, axis=0)



def _mean_operator_ratio_series(
    diagnostics: Dict[str, Dict[str, List[List[KEMMChangeDiagnostics]]]] | None,
    algo_name: str = "KEMM",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not diagnostics or algo_name not in diagnostics:
        return np.zeros(0), {}

    records: dict[int, dict[str, list[float]]] = {}
    for problem_runs in diagnostics[algo_name].values():
        for run_history in problem_runs:
            for item in run_history:
                bucket = records.setdefault(item.time_step, {
                    "memory": [],
                    "prediction": [],
                    "transfer": [],
                    "reinit": [],
                })
                for key, value in item.operator_ratios.items():
                    if key in bucket:
                        bucket[key].append(float(value))

    if not records:
        return np.zeros(0), {}

    steps = np.array(sorted(records.keys()), dtype=float)
    series = {
        key: np.array([np.mean(records[int(step)][key]) for step in steps], dtype=float)
        for key in ["memory", "prediction", "transfer", "reinit"]
    }
    return steps, series



def _mean_scalar_diagnostic_series(
    diagnostics: Dict[str, Dict[str, List[List[KEMMChangeDiagnostics]]]] | None,
    field: str,
    algo_name: str = "KEMM",
) -> tuple[np.ndarray, np.ndarray]:
    if not diagnostics or algo_name not in diagnostics:
        return np.zeros(0), np.zeros(0)

    buckets: dict[int, list[float]] = {}
    for problem_runs in diagnostics[algo_name].values():
        for run_history in problem_runs:
            for item in run_history:
                buckets.setdefault(item.time_step, []).append(float(getattr(item, field)))

    if not buckets:
        return np.zeros(0), np.zeros(0)

    steps = np.array(sorted(buckets.keys()), dtype=float)
    values = np.array([np.mean(buckets[int(step)]) for step in steps], dtype=float)
    return steps, values


class PerformanceComparisonPlots:
    """benchmark 整体性能对比图。"""

    @staticmethod
    def plot_migd_main_table(results, problems, save_path="fig_A1_migd_bars.png", plot_config: BenchmarkPlotConfig | None = None):
        PerformanceComparisonPlots.plot_metric_bars_grid(results, problems, "MIGD", save_path, plot_config=plot_config)

    @staticmethod
    def plot_three_metrics_grid(results, problems, save_path="fig_A2_three_metrics.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        metrics = [("MIGD", "lower"), ("SP", "lower"), ("MS", "higher")]
        algos = list(results.keys())
        fig, axes = plt.subplots(
            3,
            len(problems),
            figsize=(cfg.metric_panel_width * len(problems), cfg.metrics_grid_height * 3),
            squeeze=False,
        )
        for row, (metric, direction) in enumerate(metrics):
            for col, problem in enumerate(problems):
                ax = axes[row][col]
                means = np.array([_mean_metric(results, algo, problem, metric) for algo in algos])
                best = int(np.argmin(means) if direction == "lower" else np.argmax(means))
                colors = [_color_for(algo, cfg) for algo in algos]
                colors[best] = cfg.best_outline_color
                ax.bar(range(len(algos)), means, color=colors, alpha=cfg.style.bar_alpha, edgecolor="black", linewidth=0.4)
                if row == 0:
                    ax.set_title(problem, fontweight="bold")
                if col == 0:
                    arrow = "↓" if direction == "lower" else "↑"
                    ax.set_ylabel(f"{metric} {arrow}")
                ax.set_xticks(range(len(algos)))
                ax.set_xticklabels(algos if row == 2 else [], rotation=35)
                ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_metric_bars_grid(results, problems, metric, save_path, direction: str | None = None, plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        direction = direction or ("higher" if metric == "MS" else "lower")
        algos = list(results.keys())
        n_cols = min(3, len(problems))
        n_rows = (len(problems) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(cfg.metric_panel_width * n_cols, cfg.metrics_grid_height * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()
        for idx, problem in enumerate(problems):
            ax = axes[idx]
            means = np.array([_mean_metric(results, algo, problem, metric) for algo in algos], dtype=float)
            stds = np.array([_std_metric(results, algo, problem, metric) for algo in algos], dtype=float)
            best = int(np.argmin(means) if direction == "lower" else np.argmax(means))
            colors = [_color_for(algo, cfg) for algo in algos]
            bars = ax.bar(range(len(algos)), means, yerr=stds, color=colors, alpha=cfg.style.bar_alpha, capsize=3)
            bars[best].set_edgecolor(cfg.best_outline_color)
            bars[best].set_linewidth(2.5)
            ax.set_title(problem, fontweight="bold")
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(algos, rotation=35)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25, axis="y")
        for idx in range(len(problems), len(axes)):
            axes[idx].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_heatmap_normalized(results, problems, save_path="fig_A4_heatmap.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        algos = list(results.keys())
        matrix = np.array([[_mean_metric(results, algo, problem, "MIGD") for problem in problems] for algo in algos], dtype=float)
        cmin = matrix.min(axis=0, keepdims=True)
        cmax = matrix.max(axis=0, keepdims=True)
        normalized = (matrix - cmin) / (cmax - cmin + 1e-12)
        fig, ax = plt.subplots(figsize=(max(9, len(problems) * 1.8), max(4, len(algos) * 0.7)))
        image = ax.imshow(normalized, cmap=cfg.heatmap_cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(problems)))
        ax.set_xticklabels(problems)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        for i in range(len(algos)):
            for j in range(len(problems)):
                color = "white" if normalized[i, j] > 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(image, ax=ax, label="Normalized MIGD (0=best)")
        ax.set_title("Benchmark MIGD Heatmap")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_rank_bar(results, problems, save_path="fig_A5_rank_bar.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        avg_ranks = _average_rank(results, problems)
        ordered = sorted(avg_ranks.items(), key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=(cfg.rank_bar_width, cfg.rank_bar_height))
        labels = [item[0] for item in ordered]
        values = [item[1] for item in ordered]
        colors = [cfg.highlight_color if label == "KEMM" else _color_for(label, cfg) for label in labels]
        ax.barh(range(len(labels)), values, color=colors, alpha=cfg.style.bar_alpha)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title("Average Rank Comparison")
        ax.grid(True, alpha=0.25, axis="x")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class ProcessAnalysisPlots:
    """benchmark 过程分析图。"""

    @staticmethod
    def plot_igd_convergence(igd_curves, problems, save_path="fig_B1_igd_convergence.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL or not igd_curves:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        algos = list(igd_curves.keys())
        n_cols = min(3, len(problems))
        n_rows = (len(problems) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(cfg.metric_panel_width * n_cols, cfg.metrics_grid_height * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()
        for idx, problem in enumerate(problems):
            ax = axes[idx]
            for algo in algos:
                mean_curve, std_curve = _mean_curve(igd_curves.get(algo, {}).get(problem, []))
                if mean_curve.size == 0:
                    continue
                x = np.arange(1, len(mean_curve) + 1)
                ax.plot(
                    x,
                    mean_curve,
                    label=algo,
                    color=_color_for(algo, cfg),
                    marker=_marker_for(algo, cfg),
                    linewidth=cfg.style.emphasis_line_width if algo == "KEMM" else cfg.style.line_width,
                )
                ax.fill_between(
                    x,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=_color_for(algo, cfg),
                    alpha=cfg.style.band_alpha,
                )
            ax.set_title(problem, fontweight="bold")
            ax.set_xlabel("Change Index")
            ax.set_ylabel("IGD")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        for idx in range(len(problems), len(axes)):
            axes[idx].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_operator_ratio_history(diagnostics, save_path="fig_B2_operator_ratios.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        steps, series = _mean_operator_ratio_series(diagnostics)
        if steps.size == 0 or not series:
            return
        _ensure_parent(Path(save_path))
        fig, ax = plt.subplots(figsize=(cfg.dashboard_width * 0.7, cfg.dashboard_height * 0.6))
        for name, values in series.items():
            ax.plot(steps, values, label=name, linewidth=2.0)
        ax.set_title("KEMM Operator Allocation")
        ax.set_xlabel("Change Step")
        ax.set_ylabel("Mean Ratio")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_response_quality_history(diagnostics, save_path="fig_B3_response_quality.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        steps, values = _mean_scalar_diagnostic_series(diagnostics, "response_quality")
        if steps.size == 0:
            return
        _ensure_parent(Path(save_path))
        fig, ax = plt.subplots(figsize=(cfg.dashboard_width * 0.7, cfg.dashboard_height * 0.6))
        ax.plot(steps, values, color=cfg.highlight_color, linewidth=cfg.style.emphasis_line_width, marker="o")
        ax.set_title("KEMM Response Quality Proxy")
        ax.set_xlabel("Change Step")
        ax.set_ylabel("Response Quality")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_prediction_confidence(diagnostics, save_path="fig_B4_prediction_confidence.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        steps, values = _mean_scalar_diagnostic_series(diagnostics, "prediction_confidence")
        if steps.size == 0:
            return
        _ensure_parent(Path(save_path))
        fig, ax = plt.subplots(figsize=(cfg.dashboard_width * 0.7, cfg.dashboard_height * 0.6))
        ax.plot(steps, values, color=_color_for("PPS", cfg), linewidth=cfg.style.emphasis_line_width, marker="s")
        ax.set_title("KEMM Prediction Confidence")
        ax.set_xlabel("Change Step")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class StatisticalAnalysisPlots:
    """benchmark 统计图。"""

    @staticmethod
    def plot_wilcoxon_heatmap(
        results,
        problems,
        save_path="fig_C1_wilcoxon_heatmap.png",
        our_algo="KEMM",
        plot_config: BenchmarkPlotConfig | None = None,
    ):
        if not HAS_MPL or not HAS_SCIPY:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        algos = [algo for algo in results.keys() if algo != our_algo]
        matrix = np.zeros((len(algos), len(problems)), dtype=float)
        for i, algo in enumerate(algos):
            for j, problem in enumerate(problems):
                ours = results[our_algo][problem]["MIGD"]
                others = results[algo][problem]["MIGD"]
                if len(ours) < 2 or len(others) < 2:
                    matrix[i, j] = 1.0
                    continue
                _, p_value = ranksums(ours, others)
                matrix[i, j] = p_value
        fig, ax = plt.subplots(figsize=(max(8, len(problems) * 1.4), max(4, len(algos) * 0.7)))
        image = ax.imshow(matrix, cmap=cfg.significance_cmap, aspect="auto", vmin=0, vmax=0.1)
        ax.set_xticks(range(len(problems)))
        ax.set_xticklabels(problems)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        for i in range(len(algos)):
            for j in range(len(problems)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="p-value")
        ax.set_title("Wilcoxon Rank-Sum Test: KEMM vs Others")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_pairwise_win_matrix(results, problems, save_path="fig_C2_pairwise_wins.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        algos = list(results.keys())
        wins = np.zeros((len(algos), len(algos)), dtype=float)
        for i, row_algo in enumerate(algos):
            for j, col_algo in enumerate(algos):
                if i == j:
                    continue
                count = 0
                for problem in problems:
                    row_val = _mean_metric(results, row_algo, problem, "MIGD")
                    col_val = _mean_metric(results, col_algo, problem, "MIGD")
                    if row_val < col_val:
                        count += 1
                wins[i, j] = count
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(wins, cmap=cfg.pairwise_cmap, aspect="auto")
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=35)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        for i in range(len(algos)):
            for j in range(len(algos)):
                ax.text(j, i, int(wins[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="wins on MIGD")
        ax.set_title("Pairwise Win Matrix")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class AlgorithmMechanismPlots:
    """KEMM 机制解释图。"""

    @staticmethod
    def plot_change_diagnostics_dashboard(diagnostics, save_path="fig_D1_change_dashboard.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL:
            return
        cfg = _resolve_plot_config(plot_config)
        steps_ratio, ratio_series = _mean_operator_ratio_series(diagnostics)
        steps_quality, quality = _mean_scalar_diagnostic_series(diagnostics, "response_quality")
        steps_conf, confidence = _mean_scalar_diagnostic_series(diagnostics, "prediction_confidence")
        steps_change, change_mag = _mean_scalar_diagnostic_series(diagnostics, "change_magnitude")
        if steps_ratio.size == 0:
            return

        _ensure_parent(Path(save_path))
        fig, axes = plt.subplots(2, 2, figsize=(cfg.dashboard_width, cfg.dashboard_height))
        ax_ratio, ax_quality, ax_conf, ax_change = axes.flatten()

        for name, values in ratio_series.items():
            ax_ratio.plot(steps_ratio, values, label=name, linewidth=2.0)
        ax_ratio.set_title("Operator Ratios")
        ax_ratio.set_xlabel("Change Step")
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.grid(True, alpha=0.25)
        ax_ratio.legend(loc="best")

        if steps_quality.size > 0:
            ax_quality.plot(steps_quality, quality, color=cfg.highlight_color, marker="o")
        ax_quality.set_title("Response Quality")
        ax_quality.set_xlabel("Change Step")
        ax_quality.grid(True, alpha=0.25)

        if steps_conf.size > 0:
            ax_conf.plot(steps_conf, confidence, color=_color_for("PPS", cfg), marker="s")
        ax_conf.set_title("Prediction Confidence")
        ax_conf.set_xlabel("Change Step")
        ax_conf.set_ylim(0.0, 1.05)
        ax_conf.grid(True, alpha=0.25)

        if steps_change.size > 0:
            ax_change.plot(steps_change, change_mag, color=_color_for("KF", cfg), marker="^")
        ax_change.set_title("Change Magnitude")
        ax_change.set_xlabel("Change Step")
        ax_change.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class AblationStudyPlots:
    """消融实验图。"""

    @staticmethod
    def plot_ablation_comparison(ablation_results, problems, save_path="fig_E1_ablation.png", plot_config: BenchmarkPlotConfig | None = None):
        if not HAS_MPL or not ablation_results:
            return
        cfg = _resolve_plot_config(plot_config)
        _ensure_parent(Path(save_path))
        variants = list(ablation_results.keys())
        means = [float(np.mean([np.mean(ablation_results[variant][problem]["MIGD"]) for problem in problems])) for variant in variants]
        fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.2), 4.5))
        colors = [cfg.highlight_color if "KEMM" in variant else "#95a5a6" for variant in variants]
        ax.bar(range(len(variants)), means, color=colors, alpha=cfg.style.bar_alpha)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=30)
        ax.set_ylabel("Mean MIGD")
        ax.set_title("Ablation Comparison")
        ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)



def generate_all_figures(
    results=None,
    problems=None,
    igd_curves=None,
    diagnostics=None,
    ablation_results=None,
    kemm_algo_instance=None,
    output_prefix="benchmark",
    payload: BenchmarkFigurePayload | None = None,
    plot_config: BenchmarkPlotConfig | None = None,
):
    """统一生成 benchmark 图表。

    支持两种调用方式：

    1. 直接传 `payload`
    2. 用旧风格参数逐个传入
    """

    if not HAS_MPL:
        return

    if payload is None:
        if diagnostics is None and kemm_algo_instance is not None:
            # 仅使用公开的结构化变化诊断，不再访问私有属性。
            diagnostics = {"KEMM": {"adhoc": [list(getattr(kemm_algo_instance, "change_diagnostics_history", []))]}}
        payload = BenchmarkFigurePayload(
            results=results or {},
            problems=problems or [],
            igd_curves=igd_curves,
            diagnostics=diagnostics,
            ablation_results=ablation_results,
            plot_config=plot_config or BenchmarkPlotConfig(),
        )
    elif plot_config is not None:
        payload.plot_config = plot_config

    prefix = Path(output_prefix)
    cfg = payload.plot_config
    with plot_style_context(cfg.style):
        PerformanceComparisonPlots.plot_migd_main_table(
            payload.results,
            payload.problems,
            save_path=f"{prefix}_migd_bar.png",
            plot_config=cfg,
        )
        PerformanceComparisonPlots.plot_three_metrics_grid(
            payload.results,
            payload.problems,
            save_path=f"{prefix}_metrics_grid.png",
            plot_config=cfg,
        )
        PerformanceComparisonPlots.plot_heatmap_normalized(
            payload.results,
            payload.problems,
            save_path=f"{prefix}_heatmap.png",
            plot_config=cfg,
        )
        PerformanceComparisonPlots.plot_rank_bar(
            payload.results,
            payload.problems,
            save_path=f"{prefix}_rank_bar.png",
            plot_config=cfg,
        )

        if payload.igd_curves:
            ProcessAnalysisPlots.plot_igd_convergence(
                payload.igd_curves,
                payload.problems,
                save_path=f"{prefix}_igd_time.png",
                plot_config=cfg,
            )
        if payload.diagnostics:
            ProcessAnalysisPlots.plot_operator_ratio_history(
                payload.diagnostics,
                save_path=f"{prefix}_operator_ratios.png",
                plot_config=cfg,
            )
            ProcessAnalysisPlots.plot_response_quality_history(
                payload.diagnostics,
                save_path=f"{prefix}_response_quality.png",
                plot_config=cfg,
            )
            ProcessAnalysisPlots.plot_prediction_confidence(
                payload.diagnostics,
                save_path=f"{prefix}_prediction_confidence.png",
                plot_config=cfg,
            )
            AlgorithmMechanismPlots.plot_change_diagnostics_dashboard(
                payload.diagnostics,
                save_path=f"{prefix}_change_dashboard.png",
                plot_config=cfg,
            )

        StatisticalAnalysisPlots.plot_wilcoxon_heatmap(
            payload.results,
            payload.problems,
            save_path=f"{prefix}_wilcoxon.png",
            plot_config=cfg,
        )
        StatisticalAnalysisPlots.plot_pairwise_win_matrix(
            payload.results,
            payload.problems,
            save_path=f"{prefix}_pairwise_wins.png",
            plot_config=cfg,
        )

        if payload.ablation_results:
            AblationStudyPlots.plot_ablation_comparison(
                payload.ablation_results,
                payload.problems,
                save_path=f"{prefix}_ablation.png",
                plot_config=cfg,
            )


__all__ = [
    "AblationStudyPlots",
    "AlgorithmMechanismPlots",
    "BenchmarkFigurePayload",
    "PerformanceComparisonPlots",
    "ProcessAnalysisPlots",
    "StatisticalAnalysisPlots",
    "generate_all_figures",
]
