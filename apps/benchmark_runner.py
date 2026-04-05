"""Benchmark application runner.

这个文件是 benchmark 主线的真实应用入口。

职责分成三类：

1. 定义实验配置
2. 批量执行“算法 × 问题 × 重复运行”实验
3. 把结果整理成控制台表格、图表和结构化报告

说明：

- 根目录 `run_experiments.py` 现在只是 thin wrapper
- benchmark 主线的真实逻辑以本文件为准
- 这里仍然同时包含 runner 和 presenter，后续还可以继续拆分
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from kemm.adapters import BenchmarkPriorAdapter
from kemm.algorithms import (
    KF_DMOEA,
    KEMM_DMOEA_Improved,
    MMTL_DMOEA,
    PPS_DMOEA,
    RI_DMOEA,
    SVR_DMOEA,
    Tr_DMOEA,
)
from kemm.benchmark import DynamicTestProblems, PerformanceMetrics
from kemm.core.types import KEMMConfig as RuntimeKEMMConfig
from kemm.reporting import build_report_paths, export_benchmark_report
from reporting_config import build_benchmark_plot_config

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


class ExperimentConfig:
    """Benchmark 实验配置。

    这个类把 benchmark 主线里最常改的参数集中管理，包括：

    - 问题规模参数
    - 动态环境变化参数
    - 重复运行次数
    - 算法集合
    - 消融算法集合

    这样做的目的是让 quick/full 模式切换和论文复现实验更可控。
    """

    POP_SIZE = 100
    N_VAR = 10
    N_OBJ = 2
    NT = 10
    TAU_T = 10
    N_CHANGES = 10
    GENS_PER_CHANGE = 20
    N_RUNS = 5
    SIGNIFICANCE = 0.05

    PROBLEMS_STANDARD = ["FDA1", "FDA2", "FDA3", "dMOP1", "dMOP2", "dMOP3"]
    PROBLEMS_JY = ["JY1", "JY4"]
    PROBLEMS = PROBLEMS_STANDARD

    ALGORITHMS = {
        "RI": RI_DMOEA,
        "PPS": PPS_DMOEA,
        "KF": KF_DMOEA,
        "SVR": SVR_DMOEA,
        "Tr": Tr_DMOEA,
        "MMTL": MMTL_DMOEA,
        "KEMM": KEMM_DMOEA_Improved,
    }

    ABLATION_ALGORITHMS = {
        "KEMM-Full": KEMM_DMOEA_Improved,
        "MMTL-Original": MMTL_DMOEA,
    }
    KEMM_CONFIG = RuntimeKEMMConfig()


class ExperimentRunner:
    """benchmark 批量实验运行器。"""

    def __init__(self, config: ExperimentConfig | None = None):
        # `results` 保存最终统计量。
        # `igd_curves` 保存环境变化过程中的 IGD 曲线，后续画过程图会用到。
        # `algorithm_diagnostics` 保存算法变化响应的结构化诊断，供可视化和调试使用。
        self.cfg = config or ExperimentConfig()
        self.problems = DynamicTestProblems(nt=self.cfg.NT, tau_t=self.cfg.TAU_T)
        self.metrics = PerformanceMetrics()
        self.results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.igd_curves: Dict[str, Dict[str, List[List[float]]]] = {}
        self.hv_curves: Dict[str, Dict[str, List[List[float]]]] = {}
        self.algorithm_diagnostics: Dict[str, Dict[str, List[list]]] = {}
        self.ablation_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    def run_all(self):
        """运行所有算法、问题和重复次数的组合实验。"""

        total = len(self.cfg.ALGORITHMS) * len(self.cfg.PROBLEMS) * self.cfg.N_RUNS
        counter = 0
        t_start = time.time()

        for algo_name, algo_class in self.cfg.ALGORITHMS.items():
            self.results[algo_name] = {}
            self.igd_curves[algo_name] = {}
            self.hv_curves[algo_name] = {}
            self.algorithm_diagnostics[algo_name] = {}
            for prob_name in self.cfg.PROBLEMS:
                self.results[algo_name][prob_name] = {"MIGD": [], "SP": [], "MS": [], "TIME": []}
                self.igd_curves[algo_name][prob_name] = []
                self.hv_curves[algo_name][prob_name] = []
                self.algorithm_diagnostics[algo_name][prob_name] = []
                obj_func, pof_func = self.problems.get_problem(prob_name)

                for run in range(self.cfg.N_RUNS):
                    counter += 1
                    elapsed = time.time() - t_start
                    rate = counter / (elapsed + 1e-6)
                    eta = (total - counter) / (rate + 1e-6)
                    print(
                        f"\r  [{counter:4d}/{total}] {algo_name:>6s}|{prob_name:>5s}|R{run+1} | "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                        end="",
                        flush=True,
                    )

                    seed_offset = sum((index + 1) * ord(ch) for index, ch in enumerate(algo_name)) % 10000
                    np.random.seed(run * 1000 + seed_offset)
                    result = self._run_single(algo_class, obj_func, pof_func)
                    self.results[algo_name][prob_name]["MIGD"].append(result["migd"])
                    self.results[algo_name][prob_name]["SP"].append(result["sp"])
                    self.results[algo_name][prob_name]["MS"].append(result["ms"])
                    self.results[algo_name][prob_name]["TIME"].append(result["time"])
                    self.igd_curves[algo_name][prob_name].append(result["igd_curve"])
                    self.hv_curves[algo_name][prob_name].append(result["hv_curve"])
                    self.algorithm_diagnostics[algo_name][prob_name].append(result["change_diagnostics"])

        total_time = time.time() - t_start
        print(f"\n  完成, 总耗时 {total_time:.1f}s")
        return self.results

    def run_ablation_all(self):
        """运行默认消融对比。"""

        self.ablation_results = {}
        if not self.cfg.ABLATION_ALGORITHMS:
            return self.ablation_results

        print("\n  运行消融/对照变体...", flush=True)
        total = len(self.cfg.ABLATION_ALGORITHMS) * len(self.cfg.PROBLEMS) * self.cfg.N_RUNS
        counter = 0
        t_start = time.time()
        for algo_name, algo_class in self.cfg.ABLATION_ALGORITHMS.items():
            self.ablation_results[algo_name] = {}
            for prob_name in self.cfg.PROBLEMS:
                self.ablation_results[algo_name][prob_name] = {"MIGD": [], "SP": [], "MS": [], "TIME": []}
                obj_func, pof_func = self.problems.get_problem(prob_name)
                for run in range(self.cfg.N_RUNS):
                    counter += 1
                    elapsed = time.time() - t_start
                    rate = counter / (elapsed + 1e-6)
                    eta = (total - counter) / (rate + 1e-6)
                    print(
                        f"\r  [ABL {counter:3d}/{total}] {algo_name:>12s}|{prob_name:>5s}|R{run+1} | "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                        end="",
                        flush=True,
                    )
                    seed_offset = sum((index + 1) * ord(ch) for index, ch in enumerate(algo_name)) % 10000
                    np.random.seed(run * 1000 + seed_offset)
                    result = self._run_single(algo_class, obj_func, pof_func)
                    self.ablation_results[algo_name][prob_name]["MIGD"].append(result["migd"])
                    self.ablation_results[algo_name][prob_name]["SP"].append(result["sp"])
                    self.ablation_results[algo_name][prob_name]["MS"].append(result["ms"])
                    self.ablation_results[algo_name][prob_name]["TIME"].append(result["time"])
        print(f"\n  消融完成, 总耗时 {time.time() - t_start:.1f}s")
        return self.ablation_results

    def _run_single(self, algo_class, obj_func, pof_func):
        """运行一次单算法-单问题实验。

        这一步内部还包含多个环境变化阶段。
        在每次变化时：

        - 获得当前时间变量 `t`
        - 首次直接评价初始种群
        - 后续调用 `respond_to_change()`
        - 再执行若干代 `evolve_one_gen()`
        - 阶段结束后记录 `IGD / SP / MS`
        """

        lb = np.zeros(self.cfg.N_VAR)
        ub = np.ones(self.cfg.N_VAR)
        lb[1:] = -1.0
        ub[1:] = 1.0

        if algo_class is KEMM_DMOEA_Improved:
            kemm_config = replace(
                self.cfg.KEMM_CONFIG,
                pop_size=self.cfg.POP_SIZE,
                n_var=self.cfg.N_VAR,
                n_obj=self.cfg.N_OBJ,
                benchmark_aware_prior=True,
            )
            algo = algo_class(
                self.cfg.POP_SIZE,
                self.cfg.N_VAR,
                self.cfg.N_OBJ,
                (lb, ub),
                config=kemm_config,
                benchmark_adapter=BenchmarkPriorAdapter(),
            )
        else:
            algo = algo_class(self.cfg.POP_SIZE, self.cfg.N_VAR, self.cfg.N_OBJ, (lb, ub))
        algo.initialize()
        t0 = time.time()

        igd_list: List[float] = []
        hv_list: List[float] = []
        sp_list: List[float] = []
        ms_list: List[float] = []
        generation = 0

        for change_index in range(self.cfg.N_CHANGES):
            t = self.problems.get_time(generation)

            if change_index == 0:
                algo.fitness = algo.evaluate(algo.population, obj_func, t)
            else:
                algo.respond_to_change(obj_func, t)

            for _ in range(self.cfg.GENS_PER_CHANGE):
                algo.evolve_one_gen(obj_func, t)

            generation += self.cfg.GENS_PER_CHANGE
            obtained = algo.get_pareto_front()

            try:
                true_pof = pof_func(t=t)
            except TypeError:
                true_pof = pof_func()
            ref_point = np.max(np.vstack([true_pof, obtained]), axis=0) + 0.1

            igd_list.append(self.metrics.igd(obtained, true_pof))
            hv_list.append(self.metrics.hypervolume(obtained, ref_point))
            sp_list.append(self.metrics.spacing(obtained))
            ms_list.append(self.metrics.maximum_spread(obtained, true_pof))

        return {
            "migd": self.metrics.migd(igd_list),
            "sp": float(np.mean(sp_list)),
            "ms": float(np.mean(ms_list)),
            "time": time.time() - t0,
            "igd_curve": igd_list,
            "hv_curve": hv_list,
            "change_diagnostics": list(getattr(algo, "change_diagnostics_history", [])),
            "algo_instance": algo,
        }


def wilcoxon_test(ours: list, others: list, alpha: float = 0.05) -> str:
    """执行简化版 Wilcoxon/ranksums 显著性标记。

    返回：

    - `+`：ours 显著优于 others
    - `-`：ours 显著劣于 others
    - `≈`：无显著差异或样本不足
    """

    if len(ours) < 3:
        return "≈"
    try:
        from scipy.stats import ranksums

        _, p_value = ranksums(ours, others)
        if p_value < alpha:
            return "+" if np.mean(ours) < np.mean(others) else "-"
        return "≈"
    except Exception:
        return "≈"


class ResultPresenter:
    """benchmark 结果展示器。"""

    def __init__(self, results, config, igd_curves=None, hv_curves=None, algorithm_diagnostics=None, ablation_results=None):
        self.results = results
        self.cfg = config
        self.igd_curves = igd_curves or {}
        self.hv_curves = hv_curves or {}
        self.algorithm_diagnostics = algorithm_diagnostics or {}
        self.ablation_results = ablation_results or {}
        self.our_algo = "KEMM"

    def print_tables(self):
        """逐指标打印 benchmark 结果表。"""

        our = self.our_algo
        metrics_info = {"MIGD": "smaller", "SP": "smaller", "MS": "larger"}
        for metric, direction in metrics_info.items():
            arrow = "↓" if direction == "smaller" else "↑"
            print(f"\n{'=' * 110}")
            print(f"  TABLE: {metric} {arrow}  (Mean ± Std) [{self.cfg.N_RUNS} runs]")
            print(f"{'=' * 110}")
            algos = list(self.results.keys())
            header = f"{'Prob':>6s}"
            for algo in algos:
                header += f" | {algo:>16s}"
            print(header)
            print("-" * len(header))

            wins = defaultdict(int)
            for problem in self.cfg.PROBLEMS:
                row = f"{problem:>6s}"
                means = {algo: np.mean(self.results[algo][problem][metric]) for algo in algos}
                best = (min if direction == "smaller" else max)(means, key=means.get)
                wins[best] += 1
                for algo in algos:
                    values = self.results[algo][problem][metric]
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    if algo != our and our in self.results:
                        sig = wilcoxon_test(self.results[our][problem][metric], values)
                        if direction == "larger":
                            sig = {"+": "-", "-": "+", "≈": "≈"}[sig]
                    else:
                        sig = " "
                    mark = "**" if algo == best else "  "
                    row += f" | {mark}{mean_value:.4f}±{std_value:.4f}{sig}"
                print(row)

            print("-" * len(header))
            row = f"{'Wins':>6s}"
            for algo in algos:
                row += f" | {wins[algo]:>16d}"
            print(row)

    def print_ranking(self):
        """打印综合平均排名。"""

        print(f"\n{'=' * 60}")
        print("  综合排名 (MIGD+SP+MS 平均排名)")
        print(f"{'=' * 60}")
        algos = list(self.results.keys())
        all_ranks = {algo: [] for algo in algos}
        for metric in ["MIGD", "SP", "MS"]:
            direction = "smaller" if metric != "MS" else "larger"
            for problem in self.cfg.PROBLEMS:
                means = np.array([np.mean(self.results[algo][problem][metric]) for algo in algos])
                if direction == "larger":
                    means = -means
                ranks = np.argsort(np.argsort(means)) + 1
                for index, algo in enumerate(algos):
                    all_ranks[algo].append(ranks[index])

        avg_ranks = {algo: np.mean(rank_values) for algo, rank_values in all_ranks.items()}
        for rank, algo in enumerate(sorted(avg_ranks, key=avg_ranks.get), 1):
            marker = " ★ (KEMM 改进版)" if algo == self.our_algo else ""
            print(f"  #{rank}: {algo:>6s}  AvgRank = {avg_ranks[algo]:.2f}{marker}")

    def plot_mab_history(self, prefix="out"):
        """预留的 MAB 历史图接口。"""

        if not HAS_MPL:
            return
        print("  [INFO] 当前未从算法实例自动提取 MAB 历史数据。")

    def plot_all(self, prefix="out", plot_config=None):
        """导出 benchmark 侧最常用的一组图表。"""

        if not HAS_MPL:
            print("  [WARN] matplotlib 未安装")
            return

        from apps.reporting import BenchmarkFigurePayload, BenchmarkPlotConfig, generate_all_figures

        print("\n  生成可视化图表...")
        # 图表层只接收结构化 payload，而不是继续直接探测算法实例内部字段。
        # 这样后续即便你重构 KEMM 内部实现，只要 payload 结构保持稳定，报告层就不用跟着改。
        payload = BenchmarkFigurePayload(
            results=self.results,
            problems=self.cfg.PROBLEMS,
            igd_curves=self.igd_curves,
            hv_curves=self.hv_curves,
            diagnostics=self.algorithm_diagnostics,
            ablation_results=self.ablation_results,
            plot_config=plot_config if plot_config is not None else BenchmarkPlotConfig(),
        )
        generate_all_figures(payload=payload, output_prefix=prefix)
        print(f"  图表已保存 (前缀: {prefix}_*)")

    def _plot_metric_bars(self, prefix):
        algos = list(self.results.keys())
        n_algos = len(algos)
        base_colors = plt.cm.Set2(np.linspace(0, 1, n_algos))
        for metric, direction in [("MIGD", "smaller"), ("SP", "smaller"), ("MS", "larger")]:
            n_problems = len(self.cfg.PROBLEMS)
            n_cols = min(3, n_problems)
            n_rows = (n_problems + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            axes = np.atleast_1d(axes).ravel()
            for idx, problem in enumerate(self.cfg.PROBLEMS):
                ax = axes[idx]
                means = [np.mean(self.results[algo][problem][metric]) for algo in algos]
                stds = [np.std(self.results[algo][problem][metric]) for algo in algos]
                best_index = int(np.argmin(means) if direction == "smaller" else np.argmax(means))
                colors = []
                for algo_index, algo in enumerate(algos):
                    if algo_index == best_index:
                        colors.append("gold")
                    elif algo == self.our_algo:
                        colors.append("#e74c3c")
                    else:
                        colors.append(base_colors[algo_index])
                ax.bar(range(n_algos), means, yerr=stds, capsize=3, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
                ax.set_title(problem, fontweight="bold")
                ax.set_xticks(range(n_algos))
                ax.set_xticklabels(algos, rotation=45, fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
            for idx in range(n_problems, len(axes)):
                axes[idx].set_visible(False)
            fig.suptitle(f"{metric} Comparison (gold=best, red=KEMM-Improved)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_{metric.lower()}_bar.png", dpi=150, bbox_inches="tight")
            plt.close()

    def _plot_igd_over_time(self, prefix):
        if not self.igd_curves:
            return
        algos = list(self.results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(algos)))
        markers = ["o", "s", "^", "D", "v", "P", "*"]
        n_problems = len(self.cfg.PROBLEMS)
        n_cols = min(3, n_problems)
        n_rows = (n_problems + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for idx, problem in enumerate(self.cfg.PROBLEMS):
            ax = axes[idx]
            for algo_index, algo in enumerate(algos):
                curves = self.igd_curves.get(algo, {}).get(problem, [])
                if not curves:
                    continue
                max_len = max(len(curve) for curve in curves)
                padded = [curve + [curve[-1]] * (max_len - len(curve)) for curve in curves]
                mean_curve = np.mean(padded, axis=0)
                std_curve = np.std(padded, axis=0)
                x = np.arange(1, len(mean_curve) + 1)
                linewidth = 2.5 if algo == self.our_algo else 1.2
                linestyle = "-" if algo == self.our_algo else "--"
                ax.plot(
                    x,
                    mean_curve,
                    marker=markers[algo_index % len(markers)],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    markersize=5,
                    label=algo,
                    color=colors[algo_index],
                )
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=colors[algo_index], alpha=0.08)
            ax.set_xlabel("Change Index")
            ax.set_ylabel("IGD")
            ax.set_title(problem, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for idx in range(n_problems, len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle("IGD Over Time", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{prefix}_igd_time.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_heatmap(self, prefix):
        algos = list(self.results.keys())
        matrix = np.array([[np.mean(self.results[algo][problem]["MIGD"]) for problem in self.cfg.PROBLEMS] for algo in algos])
        fig, ax = plt.subplots(figsize=(max(10, len(self.cfg.PROBLEMS) * 2), max(5, len(algos) * 0.8)))
        cmin = matrix.min(0, keepdims=True)
        cmax = matrix.max(0, keepdims=True)
        normalized = (matrix - cmin) / (cmax - cmin + 1e-12)
        im = ax.imshow(normalized, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(self.cfg.PROBLEMS)))
        ax.set_xticklabels(self.cfg.PROBLEMS, fontsize=11)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=11)
        for i in range(len(algos)):
            for j in range(len(self.cfg.PROBLEMS)):
                text_color = "white" if normalized[i, j] > 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Normalized MIGD (0=best)")
        ax.set_title("MIGD Heatmap (KEMM highlighted in tables)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{prefix}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_cd_diagram(self, prefix):
        algos = list(self.results.keys())
        all_ranks = {algo: [] for algo in algos}
        for metric in ["MIGD", "SP", "MS"]:
            direction = "smaller" if metric != "MS" else "larger"
            for problem in self.cfg.PROBLEMS:
                means = np.array([np.mean(self.results[algo][problem][metric]) for algo in algos])
                if direction == "larger":
                    means = -means
                ranks = np.argsort(np.argsort(means)).astype(float) + 1
                for index, algo in enumerate(algos):
                    all_ranks[algo].append(ranks[index])
        avg_ranks = {algo: np.mean(rank_values) for algo, rank_values in all_ranks.items()}
        sorted_algos = sorted(avg_ranks, key=avg_ranks.get)
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#e74c3c" if algo == self.our_algo else "#3498db" for algo in sorted_algos]
        ax.barh(range(len(sorted_algos)), [avg_ranks[algo] for algo in sorted_algos], color=colors, alpha=0.8, edgecolor="k", height=0.6)
        for index, algo in enumerate(sorted_algos):
            marker = " ★" if algo == self.our_algo else ""
            ax.text(avg_ranks[algo] + 0.05, index, f"{avg_ranks[algo]:.2f}{marker}", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(sorted_algos)))
        ax.set_yticklabels(sorted_algos, fontsize=12)
        ax.set_xlabel("Average Rank (lower=better)")
        ax.set_title("Average-Rank Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(f"{prefix}_cd_rank.png", dpi=150, bbox_inches="tight")
        plt.close()


def run_benchmark(
    quick: bool = False,
    with_jy: bool = False,
    output_dir: str | None = None,
    plot_config=None,
):
    """benchmark 主线常用入口。"""

    print("=" * 70)
    print("  KEMM-DMOEA (改进版) 基准实验")
    print("  改进: 正确SGF + VAE记忆 + MAB算子选择 + GP漂移预测")
    print("=" * 70)

    cfg = ExperimentConfig()
    if quick:
        cfg.N_RUNS = 2
        cfg.N_CHANGES = 5
        cfg.GENS_PER_CHANGE = 10
        cfg.PROBLEMS = ["FDA1", "FDA3", "dMOP2"]
        print("  [MODE] 快速验证 (2次运行, 5次变化)")
    elif with_jy:
        cfg.PROBLEMS = cfg.PROBLEMS_STANDARD + cfg.PROBLEMS_JY
        print(f"  [MODE] 含 JY 系列 ({len(cfg.PROBLEMS)} 个测试函数)")

    print(f"\n  种群={cfg.POP_SIZE} 变量={cfg.N_VAR} 问题={cfg.PROBLEMS}")
    print(f"  运行次数={cfg.N_RUNS} (建议 ≥20 用于 Wilcoxon 检验)")
    print(f"  算法: {list(cfg.ALGORITHMS.keys())}\n")

    report_root = build_report_paths(Path(output_dir) if output_dir else None, prefix="benchmark")
    (report_root / "figures").mkdir(parents=True, exist_ok=True)
    figures_prefix = str(report_root / "figures" / "benchmark")

    runner = ExperimentRunner(cfg)
    results = runner.run_all()
    ablation_results = runner.run_ablation_all()

    presenter = ResultPresenter(
        results,
        cfg,
        igd_curves=runner.igd_curves,
        hv_curves=runner.hv_curves,
        algorithm_diagnostics=runner.algorithm_diagnostics,
        ablation_results=ablation_results,
    )
    print()
    presenter.print_tables()
    presenter.print_ranking()
    presenter.plot_all(prefix=figures_prefix, plot_config=plot_config)
    export_benchmark_report(results, cfg, output_root=report_root, ablation_results=ablation_results)
    print(f"\n  报告输出目录: {report_root}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark experiments.")
    parser.add_argument("--quick", action="store_true", help="Run a lightweight smoke configuration.")
    parser.add_argument("--full", action="store_true", help="Run the default full benchmark suite.")
    parser.add_argument("--with-jy", action="store_true", help="Append JY problems to the benchmark suite.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--plot-preset", default="paper", help="Plot preset: default/paper/ieee/nature/thesis.")
    parser.add_argument("--science-style", default="", help="Comma-separated SciencePlots style tuple.")
    parser.add_argument("--appendix-plots", action="store_true", help="Export appendix/debug benchmark plots.")
    parser.add_argument("--interactive-figures", action="store_true", help="Also export interactive matplotlib figure bundles (.fig.pickle).")
    return parser.parse_args()


def main():
    """命令行入口。"""

    args = _parse_args()
    style_overrides = {}
    if args.science_style:
        style_overrides["use_scienceplots"] = True
        style_overrides["science_styles"] = tuple(part.strip() for part in args.science_style.split(",") if part.strip())
    plot_config = build_benchmark_plot_config(
        preset=args.plot_preset,
        style_overrides=style_overrides,
        appendix_plots=args.appendix_plots,
        interactive_figures=args.interactive_figures,
    )

    if args.full:
        run_benchmark(quick=False, output_dir=args.output_dir, plot_config=plot_config)
    elif args.with_jy:
        run_benchmark(quick=False, with_jy=True, output_dir=args.output_dir, plot_config=plot_config)
    elif args.quick:
        run_benchmark(quick=True, output_dir=args.output_dir, plot_config=plot_config)
    else:
        run_benchmark(quick=True, output_dir=args.output_dir, plot_config=plot_config)
        print("\n  使用选项: --quick | --full | --with-jy | --output-dir <path> | --plot-preset <preset>")


if __name__ == "__main__":
    main()
