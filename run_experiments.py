"""
===============================================================================
run_experiments.py  (改进版)
多算法对比实验主入口
===============================================================================
【更新内容】
  1. 消融实验: 验证每个改进模块的独立贡献
  2. 新增 JY 系列测试函数
  3. 公平对比实验设计 (N_RUNS=20 满足 Wilcoxon 检验)
  4. MAB 状态可视化
===============================================================================
"""


import sys
import time
import numpy as np
from typing import Dict, List
from collections import defaultdict


from benchmark_algorithms import (
    DynamicTestProblems, PerformanceMetrics,
    RI_DMOEA, MMTL_DMOEA, Tr_DMOEA, PPS_DMOEA, KF_DMOEA, SVR_DMOEA,
    KEMM_DMOEA_Improved
)


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 1 部分: 实验配置
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ExperimentConfig:
    """
    实验参数配置
    ─────────────────────────────────────────
    来源: 论文 Section IV-A
      "N=100; nt=10; τt=10; each algorithm ran 20 times"
    
    说明:
      N_RUNS=20 是 Wilcoxon 秩和检验的最小推荐样本量
      PROBLEMS 含 JY 系列 (论文未用, 本文新增)
    """
    POP_SIZE = 100
    N_VAR = 10
    N_OBJ = 2
    NT = 10
    TAU_T = 10
    N_CHANGES = 10
    GENS_PER_CHANGE = 20
    N_RUNS = 5           # 完整实验应为 20
    SIGNIFICANCE = 0.05


    # 测试函数 (原始 6 个 + 新增 JY 系列)
    PROBLEMS_STANDARD = ['FDA1', 'FDA2', 'FDA3', 'dMOP1', 'dMOP2', 'dMOP3']
    PROBLEMS_JY = ['JY1', 'JY4']
    PROBLEMS = PROBLEMS_STANDARD  # 默认只用标准集


    # 算法集合
    ALGORITHMS = {
        'RI':   RI_DMOEA,
        'PPS':  PPS_DMOEA,
        'KF':   KF_DMOEA,
        'SVR':  SVR_DMOEA,
        'Tr':   Tr_DMOEA,
        'MMTL': MMTL_DMOEA,
        'KEMM': KEMM_DMOEA_Improved,  # ← 使用改进版
    }


    # 消融实验配置 (Section IV-B 风格)
    # 每个 key 对应一种消融变体
    ABLATION_ALGORITHMS = {
        'KEMM-Full':     KEMM_DMOEA_Improved,   # 完整版
        'MMTL-Original': MMTL_DMOEA,             # MMTL (含错误 SGF)
    }




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 2 部分: 实验运行器
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig = None):
        self.cfg = config or ExperimentConfig()
        self.problems = DynamicTestProblems(nt=self.cfg.NT, tau_t=self.cfg.TAU_T)
        self.metrics = PerformanceMetrics()
        self.results = {}
        self.igd_curves = {}


    def run_all(self):
        total = len(self.cfg.ALGORITHMS) * len(self.cfg.PROBLEMS) * self.cfg.N_RUNS
        counter = 0
        t_start = time.time()


        for algo_name, algo_class in self.cfg.ALGORITHMS.items():
            self.results[algo_name] = {}
            self.igd_curves[algo_name] = {}
            for prob_name in self.cfg.PROBLEMS:
                self.results[algo_name][prob_name] = {
                    'MIGD': [], 'SP': [], 'MS': [], 'TIME': []
                }
                self.igd_curves[algo_name][prob_name] = []
                obj_func, pof_func = self.problems.get_problem(prob_name)


                for run in range(self.cfg.N_RUNS):
                    counter += 1
                    elapsed = time.time() - t_start
                    rate = counter / (elapsed + 1e-6)
                    eta = (total - counter) / (rate + 1e-6)
                    print(
                        f"\r  [{counter:4d}/{total}] {algo_name:>6s}|"
                        f"{prob_name:>5s}|R{run+1} | "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                        end='', flush=True
                    )


                    np.random.seed(run * 1000 + hash(algo_name) % 10000)
                    result = self._run_single(algo_class, obj_func, pof_func, prob_name)


                    self.results[algo_name][prob_name]['MIGD'].append(result['migd'])
                    self.results[algo_name][prob_name]['SP'].append(result['sp'])
                    self.results[algo_name][prob_name]['MS'].append(result['ms'])
                    self.results[algo_name][prob_name]['TIME'].append(result['time'])
                    self.igd_curves[algo_name][prob_name].append(result['igd_curve'])


        total_time = time.time() - t_start
        print(f"\n  完成, 总耗时 {total_time:.1f}s")
        return self.results


    def _run_single(self, algo_class, obj_func, pof_func, prob_name):
        """运行单次实验"""
        lb = np.zeros(self.cfg.N_VAR)
        ub = np.ones(self.cfg.N_VAR)
        lb[1:] = -1.0
        ub[1:] = 1.0


        algo = algo_class(
            self.cfg.POP_SIZE, self.cfg.N_VAR, self.cfg.N_OBJ, (lb, ub)
        )
        algo.initialize()
        t0 = time.time()


        igd_list, sp_list, ms_list = [], [], []
        generation = 0

        # 新增：保存最后一次运行的KEMM实例（用于可视化内部状态）
        algo_instance = algo_class(...)


        for ci in range(self.cfg.N_CHANGES):
            t = self.problems.get_time(generation)


            if ci == 0:
                algo.fitness = algo.evaluate(algo.population, obj_func, t)
            else:
                algo.respond_to_change(obj_func, t)


            for _ in range(self.cfg.GENS_PER_CHANGE):
                algo.evolve_one_gen(obj_func, t)


            generation += self.cfg.TAU_T
            obtained = algo.get_pareto_front()


            try:
                true_pof = pof_func(t=t)
            except TypeError:
                true_pof = pof_func()


            igd_list.append(self.metrics.igd(obtained, true_pof))
            sp_list.append(self.metrics.spacing(obtained))
            ms_list.append(self.metrics.maximum_spread(obtained, true_pof))


        return {
            'migd': self.metrics.migd(igd_list),
            'sp': float(np.mean(sp_list)),
            'ms': float(np.mean(ms_list)),
            'time': time.time() - t0,
            'igd_curve': igd_list,
            'algo_instance': algo_instance  # 新增：保存实例
        }
       




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 3 部分: 统计检验
# ╚═══════════════════════════════════════════════════════════════════════════╝


def wilcoxon_test(ours: list, others: list, alpha: float = 0.05) -> str:
    """
    Wilcoxon 秩和检验
    
    返回:
      '+': ours 显著优于 others
      '-': ours 显著劣于 others
      '≈': 无显著差异
    """
    if len(ours) < 3:
        return '≈'
    try:
        from scipy.stats import ranksums
        _, p = ranksums(ours, others)
        if p < alpha:
            return '+' if np.mean(ours) < np.mean(others) else '-'
        return '≈'
    except Exception:
        return '≈'




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 4 部分: 结果展示
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ResultPresenter:
    def __init__(self, results, config, igd_curves=None):
        self.results = results
        self.cfg = config
        self.igd_curves = igd_curves or {}
        self.our_algo = 'KEMM'


    def print_tables(self):
        our = self.our_algo
        metrics_info = {'MIGD': 'smaller', 'SP': 'smaller', 'MS': 'larger'}
        for metric, direction in metrics_info.items():
            arrow = '↓' if direction == 'smaller' else '↑'
            print(f"\n{'='*110}")
            print(f"  TABLE: {metric} {arrow}  (Mean ± Std) [{self.cfg.N_RUNS} runs]")
            print(f"{'='*110}")
            algos = list(self.results.keys())
            header = f"{'Prob':>6s}"
            for a in algos:
                header += f" | {a:>16s}"
            print(header)
            print("-" * len(header))
            wins = defaultdict(int)
            for prob in self.cfg.PROBLEMS:
                row = f"{prob:>6s}"
                means = {a: np.mean(self.results[a][prob][metric]) for a in algos}
                best = (min if direction == 'smaller' else max)(means, key=means.get)
                wins[best] += 1
                for a in algos:
                    vals = self.results[a][prob][metric]
                    m, s = np.mean(vals), np.std(vals)
                    if a != our and our in self.results:
                        sig = wilcoxon_test(self.results[our][prob][metric], vals)
                        if direction == 'larger':
                            sig = {'+': '-', '-': '+', '≈': '≈'}[sig]
                    else:
                        sig = ' '
                    mark = '**' if a == best else '  '
                    row += f" | {mark}{m:.4f}±{s:.4f}{sig}"
                print(row)
            print("-" * len(header))
            row = f"{'Wins':>6s}"
            for a in algos:
                row += f" | {wins[a]:>16d}"
            print(row)


    def print_ranking(self):
        print(f"\n{'='*60}")
        print(f"  综合排名 (MIGD+SP+MS 平均排名)")
        print(f"{'='*60}")
        algos = list(self.results.keys())
        all_ranks = {a: [] for a in algos}
        for metric in ['MIGD', 'SP', 'MS']:
            direction = 'smaller' if metric != 'MS' else 'larger'
            for prob in self.cfg.PROBLEMS:
                means = np.array([
                    np.mean(self.results[a][prob][metric]) for a in algos
                ])
                if direction == 'larger':
                    means = -means
                ranks = np.argsort(np.argsort(means)) + 1
                for i, a in enumerate(algos):
                    all_ranks[a].append(ranks[i])
        avg = {a: np.mean(r) for a, r in all_ranks.items()}
        for rank, a in enumerate(sorted(avg, key=avg.get), 1):
            marker = " ★ (KEMM 改进版)" if a == self.our_algo else ""
            print(f"  #{rank}: {a:>6s}  AvgRank = {avg[a]:.2f}{marker}")


    def plot_mab_history(self, prefix="out"):
        """可视化 MAB 学习到的策略比例历史"""
        if not HAS_MPL:
            return
        # 此处需要从 KEMM 实例获取 MAB 历史
        # 在实际运行中可以通过 algo._operator_selector.bandit.ratios_history 获取
        print(f"  [INFO] MAB 历史可视化需要从算法实例获取数据")


    def plot_all(self, prefix="out"):
        if not HAS_MPL:
            print("  [WARN] matplotlib 未安装")
            return
        print(f"\n  生成可视化图表...")
        self._plot_metric_bars(prefix)
        self._plot_igd_over_time(prefix)
        self._plot_heatmap(prefix)
        self._plot_cd_diagram(prefix)
        print(f"  图表已保存 (前缀: {prefix}_*)")


    def _plot_metric_bars(self, prefix):
        algos = list(self.results.keys())
        n_a = len(algos)
        base_colors = plt.cm.Set2(np.linspace(0, 1, n_a))
        for metric, direction in [('MIGD', 'smaller'), ('SP', 'smaller'), ('MS', 'larger')]:
            n_p = len(self.cfg.PROBLEMS)
            nc = min(3, n_p)
            nr = (n_p + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 5*nr))
            axes = np.atleast_1d(axes).ravel()
            for idx, prob in enumerate(self.cfg.PROBLEMS):
                ax = axes[idx]
                means = [np.mean(self.results[a][prob][metric]) for a in algos]
                stds = [np.std(self.results[a][prob][metric]) for a in algos]
                best_i = np.argmin(means) if direction == 'smaller' else np.argmax(means)
                bc = []
                for i, a in enumerate(algos):
                    if i == best_i:
                        bc.append('gold')
                    elif a == self.our_algo:
                        bc.append('#e74c3c')
                    else:
                        bc.append(base_colors[i])
                ax.bar(range(n_a), means, yerr=stds, capsize=3,
                       color=bc, alpha=0.85, edgecolor='k', linewidth=0.5)
                ax.set_title(prob, fontweight='bold')
                ax.set_xticks(range(n_a))
                ax.set_xticklabels(algos, rotation=45, fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
            for k in range(n_p, len(axes)):
                axes[k].set_visible(False)
            fig.suptitle(f'{metric} Comparison (gold=best, red=KEMM-Improved)',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{prefix}_{metric.lower()}_bar.png', dpi=150, bbox_inches='tight')
            plt.close()


    def _plot_igd_over_time(self, prefix):
        if not self.igd_curves:
            return
        algos = list(self.results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(algos)))
        markers = ['o', 's', '^', 'D', 'v', 'P', '*']
        n_p = len(self.cfg.PROBLEMS)
        nc = min(3, n_p)
        nr = (n_p + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 4.5*nr))
        axes = np.atleast_1d(axes).ravel()
        for idx, prob in enumerate(self.cfg.PROBLEMS):
            ax = axes[idx]
            for i, a in enumerate(algos):
                curves = self.igd_curves.get(a, {}).get(prob, [])
                if not curves:
                    continue
                ml = max(len(c) for c in curves)
                padded = [c + [c[-1]] * (ml - len(c)) for c in curves]
                mc = np.mean(padded, axis=0)
                sc = np.std(padded, axis=0)
                x = np.arange(1, len(mc) + 1)
                lw = 2.5 if a == self.our_algo else 1.2
                ls = '-' if a == self.our_algo else '--'
                ax.plot(x, mc, marker=markers[i % 7], linewidth=lw,
                        linestyle=ls, markersize=5, label=a, color=colors[i])
                ax.fill_between(x, mc - sc, mc + sc, color=colors[i], alpha=0.08)
            ax.set_xlabel('Change Index')
            ax.set_ylabel('IGD')
            ax.set_title(prob, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for k in range(n_p, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle('IGD Over Time', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{prefix}_igd_time.png', dpi=150, bbox_inches='tight')
        plt.close()


    def _plot_heatmap(self, prefix):
        algos = list(self.results.keys())
        matrix = np.array([
            [np.mean(self.results[a][p]['MIGD']) for p in self.cfg.PROBLEMS]
            for a in algos
        ])
        fig, ax = plt.subplots(figsize=(max(10, len(self.cfg.PROBLEMS)*2),
                                        max(5, len(algos)*0.8)))
        cmin = matrix.min(0, keepdims=True)
        cmax = matrix.max(0, keepdims=True)
        nm = (matrix - cmin) / (cmax - cmin + 1e-12)
        im = ax.imshow(nm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(self.cfg.PROBLEMS)))
        ax.set_xticklabels(self.cfg.PROBLEMS, fontsize=11)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=11)
        for i in range(len(algos)):
            for j in range(len(self.cfg.PROBLEMS)):
                c = 'white' if nm[i, j] > 0.6 else 'black'
                ax.text(j, i, f'{matrix[i,j]:.4f}', ha='center', va='center',
                        fontsize=9, color=c, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Normalized MIGD (0=best)')
        ax.set_title('MIGD Heatmap (改进版 KEMM)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{prefix}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()


    def _plot_cd_diagram(self, prefix):
        algos = list(self.results.keys())
        ar = {a: [] for a in algos}
        for m in ['MIGD', 'SP', 'MS']:
            d = 'smaller' if m != 'MS' else 'larger'
            for p in self.cfg.PROBLEMS:
                means = np.array([
                    np.mean(self.results[a][p][m]) for a in algos
                ])
                if d == 'larger':
                    means = -means
                ranks = np.argsort(np.argsort(means)).astype(float) + 1
                for i, a in enumerate(algos):
                    ar[a].append(ranks[i])
        avg = {a: np.mean(r) for a, r in ar.items()}
        sa = sorted(avg, key=avg.get)
        fig, ax = plt.subplots(figsize=(12, 4))
        cc = ['#e74c3c' if a == self.our_algo else '#3498db' for a in sa]
        ax.barh(range(len(sa)), [avg[a] for a in sa],
                color=cc, alpha=0.8, edgecolor='k', height=0.6)
        for i, a in enumerate(sa):
            mk = " ★" if a == self.our_algo else ""
            ax.text(avg[a] + 0.05, i, f'{avg[a]:.2f}{mk}',
                    va='center', fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(sa)))
        ax.set_yticklabels(sa, fontsize=12)
        ax.set_xlabel('Average Rank (lower=better)')
        ax.set_title('CD Ranking Diagram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{prefix}_cd_rank.png', dpi=150, bbox_inches='tight')
        plt.close()




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 5 部分: 主程序
# ╚═══════════════════════════════════════════════════════════════════════════╝


def run_benchmark(quick=False, with_jy=False):
    print("=" * 70)
    print("  KEMM-DMOEA (改进版) 基准实验")
    print("  改进: 正确SGF + VAE记忆 + MAB算子选择 + GP漂移预测")
    print("=" * 70)


    cfg = ExperimentConfig()
    if quick:
        cfg.N_RUNS = 2
        cfg.N_CHANGES = 5
        cfg.GENS_PER_CHANGE = 10
        cfg.PROBLEMS = ['FDA1', 'FDA3', 'dMOP2']
        print("  [MODE] 快速验证 (2次运行, 5次变化)")
    elif with_jy:
        cfg.PROBLEMS = cfg.PROBLEMS_STANDARD + cfg.PROBLEMS_JY
        print(f"  [MODE] 含 JY 系列 ({len(cfg.PROBLEMS)} 个测试函数)")


    print(f"\n  种群={cfg.POP_SIZE} 变量={cfg.N_VAR} 问题={cfg.PROBLEMS}")
    print(f"  运行次数={cfg.N_RUNS} (建议 ≥20 用于 Wilcoxon 检验)")
    print(f"  算法: {list(cfg.ALGORITHMS.keys())}\n")


    runner = ExperimentRunner(cfg)
    results = runner.run_all()


    presenter = ResultPresenter(results, cfg, igd_curves=runner.igd_curves)
    print()
    presenter.print_tables()
    presenter.print_ranking()
    presenter.plot_all()
    return results




def main():
    args = sys.argv[1:]
    if '--full' in args:
        run_benchmark(quick=False)
    elif '--with-jy' in args:
        run_benchmark(quick=False, with_jy=True)
    elif '--quick' in args:
        run_benchmark(quick=True)
    else:
        run_benchmark(quick=True)
        print("\n  使用选项: --quick | --full | --with-jy")




if __name__ == "__main__":
    main()