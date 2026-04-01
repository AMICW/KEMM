"""
===============================================================================
visualization.py
学术级可视化模块 — 算法对比、过程分析、统计检验图表
===============================================================================
【包含的图表类型】
  A. 性能对比图: MIGD/SP/MS 柱状图、箱线图、热力图
  B. 过程分析图: IGD曲线、MAB学习历史、VAE损失、GP预测
  C. 统计分析图: 秩图、CD图、两两对比矩阵
  D. 算法机制图: Grassmann测地流可视化、Pareto前沿演化
  E. 消融实验图: 各模块贡献柱状图
===============================================================================
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，跳过可视化")


try:
    from scipy.stats import ranksums, friedmanchisquare
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# 统一配色方案（论文级）
ALGO_COLORS = {
    'RI':   '#95a5a6',
    'PPS':  '#3498db',
    'KF':   '#2ecc71',
    'SVR':  '#f39c12',
    'Tr':   '#9b59b6',
    'MMTL': '#e67e22',
    'KEMM': '#e74c3c',  # 红色突出显示
}
ALGO_MARKERS = {
    'RI': 'o', 'PPS': 's', 'KF': '^',
    'SVR': 'D', 'Tr': 'v', 'MMTL': 'P', 'KEMM': '*'
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  1、性能对比图组
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PerformanceComparisonPlots:
    """A类图表：性能对比"""
    
    @staticmethod
    def plot_migd_main_table(results, problems, save_path='fig_A1_migd_bars.pdf'):
        """
        图A1: MIGD主结果柱状图（论文Table对应的可视化）
        设计：每个问题一个子图，算法并排柱状图，标注最优值
        """
        if not HAS_MPL: return
        algos = list(results.keys())
        n_p = len(problems)
        cols = min(3, n_p)
        rows = (n_p + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.atleast_1d(axes).ravel()
        
        for idx, prob in enumerate(problems):
            ax = axes[idx]
            means = [np.mean(results[a][prob]['MIGD']) for a in algos]
            stds  = [np.std(results[a][prob]['MIGD'])  for a in algos]
            best_i = int(np.argmin(means))
            
            colors = [ALGO_COLORS.get(a, '#aaa') for a in algos]
            # 最优值加金色边框
            edgecolors = ['gold' if i == best_i else 'none' for i in range(len(algos))]
            linewidths = [2.5   if i == best_i else 0.5 for i in range(len(algos))]
            
            bars = ax.bar(range(len(algos)), means, yerr=stds,
                          color=colors, capsize=4, alpha=0.85,
                          edgecolor=edgecolors, linewidth=linewidths,
                          error_kw={'linewidth': 1.2})
            
            # 在最优柱顶标注 ★
            ax.text(best_i, means[best_i] + stds[best_i] * 1.1 + 0.001,
                    '★', ha='center', fontsize=14, color='gold')
            
            ax.set_title(prob, fontweight='bold', fontsize=12)
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(algos, rotation=30, fontsize=9)
            ax.set_ylabel('MIGD ↓', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for k in range(n_p, len(axes)):
            axes[k].set_visible(False)
        
        fig.suptitle('MIGD Comparison Across Problems\n(★=best, error bars=std)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图A1] 已保存: {save_path}")
    
    @staticmethod
    def plot_three_metrics_grid(results, problems, 
                                 save_path='fig_A2_three_metrics.pdf'):
        """
        图A2: MIGD/SP/MS三指标联合网格图
        设计：3行（每行一个指标）× n列（每列一个问题），共3×n子图
        """
        if not HAS_MPL: return
        algos = list(results.keys())
        metrics = [('MIGD','↓','#e74c3c'), ('SP','↓','#3498db'), ('MS','↑','#2ecc71')]
        n_p = len(problems)
        
        fig, axes = plt.subplots(3, n_p, figsize=(3*n_p, 9))
        if n_p == 1:
            axes = axes.reshape(3, 1)
        
        for mi, (metric, arrow, mcolor) in enumerate(metrics):
            for pi, prob in enumerate(problems):
                ax = axes[mi, pi]
                vals = [results[a][prob][metric] for a in algos]
                means = [np.mean(v) for v in vals]
                
                direction = 'smaller' if arrow == '↓' else 'larger'
                best_i = int(np.argmin(means) if direction=='smaller' else np.argmax(means))
                
                colors = [mcolor if i == best_i else '#bdc3c7' for i in range(len(algos))]
                # KEMM特殊标注
                colors = ['#e74c3c' if a == 'KEMM' and i != best_i else c
                          for i, (a, c) in enumerate(zip(algos, colors))]
                
                ax.bar(range(len(algos)), means, color=colors, alpha=0.8,
                       edgecolor='k', linewidth=0.3)
                
                if mi == 0:
                    ax.set_title(prob, fontsize=9, fontweight='bold')
                if pi == 0:
                    ax.set_ylabel(f'{metric} {arrow}', fontsize=9, color=mcolor,
                                  fontweight='bold')
                ax.set_xticks(range(len(algos)))
                ax.set_xticklabels(algos if mi == 2 else [], rotation=45, fontsize=7)
                ax.grid(True, alpha=0.2, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        fig.suptitle('Three-Metric Comparison (colored=best, red=KEMM)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图A2] 已保存: {save_path}")
    
    @staticmethod
    def plot_boxplots_all(results, problems, save_path='fig_A3_boxplots.pdf'):
        """
        图A3: 所有算法MIGD分布箱线图
        设计：每个问题一行，横向排列各算法箱线图，直观显示分布
        """
        if not HAS_MPL: return
        algos = list(results.keys())
        n_p = len(problems)
        
        fig, axes = plt.subplots(n_p, 1, figsize=(10, 2.5*n_p))
        if n_p == 1:
            axes = [axes]
        
        for idx, (ax, prob) in enumerate(zip(axes, problems)):
            data = [results[a][prob]['MIGD'] for a in algos]
            
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                            showmeans=True, showfliers=True,
                            meanprops=dict(marker='D', markerfacecolor='red',
                                           markersize=6, zorder=5),
                            flierprops=dict(marker='o', markersize=3, alpha=0.5),
                            medianprops=dict(linewidth=2, color='white'))
            
            for i, (patch, algo) in enumerate(zip(bp['boxes'], algos)):
                c = ALGO_COLORS.get(algo, '#aaa')
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
                if algo == 'KEMM':
                    patch.set_linewidth(2.5)
                    patch.set_edgecolor('#c0392b')
            
            ax.set_xticklabels(algos, fontsize=9)
            ax.set_ylabel('MIGD', fontsize=9)
            ax.set_title(f'{prob}  (◆=mean, ─=median)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.suptitle('MIGD Distribution (red border=KEMM)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图A3] 已保存: {save_path}")
    
    @staticmethod
    def plot_heatmap_normalized(results, problems, save_path='fig_A4_heatmap.pdf'):
        """
        图A4: 归一化MIGD热力图（越绿越好）
        设计：行=算法，列=问题，颜色编码相对排名
        """
        if not HAS_MPL: return
        algos = list(results.keys())
        
        matrix = np.array([[np.mean(results[a][p]['MIGD']) for p in problems]
                            for a in algos])
        # 列归一化：每个问题内部归一化
        col_min = matrix.min(axis=0, keepdims=True)
        col_max = matrix.max(axis=0, keepdims=True)
        norm_matrix = (matrix - col_min) / (col_max - col_min + 1e-12)
        
        fig, ax = plt.subplots(figsize=(max(8, len(problems)*1.5), len(algos)*0.8+2))
        
        # 自定义颜色：绿(好)→黄→红(差)
        cmap = LinearSegmentedColormap.from_list('perf', 
               ['#27ae60', '#f1c40f', '#e74c3c'])
        im = ax.imshow(norm_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        for i, algo in enumerate(algos):
            for j, prob in enumerate(problems):
                val = matrix[i, j]
                norm_val = norm_matrix[i, j]
                text_color = 'white' if norm_val > 0.55 else 'black'
                # 标注原始值
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                        fontsize=8.5, color=text_color, fontweight='bold')
        
        ax.set_xticks(range(len(problems)))
        ax.set_xticklabels(problems, fontsize=11)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=11)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Normalized MIGD\n(0=best, 1=worst)', fontsize=10)
        
        ax.set_title('MIGD Performance Heatmap\n(green=best, red=worst per problem)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图A4] 已保存: {save_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
#  2、过程分析图组
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ProcessAnalysisPlots:
    """B类图表：算法过程分析"""
    
    @staticmethod
    def plot_igd_convergence(igd_curves, problems, save_path='fig_B1_igd_convergence.pdf'):
        """
        图B1: IGD随变化次数的收敛曲线（核心图表）
        设计：每个问题一个子图，展示均值±标准差阴影区域
        """
        if not HAS_MPL: return
        algos = list(igd_curves.keys())
        n_p = len(problems)
        cols = min(3, n_p)
        rows = (n_p + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 4*rows))
        axes = np.atleast_1d(axes).ravel()
        
        for idx, prob in enumerate(problems):
            ax = axes[idx]
            
            for algo in algos:
                curves = igd_curves.get(algo, {}).get(prob, [])
                if not curves:
                    continue
                max_len = max(len(c) for c in curves)
                padded = np.array([c + [c[-1]]*(max_len-len(c)) for c in curves])
                mean_c = padded.mean(axis=0)
                std_c  = padded.std(axis=0)
                x = np.arange(1, max_len+1)
                
                lw = 2.5 if algo == 'KEMM' else 1.3
                ls = '-'  if algo == 'KEMM' else '--'
                ms = 7    if algo == 'KEMM' else 4
                zorder = 10 if algo == 'KEMM' else 3
                
                color = ALGO_COLORS.get(algo, '#aaa')
                marker = ALGO_MARKERS.get(algo, 'o')
                
                # 每隔几个点画一个marker
                step = max(1, max_len // 6)
                ax.plot(x, mean_c, color=color, linewidth=lw, linestyle=ls,
                        marker=marker, markevery=step, markersize=ms,
                        label=algo, zorder=zorder)
                ax.fill_between(x, mean_c-std_c, mean_c+std_c,
                                color=color, alpha=0.12, zorder=zorder-1)
            
            ax.set_xlabel('Environment Change Index', fontsize=9)
            ax.set_ylabel('IGD', fontsize=9)
            ax.set_title(prob, fontweight='bold', fontsize=11)
            ax.legend(fontsize=7, ncol=2, framealpha=0.8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for k in range(n_p, len(axes)):
            axes[k].set_visible(False)
        
        fig.suptitle('IGD Convergence Over Dynamic Changes\n(shaded=±std)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图B1] 已保存: {save_path}")
    
    @staticmethod
    def plot_mab_learning_history(mab_ratios_history, save_path='fig_B2_mab_history.pdf'):
        """
        图B2: MAB学习历史（策略比例随时间变化）
        需要在run_experiments中收集：
            algo._operator_selector.bandit.ratios_history
        """
        if not HAS_MPL or not mab_ratios_history:
            return
        
        ratios = np.array(mab_ratios_history)  # (T, 4)
        T = len(ratios)
        arm_names = ['Memory', 'Predict', 'Transfer', 'Reinit']
        arm_colors = ['#9b59b6', '#3498db', '#e74c3c', '#95a5a6']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        
        # 上图：堆叠面积图（比例随时间变化）
        x = np.arange(1, T+1)
        ax1.stackplot(x, ratios.T, labels=arm_names, colors=arm_colors,
                      alpha=0.8)
        ax1.set_ylabel('Strategy Ratio', fontsize=11)
        ax1.set_title('MAB Adaptive Operator Selection: Strategy Ratios Over Time',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=4)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.2)
        ax1.axhline(0.25, color='gray', linestyle=':', linewidth=1,
                    label='Uniform (0.25)')
        
        # 下图：每个臂的折线图（更清晰看趋势）
        for i, (name, color) in enumerate(zip(arm_names, arm_colors)):
            ax2.plot(x, ratios[:, i], color=color, linewidth=2, label=name)
        ax2.axhline(0.25, color='gray', linestyle=':', linewidth=1.5,
                    label='Initial uniform')
        ax2.set_xlabel('Time Step (Environment Change)', fontsize=11)
        ax2.set_ylabel('Allocation Ratio', fontsize=11)
        ax2.set_title('Individual Strategy Ratio Trends', fontsize=11)
        ax2.legend(fontsize=9, ncol=5)
        ax2.set_ylim(0, 0.7)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图B2] 已保存: {save_path}")
    
    @staticmethod
    def plot_vae_training_loss(vae_loss_history, save_path='fig_B3_vae_loss.pdf'):
        """
        图B3: VAE在线训练损失曲线
        需要在KEMM运行时收集：algo._vae_memory.vae.loss_history
        """
        if not HAS_MPL or not vae_loss_history:
            return
        
        losses = np.array(vae_loss_history)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(1, len(losses)+1)
        
        ax.plot(x, losses, 'b-', linewidth=1.5, alpha=0.7)
        # 移动平均平滑
        window = min(20, len(losses)//5)
        if window > 1:
            smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
            x_smooth = np.arange(window, len(losses)+1)
            ax.plot(x_smooth, smooth, 'r-', linewidth=2.5, label=f'Moving avg (w={window})')
        
        ax.set_xlabel('Training Batch (across all time steps)', fontsize=11)
        ax.set_ylabel('ELBO Loss', fontsize=11)
        ax.set_title('VAE Online Training Loss\n(lower = better compression quality)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图B3] 已保存: {save_path}")
    
    @staticmethod
    def plot_gp_prediction_demo(predictor, save_path='fig_B4_gp_prediction.pdf'):
        """
        图B4: GP漂移预测效果演示
        需要传入训练好的 ParetoFrontDriftPredictor 实例
        """
        if not HAS_MPL: return
        if len(predictor.time_steps) < 4:
            print("  [图B4] GP历史不足，跳过")
            return
        
        features_arr = np.array(predictor.features)
        times = np.array(predictor.time_steps)
        n_demo_dims = min(4, features_arr.shape[1])
        dim_names = ['f1 mean', 'f2 mean', 'f1 std', 'f2 std']
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()
        
        for i in range(n_demo_dims):
            ax = axes[i]
            y = features_arr[:, i]
            
            # 真实历史点
            ax.scatter(times, y, color='k', zorder=5, s=40, label='Observed')
            
            # GP预测（在观测点上）
            gp = predictor.gp_models[i]
            if gp.is_fitted:
                t_pred = np.linspace(times[0], times[-1]*1.3, 50).reshape(-1,1)
                mu, std = gp.predict(t_pred, return_std=True)
                # 反归一化
                y_mean_gp = getattr(gp, '_y_mean', 0.0)
                y_std_gp  = getattr(gp, '_y_std',  1.0)
                mu  = mu  * y_std_gp + y_mean_gp
                std = std * y_std_gp if std is not None else np.ones(len(mu)) * 0.1
                
                ax.plot(t_pred, mu, 'r-', linewidth=2, label='GP prediction')
                ax.fill_between(t_pred.ravel(), mu-2*std, mu+2*std,
                                color='red', alpha=0.15, label='95% CI')
                
                # 标记预测点（最后一个历史点之后）
                last_t = times[-1]
                next_t = np.array([[last_t + 1]])
                mu_next, std_next = gp.predict(next_t, return_std=True)
                mu_next  = float(mu_next[0]) * y_std_gp + y_mean_gp
                std_next = float(std_next[0]) * y_std_gp if std_next is not None else 0.1
                ax.errorbar(last_t+1, mu_next, yerr=2*std_next,
                            fmt='r*', markersize=12, capsize=5,
                            label='Next step pred', zorder=10)
            
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel(dim_names[i] if i < len(dim_names) else f'Feature {i}',
                          fontsize=9)
            ax.set_title(f'PF Feature [{dim_names[i] if i < len(dim_names) else i}]',
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Gaussian Process Prediction of Pareto Front Features',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图B4] 已保存: {save_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
#  3、统计分析图组
# ╚═══════════════════════════════════════════════════════════════════════════╝


class StatisticalAnalysisPlots:
    """C类图表：统计分析"""
    
    @staticmethod
    def plot_critical_difference_diagram(results, problems,
                                          save_path='fig_C1_cd_diagram.pdf'):
        """
        图C1: 平均排名CD图（Critical Difference Diagram）
        横轴=平均排名，越靠左越好，包含Friedman检验p值
        """
        if not HAS_MPL: return
        algos = list(results.keys())
        
        # 计算每个算法在所有问题×指标上的平均排名
        all_ranks = {a: [] for a in algos}
        for metric in ['MIGD', 'SP', 'MS']:
            direction = 'smaller' if metric != 'MS' else 'larger'
            for prob in problems:
                means = np.array([np.mean(results[a][prob][metric]) for a in algos])
                if direction == 'larger':
                    means = -means
                ranks = np.argsort(np.argsort(means)).astype(float) + 1
                for i, a in enumerate(algos):
                    all_ranks[a].append(ranks[i])
        
        avg_ranks = {a: np.mean(r) for a, r in all_ranks.items()}
        sorted_algos = sorted(avg_ranks, key=avg_ranks.get)
        rank_vals = [avg_ranks[a] for a in sorted_algos]
        
        # Friedman检验
        friedman_p = None
        if HAS_SCIPY and len(algos) >= 3:
            try:
                all_rank_matrix = np.array([all_ranks[a] for a in algos])
                _, friedman_p = friedmanchisquare(*all_rank_matrix)
            except Exception:
                pass
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(algos)*0.6+2)))
        
        colors_bar = [ALGO_COLORS.get(a, '#aaa') for a in sorted_algos]
        bars = ax.barh(range(len(sorted_algos)), rank_vals,
                       color=colors_bar, alpha=0.85, edgecolor='k', height=0.6)
        
        for i, (a, rv) in enumerate(zip(sorted_algos, rank_vals)):
            marker = ' ★ KEMM-Improved' if a == 'KEMM' else ''
            ax.text(rv + 0.05, i, f'{rv:.2f}{marker}',
                    va='center', fontsize=10,
                    fontweight='bold' if a == 'KEMM' else 'normal')
        
        ax.set_yticks(range(len(sorted_algos)))
        ax.set_yticklabels(sorted_algos, fontsize=11)
        ax.set_xlabel('Average Rank (lower = better)', fontsize=11)
        ax.set_xlim(0, len(algos) + 0.5)
        ax.invert_yaxis()
        
        title = 'Average Rank CD Diagram\n(MIGD + SP + MS across all problems)'
        if friedman_p is not None:
            title += f'\nFriedman test: p = {friedman_p:.4f}'
            if friedman_p < 0.05:
                title += ' ✓ (significant difference)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图C1] 已保存: {save_path}")
    
    @staticmethod
    def plot_wilcoxon_pvalue_matrix(results, problems, our_algo='KEMM',
                                     save_path='fig_C2_wilcoxon_matrix.pdf'):
        """
        图C2: Wilcoxon秩和检验p值矩阵热力图
        行=KEMM vs 各算法，列=各问题，颜色=p值（越深越显著）
        """
        if not HAS_MPL or not HAS_SCIPY: return
        algos = [a for a in results.keys() if a != our_algo]
        n_prob = len(problems)
        n_algo = len(algos)
        
        pval_matrix = np.ones((n_algo, n_prob))
        sign_matrix = np.zeros((n_algo, n_prob))  # +1=我们更好, -1=我们更差
        
        for j, prob in enumerate(problems):
            our_vals = results[our_algo][prob]['MIGD']
            for i, algo in enumerate(algos):
                other_vals = results[algo][prob]['MIGD']
                try:
                    _, p = ranksums(our_vals, other_vals)
                    pval_matrix[i, j] = p
                    sign_matrix[i, j] = 1 if np.mean(our_vals) < np.mean(other_vals) else -1
                except Exception:
                    pass
        
        fig, ax = plt.subplots(figsize=(max(8, n_prob*1.2), max(4, n_algo*0.8+2)))
        
        # 用-log10(p)显示（越大越显著）
        log_p = -np.log10(np.clip(pval_matrix, 1e-10, 1.0))
        im = ax.imshow(log_p, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=max(3, log_p.max()))
        
        for i in range(n_algo):
            for j in range(n_prob):
                p = pval_matrix[i, j]
                sig = sign_matrix[i, j]
                
                if p < 0.001:  stars = '***'
                elif p < 0.01: stars = '**'
                elif p < 0.05: stars = '*'
                else:          stars = 'ns'
                
                symbol = '+' if sig > 0 else '-'
                text = f'{symbol}\n{stars}' if p < 0.05 else f'{stars}'
                color = 'white' if log_p[i,j] > log_p.max()*0.6 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')
        
        ax.set_xticks(range(n_prob))
        ax.set_xticklabels(problems, fontsize=10)
        ax.set_yticks(range(n_algo))
        ax.set_yticklabels([f'KEMM vs {a}' for a in algos], fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('-log₁₀(p-value)\n(larger = more significant)', fontsize=9)
        ax.axhline(-0.5, color='k', linewidth=2)
        
        ax.set_title(f'Wilcoxon Rank-Sum Test: KEMM vs Others (MIGD)\n'
                     f'(+ = KEMM better, - = KEMM worse, * p<0.05, ** p<0.01, *** p<0.001)',
                     fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图C2] 已保存: {save_path}")
    
    @staticmethod
    def plot_pairwise_win_matrix(results, problems, 
                                  save_path='fig_C3_pairwise_win.pdf'):
        """
        图C3: 算法两两胜负矩阵
        矩阵[i,j] = 算法i在多少问题×指标上优于算法j
        """
        if not HAS_MPL: return
        algos = list(results.keys())
        na = len(algos)
        total = len(problems) * 2  # MIGD + SP
        
        win_matrix = np.zeros((na, na))
        for metric in ['MIGD', 'SP']:
            for prob in problems:
                for i, ai in enumerate(algos):
                    for j, aj in enumerate(algos):
                        if i != j:
                            mi = np.mean(results[ai][prob][metric])
                            mj = np.mean(results[aj][prob][metric])
                            if mi < mj:  # MIGD/SP 越小越好
                                win_matrix[i, j] += 1
        
        fig, ax = plt.subplots(figsize=(9, 8))
        win_ratio = win_matrix / max(total, 1)
        
        im = ax.imshow(win_ratio, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        
        for i in range(na):
            for j in range(na):
                if i == j:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=14, color='gray')
                else:
                    r = win_ratio[i, j]
                    n = int(win_matrix[i, j])
                    color = 'white' if r > 0.6 else 'black'
                    ax.text(j, i, f'{n}/{total}', ha='center', va='center',
                            fontsize=10, color=color, fontweight='bold')
        
        ax.set_xticks(range(na))
        ax.set_xticklabels(algos, fontsize=11)
        ax.set_yticks(range(na))
        ax.set_yticklabels(algos, fontsize=11)
        plt.colorbar(im, ax=ax, label='Win Ratio (row beats column)')
        
        ax.set_title('Pairwise Win Matrix\n(row beats column in MIGD+SP)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图C3] 已保存: {save_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
#  4、算法机制可视化图组
# ╚═══════════════════════════════════════════════════════════════════════════╝


class AlgorithmMechanismPlots:
    """D类图表：算法机制可视化"""
    
    @staticmethod
    def plot_grassmann_geodesic_comparison(save_path='fig_D1_grassmann_demo.pdf'):
        """
        图D1: Grassmann测地流 vs 线性插值对比演示
        用2D→1D的简单例子可视化两种方法的区别
        """
        if not HAS_MPL: return
        
        np.random.seed(42)
        D, d = 50, 2  # 50维空间，2维子空间
        
        # 创建两个不同的2维子空间
        A = np.random.randn(D, d)
        PS, _ = np.linalg.qr(A)
        PS = PS[:, :d]
        
        B = np.random.randn(D, d)
        PT, _ = np.linalg.qr(B)
        PT = PT[:, :d]
        
        n_steps = 10
        
        # 正确测地流的主角
        M = PS.T @ PT
        U, sigma, Vt = np.linalg.svd(M)
        sigma = np.clip(sigma[:d], -1+1e-7, 1-1e-7)
        theta = np.arccos(sigma)  # 主角
        
        # 计算RS（正交补）
        PT_proj = PS @ (PS.T @ PT)
        PT_residual = PT - PT_proj
        RS, _ = np.linalg.qr(PT_residual)
        RS = RS[:, :d]
        V = Vt.T
        
        # 收集两种方法的中间子空间
        angles_geodesic = []
        angles_linear   = []
        
        for k in range(n_steps + 1):
            t = k / n_steps
            
            # 正确测地流
            cos_t = np.diag(np.cos(t * theta))
            sin_t = np.diag(np.sin(t * theta))
            phi_t = PS @ U[:, :d] @ cos_t + RS @ V[:d, :] @ sin_t
            phi_t, _ = np.linalg.qr(phi_t)
            phi_t = phi_t[:, :d]
            
            # 线性插值（原版错误方法）
            P_lin = (1-t)*PS + t*PT
            P_lin, _ = np.linalg.qr(P_lin)
            P_lin = P_lin[:, :d]
            
            # 计算与PS的主角（衡量子空间距离）
            Mg = phi_t.T @ PS
            _, sg, _ = np.linalg.svd(Mg)
            angles_geodesic.append(np.mean(np.arccos(np.clip(sg[:d], -1+1e-7, 1-1e-7))))
            
            Ml = P_lin.T @ PS
            _, sl, _ = np.linalg.svd(Ml)
            angles_linear.append(np.mean(np.arccos(np.clip(sl[:d], -1+1e-7, 1-1e-7))))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：主角变化轨迹对比
        ax = axes[0]
        t_vals = np.linspace(0, 1, n_steps+1)
        
        ax.plot(t_vals, np.degrees(angles_geodesic), 'g-o', linewidth=2.5,
                markersize=7, label='Correct Geodesic Flow', zorder=5)
        ax.plot(t_vals, np.degrees(angles_linear), 'r--s', linewidth=2,
                markersize=6, label='Linear Interpolation (Original Bug)')
        
        # 理论测地线应该是线性变化的
        total_angle = np.degrees(np.mean(theta))
        ax.plot([0, 1], [total_angle, 0], 'g:', linewidth=1.5,
                label=f'Ideal geodesic (θ={total_angle:.1f}°)')
        
        ax.set_xlabel('Interpolation parameter t', fontsize=11)
        ax.set_ylabel('Mean principal angle (degrees)', fontsize=11)
        ax.set_title('Grassmann Geodesic vs Linear Interpolation\n'
                     '(geodesic follows true manifold geometry)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 右图：迁移误差对比（随机数据）
        ax = axes[1]
        n_src = 30
        source = np.random.randn(n_src, D) @ PS  # 源域数据
        
        errors_geodesic, errors_linear = [], []
        for k in range(1, n_steps):
            t = k / n_steps
            
            # 正确测地流
            cos_t = np.diag(np.cos(t * theta))
            sin_t = np.diag(np.sin(t * theta))
            phi_t = PS @ U[:, :d] @ cos_t + RS @ V[:d, :] @ sin_t
            phi_t, _ = np.linalg.qr(phi_t)
            phi_t = phi_t[:, :d]
            
            # 线性插值
            P_lin = (1-t)*PS + t*PT
            P_lin, _ = np.linalg.qr(P_lin)
            P_lin = P_lin[:, :d]
            
            # 目标：迁移后的数据应该在PT子空间中
            target_proj = source @ PT @ PT.T  # 理想投影到PT
            
            geo_proj = (source @ phi_t) @ phi_t.T
            lin_proj = (source @ P_lin) @ P_lin.T
            
            errors_geodesic.append(np.mean(np.linalg.norm(target_proj*t - geo_proj*t, axis=1)))
            errors_linear.append(np.mean(np.linalg.norm(target_proj*t - lin_proj*t, axis=1)))
        
        t_inner = np.linspace(0.1, 0.9, n_steps-1)
        ax.plot(t_inner, errors_geodesic, 'g-o', linewidth=2.5, markersize=7,
                label='Correct Geodesic')
        ax.plot(t_inner, errors_linear, 'r--s', linewidth=2, markersize=6,
                label='Linear Interpolation')
        ax.fill_between(t_inner, errors_geodesic, errors_linear,
                        where=np.array(errors_geodesic) < np.array(errors_linear),
                        color='green', alpha=0.15, label='Improvement region')
        
        ax.set_xlabel('Interpolation parameter t', fontsize=11)
        ax.set_ylabel('Transfer Error (L2)', fontsize=11)
        ax.set_title('Transfer Quality Comparison\n(lower = better knowledge transfer)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图D1] 已保存: {save_path}")
    
    @staticmethod
    def plot_pareto_front_evolution(pareto_history, true_pof_func, t_values,
                                     save_path='fig_D2_pf_evolution.pdf'):
        """
        图D2: Pareto前沿在环境变化中的演化
        需要传入每个时刻的Pareto前沿目标值列表
        """
        if not HAS_MPL or not pareto_history: return
        
        n_steps = min(6, len(pareto_history))
        indices = np.linspace(0, len(pareto_history)-1, n_steps, dtype=int)
        
        cols = 3
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.atleast_1d(axes).ravel()
        
        for k, (ax, idx) in enumerate(zip(axes, indices)):
            pf = pareto_history[idx]  # (N, 2) 目标值
            t  = t_values[idx] if idx < len(t_values) else idx/10
            
            # 真实POF
            try:
                true_pof = true_pof_func(t=t)
                ax.plot(true_pof[:, 0], true_pof[:, 1], 'k-',
                        linewidth=2, label='True POF', zorder=1)
            except Exception:
                pass
            
            # 获得的PF
            ax.scatter(pf[:, 0], pf[:, 1], c='#e74c3c', s=20,
                       alpha=0.8, label='Obtained PF', zorder=3)
            
            ax.set_xlabel('f₁', fontsize=9)
            ax.set_ylabel('f₂', fontsize=9)
            ax.set_title(f't={t:.2f} (Change #{idx+1})', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        for k in range(n_steps, len(axes)):
            axes[k].set_visible(False)
        
        fig.suptitle('Pareto Front Evolution Under Dynamic Changes\n'
                     '(black line = true POF, red dots = obtained)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图D2] 已保存: {save_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
#  5、消融实验图组
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AblationStudyPlots:
    """E类图表：消融实验"""
    
    @staticmethod
    def plot_ablation_comparison(ablation_results, problems,
                                  save_path='fig_E1_ablation.pdf'):
        """
        图E1: 消融实验柱状图（每个模块的贡献）
        ablation_results格式同 results，key为变体名称
        """
        if not HAS_MPL: return
        variants = list(ablation_results.keys())
        n_p = len(problems)
        
        # 颜色：完整版红色，其他灰色系
        variant_colors = {}
        full_name = [v for v in variants if 'Full' in v or 'KEMM' == v]
        for v in variants:
            if v in full_name:
                variant_colors[v] = '#e74c3c'
            elif 'noSGF' in v or 'no-SGF' in v:
                variant_colors[v] = '#3498db'
            elif 'noVAE' in v or 'no-VAE' in v:
                variant_colors[v] = '#9b59b6'
            elif 'noMAB' in v or 'no-MAB' in v:
                variant_colors[v] = '#e67e22'
            elif 'noGP' in v or 'no-GP' in v:
                variant_colors[v] = '#27ae60'
            else:
                variant_colors[v] = '#95a5a6'
        
        cols = min(3, n_p)
        rows = (n_p + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.atleast_1d(axes).ravel()
        
        for idx, prob in enumerate(problems):
            ax = axes[idx]
            means = [np.mean(ablation_results[v][prob]['MIGD']) for v in variants]
            stds  = [np.std(ablation_results[v][prob]['MIGD'])  for v in variants]
            colors = [variant_colors.get(v, '#aaa') for v in variants]
            
            bars = ax.bar(range(len(variants)), means, yerr=stds,
                          color=colors, capsize=4, alpha=0.85,
                          edgecolor='k', linewidth=0.5)
            
            # 标注相对完整版的提升百分比
            full_val = means[0] if variants[0] in full_name else min(means)
            for i, (bar, m) in enumerate(zip(bars, means)):
                if i > 0 and full_val > 0:
                    delta = (m - full_val) / full_val * 100
                    color = '#e74c3c' if delta > 5 else '#27ae60'
                    ax.text(bar.get_x() + bar.get_width()/2,
                            m + stds[i] + 0.001,
                            f'+{delta:.0f}%' if delta > 0 else f'{delta:.0f}%',
                            ha='center', fontsize=8, color=color, fontweight='bold')
            
            ax.set_title(prob, fontweight='bold', fontsize=11)
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels(variants, rotation=35, fontsize=8, ha='right')
            ax.set_ylabel('MIGD ↓', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for k in range(n_p, len(axes)):
            axes[k].set_visible(False)
        
        # 图例说明各颜色含义
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=v)
                           for v, c in variant_colors.items()]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=len(variants), fontsize=9, bbox_to_anchor=(0.5, -0.02))
        
        fig.suptitle('Ablation Study: Contribution of Each Module\n'
                     '(% = MIGD increase vs KEMM-Full)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close()
        print(f"  [图E1] 已保存: {save_path}")
    
    @staticmethod
    def plot_module_contribution_radar(ablation_results, problems,
                                        save_path='fig_E2_contribution_radar.pdf'):
        """
        图E2: 各模块贡献的雷达图
        展示移除每个模块后MIGD的平均退化程度
        """
        if not HAS_MPL: return
        variants = list(ablation_results.keys())
        
        # 计算每个变体相对于完整版的退化
        full_key = variants[0]
        improvements = {}
        
        for v in variants[1:]:
            diffs = []
            for prob in problems:
                full_mean  = np.mean(ablation_results[full_key][prob]['MIGD'])
                ablat_mean = np.mean(ablation_results[v][prob]['MIGD'])
                if full_mean > 0:
                    diffs.append((ablat_mean - full_mean) / full_mean * 100)
            improvements[v] = np.mean(diffs) if diffs else 0
        
        module_names = list(improvements.keys())
        values = [improvements[m] for m in module_names]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = range(len(module_names))
        
        colors = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#27ae60'
                  for v in values]
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='k')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.3 if val >= 0 else val - 0.5,
                    f'+{val:.1f}%' if val >= 0 else f'{val:.1f}%',
                    ha='center', fontsize=11, fontweight='bold')
        
        ax.axhline(0, color='k', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(module_names, fontsize=11)
        ax.set_ylabel('MIGD Degradation (%) when module removed', fontsize=11)
        ax.set_title('Module Contribution Analysis\n'
                     '(higher bar = that module contributes more to performance)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [图E2] 已保存: {save_path}")




# ══════════════════════════════════════════════════
#  统一调用接口
# ══════════════════════════════════════════════════


def generate_all_figures(results, problems, igd_curves=None,
                          ablation_results=None, kemm_algo_instance=None,
                          output_dir='.', prefix='kemm'):
    """
    一键生成所有学术图表
    
    Args:
        results: 实验结果字典
        problems: 测试函数名列表
        igd_curves: IGD曲线数据
        ablation_results: 消融实验结果（可选）
        kemm_algo_instance: KEMM算法实例（用于获取内部状态）
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    def path(name):
        return os.path.join(output_dir, f'{prefix}_{name}')
    
    print(f"\n  ═══ 生成学术图表 (输出到: {output_dir}) ═══")
    
    # A类：性能对比
    PerformanceComparisonPlots.plot_migd_main_table(
        results, problems, path('A1_migd_bars.png'))
    PerformanceComparisonPlots.plot_three_metrics_grid(
        results, problems, path('A2_three_metrics.png'))
    PerformanceComparisonPlots.plot_boxplots_all(
        results, problems, path('A3_boxplots.png'))
    PerformanceComparisonPlots.plot_heatmap_normalized(
        results, problems, path('A4_heatmap.png'))
    
    # B类：过程分析
    if igd_curves:
        ProcessAnalysisPlots.plot_igd_convergence(
            igd_curves, problems, path('B1_igd_convergence.png'))
    
    if kemm_algo_instance is not None:
        mab_hist = getattr(
            kemm_algo_instance._operator_selector.bandit, 'ratios_history', [])
        if mab_hist:
            ProcessAnalysisPlots.plot_mab_learning_history(
                mab_hist, path('B2_mab_history.png'))
        
        vae_loss = getattr(
            kemm_algo_instance._vae_memory.vae, 'loss_history', [])
        if vae_loss:
            ProcessAnalysisPlots.plot_vae_training_loss(
                vae_loss, path('B3_vae_loss.png'))
        
        predictor = getattr(kemm_algo_instance, '_drift_predictor', None)
        if predictor and len(predictor.time_steps) >= 4:
            ProcessAnalysisPlots.plot_gp_prediction_demo(
                predictor, path('B4_gp_prediction.png'))
    
    # C类：统计分析
    StatisticalAnalysisPlots.plot_critical_difference_diagram(
        results, problems, path('C1_cd_diagram.png'))
    StatisticalAnalysisPlots.plot_wilcoxon_pvalue_matrix(
        results, problems, save_path=path('C2_wilcoxon.png'))
    StatisticalAnalysisPlots.plot_pairwise_win_matrix(
        results, problems, path('C3_pairwise.png'))
    
    # D类：算法机制
    AlgorithmMechanismPlots.plot_grassmann_geodesic_comparison(
        path('D1_grassmann_demo.png'))
    
    # E类：消融实验
    if ablation_results:
        AblationStudyPlots.plot_ablation_comparison(
            ablation_results, problems, path('E1_ablation.png'))
        AblationStudyPlots.plot_module_contribution_radar(
            ablation_results, problems, path('E2_contribution.png'))
    
    print(f"  ═══ 所有图表生成完毕 ═══\n")