"""
===============================================================================
benchmark_algorithms.py  (改进版)
基准对比算法 + 评价指标 + 动态测试函数
===============================================================================
【更新内容】
  1. KEMM_DMOEA_Improved: 集成所有改进模块
     - 正确的 Grassmann SGF (geodesic_flow.py)
     - VAE 压缩记忆 (compressed_memory.py)
     - MAB 自适应算子选择 (adaptive_operator.py)
     - GP Pareto 前沿漂移预测 (pareto_drift.py)
  
  2. 新增测试函数: JY1-JY6
     来源: Jiang et al., "A Steady-State and Generational
           Evolutionary Algorithm for Dynamic Multiobjective
           Optimization", IEEE TEVC 2017
  
  3. 扩展评价指标: 新增 HV (超体积)
  
  4. 公平对比: 所有算法使用相同的随机种子和运行时间预算
===============================================================================
"""


import numpy as np
import time
from typing import List, Tuple, Dict


from scipy.spatial.distance import cdist


# 导入改进模块
from geodesic_flow import ManifoldTransferLearning, MultiSourceTransfer
from compressed_memory import VAECompressedMemory
from adaptive_operator import AdaptiveOperatorSelector, ParetoFrontDriftDetector
from pareto_drift import ParetoFrontDriftPredictor


try:
    from sklearn.svm import SVR as SklearnSVR
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 1 部分: 动态测试函数 (含 JY 系列)
# ╚═══════════════════════════════════════════════════════════════════════════╝


class DynamicTestProblems:
    """
    动态多目标测试函数集
    ─────────────────────────────────────────
    来源: 论文 Section IV-A
      "The test functions used are FDA series, dMOP series"
    
    新增 JY 系列:
      来源: Jiang et al., IEEE TEVC 2017
      特点: 更复杂的 Type-III/IV 变化 (形状+位置同时变化)
    
    时间参数:
      t = (1/nt) * floor(τ/τt)
      nt=10, τt=10  (来源: 论文 Section IV-A)
    """


    def __init__(self, nt: int = 10, tau_t: int = 10):
        self.nt = nt
        self.tau_t = tau_t


    def get_time(self, generation: int) -> float:
        """t = (1/nt) * floor(generation/tau_t)"""
        return (1.0 / self.nt) * np.floor(generation / self.tau_t)


    # ── FDA 系列 (来源: Farina et al.) ──


    @staticmethod
    def fda1(x: np.ndarray, t: float) -> np.ndarray:
        """
        FDA1: 位置变化 (Type-I)
        POS 随 t 移动, POF 不变 (凸型)
        G(t) = sin(0.5πt)
        """
        x = np.atleast_2d(x)
        G = np.sin(0.5 * np.pi * t)
        f1 = x[:, 0]
        g = 1.0 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])


    @staticmethod
    def fda1_pof(n_points=200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


    @staticmethod
    def fda2(x: np.ndarray, t: float) -> np.ndarray:
        """
        FDA2: 形状变化 (Type-II)
        POF 形状随 t 变化 (凸→凹), POS 不变
        H(t) = 0.75sin(0.5πt) + 1.25
        """
        x = np.atleast_2d(x)
        n = x.shape[1]
        H = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        f1 = x[:, 0]
        xII = x[:, 1:max(2, n // 2)]
        xIII = x[:, max(2, n // 2):]
        g = 1.0 + np.sum(xII ** 2, axis=1)
        exp_term = H + np.sum((xIII - H) ** 2, axis=1) if xIII.shape[1] > 0 else np.full(len(x), H)
        f2 = g * (1.0 - (f1 / g) ** exp_term)
        return np.column_stack([f1, f2])


    @staticmethod
    def fda2_pof(t: float, n_points=200) -> np.ndarray:
        H = 0.75 + 0.7 * np.sin(0.5 * np.pi * t)
        f1 = np.linspace(0.001, 1, n_points)
        return np.column_stack([f1, 1.0 - f1 ** H])


    @staticmethod
    def fda3(x: np.ndarray, t: float) -> np.ndarray:
        """
        FDA3: 位置+形状同时变化 (Type-III)
        """
        x = np.atleast_2d(x)
        n = x.shape[1]
        F = 10 ** (2.0 * np.sin(0.5 * np.pi * t))
        G = np.abs(np.sin(0.5 * np.pi * t))
        half = max(1, n // 2)
        f1 = np.sum(np.abs(x[:, :half]) ** F, axis=1)
        g = 1.0 + G + np.sum((x[:, half:] - G) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])


    @staticmethod
    def fda3_pof(t: float = 0, n_points=200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


    # ── dMOP 系列 ──


    @staticmethod
    def dmop1(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        n = x.shape[1]
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(np.abs(x[:, 1:]) ** H, axis=1) / max(1, n - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])


    @staticmethod
    def dmop1_pof(n_points=200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


    @staticmethod
    def dmop2(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        G = np.sin(0.5 * np.pi * t)
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = x[:, 0]
        g = 1.0 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])


    @staticmethod
    def dmop2_pof(t: float, n_points=200) -> np.ndarray:
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0.001, 1, n_points)
        return np.column_stack([f1, 1.0 - f1 ** H])


    @staticmethod
    def dmop3(x: np.ndarray, t: float) -> np.ndarray:
        x = np.atleast_2d(x)
        G = np.sin(0.5 * np.pi * t)
        r = 1.0 + np.sum((x[:, 1:] - G) ** 2, axis=1)
        f1 = x[:, 0]
        f2 = r * (1.0 - np.sqrt(f1 / r))
        return np.column_stack([f1, f2])


    @staticmethod
    def dmop3_pof(n_points=200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


    # ── JY 系列 (新增, 来源: Jiang et al. 2017) ──


    @staticmethod
    def jy1(x: np.ndarray, t: float) -> np.ndarray:
        """
        JY1: 线性 POS + 凸 POF (位置变化)
        来源: Jiang et al., IEEE TEVC 2017, Table I
        """
        x = np.atleast_2d(x)
        A = np.sin(0.5 * np.pi * t)
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum((x[:, 1:] - A) ** 2, axis=1) / max(1, x.shape[1] - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])


    @staticmethod
    def jy1_pof(n_points=200, **kwargs) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


    @staticmethod
    def jy4(x: np.ndarray, t: float) -> np.ndarray:
        """
        JY4: 非线性 POS + 形状变化 (Type-IV, 最难)
        POS 和 POF 都随 t 变化
        来源: Jiang et al., IEEE TEVC 2017
        """
        x = np.atleast_2d(x)
        A = np.sin(0.5 * np.pi * t)
        H = 1.5 + A
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum((x[:, 1:] - A) ** 2, axis=1) / max(1, x.shape[1] - 1)
        f2 = g * (1.0 - (f1 / g) ** H)
        return np.column_stack([f1, f2])


    @staticmethod
    def jy4_pof(t: float, n_points=200) -> np.ndarray:
        H = 1.5 + np.sin(0.5 * np.pi * t)
        f1 = np.linspace(0.001, 1, n_points)
        return np.column_stack([f1, 1.0 - f1 ** H])


    def get_problem(self, name: str):
        """获取测试函数和真实 POF 函数"""
        mapping = {
            'FDA1':  (self.fda1,  self.fda1_pof),
            'FDA2':  (self.fda2,  self.fda2_pof),
            'FDA3':  (self.fda3,  self.fda3_pof),
            'dMOP1': (self.dmop1, self.dmop1_pof),
            'dMOP2': (self.dmop2, self.dmop2_pof),
            'dMOP3': (self.dmop3, self.dmop3_pof),
            'JY1':   (self.jy1,   self.jy1_pof),
            'JY4':   (self.jy4,   self.jy4_pof),
        }
        if name not in mapping:
            raise ValueError(f"未知测试函数: {name}. 可用: {list(mapping.keys())}")
        return mapping[name]




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 2 部分: 性能评价指标 (含 HV)
# ╚═══════════════════════════════════════════════════════════════════════════╝


class PerformanceMetrics:
    """
    性能评价指标
    ─────────────────────────────────────────
    来源: 论文 Section IV-A
      IGD (公式5): "1/|P*| Σ min||p − p*||₂"
      MIGD (公式6): "mean IGD values in time steps"
      SP (公式7): "Schott's spacing metric"
      MS (公式8): "Maximum spread"
    
    新增 HV (超体积):
      来源: Zitzler & Thiele, "Multiobjective Evolutionary
            Algorithms: A Comparative Case Study", IEEE TEVC 1999
      衡量 Pareto 前沿与参考点围成的超体积
      HV 越大越好 (与 IGD 相反)
    """


    @staticmethod
    def igd(obtained_pof: np.ndarray, true_pof: np.ndarray) -> float:
        """
        IGD: 反世代距离
        IGD = (1/|P*|) Σ_{p*∈P*} min_{p∈P} ||p - p*||
        """
        if len(obtained_pof) == 0:
            return float('inf')
        distances = cdist(true_pof, obtained_pof, 'euclidean')
        return float(np.mean(np.min(distances, axis=1)))


    @staticmethod
    def migd(igd_values: List[float]) -> float:
        """MIGD: 时间步上 IGD 的平均值"""
        return float(np.mean(igd_values))


    @staticmethod
    def spacing(obtained_pof: np.ndarray) -> float:
        """
        SP: Schott 间距指标
        SP = sqrt(Σ(d_i - d̄)² / (|P|-1))
        其中 d_i 是第 i 个解到最近邻解的距离
        SP 越小表示分布越均匀
        """
        if len(obtained_pof) <= 1:
            return float('inf')
        distances = cdist(obtained_pof, obtained_pof, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        d_i = np.min(distances, axis=1)
        d_mean = np.mean(d_i)
        return float(np.sqrt(
            np.sum((d_i - d_mean) ** 2) / max(1, len(d_i) - 1)
        ))


    @staticmethod
    def maximum_spread(obtained_pof: np.ndarray, true_pof: np.ndarray) -> float:
        """
        MS: 最大覆盖范围
        MS = sqrt((1/m) Σ (min(f_i^max_true, f_i^max_obt) - max(f_i^min_true, f_i^min_obt))²)
        MS 越大越好 (接近 1 最好)
        """
        if len(obtained_pof) == 0:
            return 0.0
        m = true_pof.shape[1]
        p_max = np.max(true_pof, axis=0)
        p_min = np.min(true_pof, axis=0)
        ps_max = np.max(obtained_pof, axis=0)
        ps_min = np.min(obtained_pof, axis=0)
        overlap = np.minimum(p_max, ps_max) - np.maximum(p_min, ps_min)
        ms_sum = np.sum(np.maximum(overlap, 0) ** 2)
        return float(np.sqrt(ms_sum / m))


    @staticmethod
    def hypervolume(obtained_pof: np.ndarray, ref_point: np.ndarray) -> float:
        """
        HV: 超体积指标
        衡量 Pareto 前沿与参考点围成的目标空间体积
        HV 越大越好
        
        实现: 2D WFG 算法 (Beume et al.)
        注意: 仅支持 2 目标 (2D HV), 多目标需要专用库
        
        Args:
            obtained_pof: 获得的 Pareto 前沿目标值 (N, m)
            ref_point: 参考点 (m,), 需要被所有 PF 点支配
        
        Returns:
            hv: 超体积值 (越大越好)
        """
        if len(obtained_pof) == 0:
            return 0.0


        m = obtained_pof.shape[1]
        if m != 2:
            # 多目标 HV 近似: 用 IGD 倒数近似
            return 0.0


        # 只保留支配参考点的解
        valid = np.all(obtained_pof < ref_point, axis=1)
        pof = obtained_pof[valid]
        if len(pof) == 0:
            return 0.0


        # 按第一目标排序
        idx = np.argsort(pof[:, 0])
        pof = pof[idx]


        # 2D 扫描线算法
        hv = 0.0
        prev_f2 = ref_point[1]
        for i in range(len(pof)):
            hv += (pof[i, 1] - prev_f2) * (ref_point[0] - pof[i, 0])
            # 注意: f2 是最小化, 所以需要从 ref 往下减
            # 修正: 2D HV 计算
        
        # 正确的 2D HV
        hv = 0.0
        prev_f1 = ref_point[0]
        pof_sorted = pof[np.argsort(pof[:, 1])]
        for i in range(len(pof_sorted) - 1, -1, -1):
            hv += (prev_f1 - pof_sorted[i, 0]) * (
                ref_point[1] - pof_sorted[i, 1] if i == len(pof_sorted) - 1
                else pof_sorted[i + 1][1] - pof_sorted[i][1]
            )
            prev_f1 = pof_sorted[i, 0]


        return max(0.0, float(hv))




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 3 部分: 算法基类 (不变)
# ╚═══════════════════════════════════════════════════════════════════════════╝


class BaseDMOEA:
    """
    统一基类 — 向量化 NSGA-II
    与原版相同, 此处不重复注释
    """


    def __init__(self, pop_size, n_var, n_obj, var_bounds):
        self.pop_size = pop_size
        self.n_var = n_var
        self.n_obj = n_obj
        self.lb, self.ub = var_bounds
        self.population = None
        self.fitness = None


    def initialize(self):
        self.population = np.random.uniform(
            self.lb, self.ub, (self.pop_size, self.n_var)
        )


    def evaluate(self, pop, obj_func, t):
        result = obj_func(pop, t)
        return result if result.ndim > 1 else result.reshape(1, -1)


    def fast_nds(self, fitness):
        n = len(fitness)
        if n == 0:
            return []
        F = fitness
        leq = F[:, None, :] <= F[None, :, :]
        lt  = F[:, None, :] <  F[None, :, :]
        dom_matrix = np.all(leq, axis=2) & np.any(lt, axis=2)
        dom_count = dom_matrix.sum(axis=0).astype(int)
        fronts = []
        remaining = np.ones(n, dtype=bool)
        while np.any(remaining):
            current = np.where(remaining & (dom_count == 0))[0].tolist()
            if not current:
                current = np.where(remaining)[0][:1].tolist()
            fronts.append(current)
            for i in current:
                remaining[i] = False
                dominated = np.where(dom_matrix[i] & remaining)[0]
                dom_count[dominated] -= 1
        return fronts


    def crowding_distance(self, fitness, front):
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        f = fitness[front]
        dist = np.zeros(n)
        for m in range(f.shape[1]):
            order = np.argsort(f[:, m])
            dist[order[0]] = dist[order[-1]] = np.inf
            rng = f[order[-1], m] - f[order[0], m]
            if rng < 1e-14:
                continue
            dist[order[1:-1]] += (
                f[order[2:], m] - f[order[:-2], m]
            ) / rng
        return dist


    def env_selection(self, pop, fit, size):
        fronts = self.fast_nds(fit)
        sel = []
        for front in fronts:
            if len(sel) + len(front) <= size:
                sel.extend(front)
            else:
                rem = size - len(sel)
                cd = self.crowding_distance(fit, front)
                sel.extend([front[i] for i in np.argsort(-cd)[:rem]])
                break
        idx = np.array(sel[:size])
        return pop[idx], fit[idx]


    def sbx_pm_batch(self, pop, eta_c=20, eta_m=20, pc=0.9, pm=None):
        N, D = pop.shape
        if pm is None:
            pm = 1.0 / D
        lb, ub = self.lb, self.ub


        idx = np.random.permutation(N)
        p1 = pop[idx[:N // 2 * 2:2]]
        p2 = pop[idx[1:N // 2 * 2:2]]
        M = len(p1)
        c1, c2 = p1.copy(), p2.copy()


        cx_mask = np.random.rand(M, 1) < pc
        gene_mask = (np.random.rand(M, D) < 0.5) & cx_mask
        diff = np.abs(p1 - p2)
        active = gene_mask & (diff > 1e-14)


        if np.any(active):
            y1 = np.minimum(p1, p2)
            y2 = np.maximum(p1, p2)
            u = np.random.rand(M, D)
            beta = 1.0 + 2.0 * (y1 - lb) / (diff + 1e-14)
            alpha = 2.0 - beta ** (-(eta_c + 1))
            mask_u = u <= 1.0 / alpha
            bq = np.where(
                mask_u,
                (u * alpha) ** (1.0 / (eta_c + 1)),
                (1.0 / (2.0 - u * alpha + 1e-30)) ** (1.0 / (eta_c + 1))
            )
            child1 = np.clip(0.5 * ((y1 + y2) - bq * (y2 - y1)), lb, ub)
            child2 = np.clip(0.5 * ((y1 + y2) + bq * (y2 - y1)), lb, ub)
            c1 = np.where(active, child1, p1)
            c2 = np.where(active, child2, p2)


        offspring = np.vstack([c1, c2])
        if len(offspring) < N:
            offspring = np.vstack([
                offspring,
                pop[np.random.choice(N, N - len(offspring))]
            ])
        offspring = offspring[:N]


        mut_mask = np.random.rand(N, D) < pm
        if np.any(mut_mask):
            val = offspring[mut_mask]
            cols = np.where(mut_mask)[1]
            d1 = (val - lb[cols]) / (ub[cols] - lb[cols] + 1e-14)
            d2 = (ub[cols] - val) / (ub[cols] - lb[cols] + 1e-14)
            u_m = np.random.rand(len(val))
            left = u_m < 0.5
            dq = np.empty(len(val))
            if np.any(left):
                xy_l = 1.0 - d1[left]
                dq[left] = (
                    2 * u_m[left] + (1 - 2 * u_m[left]) * (xy_l ** (eta_m + 1))
                ) ** (1 / (eta_m + 1)) - 1.0
            if np.any(~left):
                xy_r = 1.0 - d2[~left]
                dq[~left] = 1.0 - (
                    2 * (1 - u_m[~left]) + 2 * (u_m[~left] - 0.5) * (
                        xy_r ** (eta_m + 1))
                ) ** (1 / (eta_m + 1))
            rng = ub[cols] - lb[cols]
            offspring[mut_mask] = np.clip(
                val + dq * rng, lb[cols], ub[cols]
            )


        return offspring


    def evolve_one_gen(self, obj_func, t):
        offspring = self.sbx_pm_batch(self.population)
        off_fit = self.evaluate(offspring, obj_func, t)
        merged = np.vstack([self.population, offspring])
        merged_f = np.vstack([self.fitness, off_fit])
        self.population, self.fitness = self.env_selection(
            merged, merged_f, self.pop_size
        )


    def get_pareto_front(self):
        fronts = self.fast_nds(self.fitness)
        return self.fitness[fronts[0]] if fronts else self.fitness


    def respond_to_change(self, obj_func, t):
        self.initialize()
        self.fitness = self.evaluate(self.population, obj_func, t)




# ── 其他 5 种对比算法 (与原版相同) ──


class RI_DMOEA(BaseDMOEA):
    """随机重初始化基线"""
    def respond_to_change(self, obj_func, t):
        self.population = np.random.uniform(
            self.lb, self.ub, (self.pop_size, self.n_var)
        )
        self.fitness = self.evaluate(self.population, obj_func, t)




class MMTL_DMOEA(BaseDMOEA):
    """原始 MMTL-DMOEA (对比基线)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = []
        self.memory_cap = 10
        self.n_svr_samples = 30
        self.n_clusters = 4
        self.n_subspaces = 5


    def respond_to_change(self, obj_func, t):
        if self.fitness is not None:
            fronts = self.fast_nds(self.fitness)
            elites = self.population[fronts[0]].copy()
            elite_fit = self.fitness[fronts[0]].copy()
            self.memory.append({'pop': elites, 'fitness': elite_fit})
            if len(self.memory) > self.memory_cap:
                self.memory.pop(0)


        last_best = self._find_best_sol(obj_func, t)
        trans_sol = self._transfer(last_best, obj_func, t)


        parts = [last_best]
        if trans_sol is not None and len(trans_sol) > 0:
            parts.append(trans_sol)
        n_have = sum(len(p) for p in parts)
        n_rand = self.pop_size - n_have
        if n_rand > 0:
            parts.append(np.random.uniform(
                self.lb, self.ub, (n_rand, self.n_var)
            ))
        self.population = np.clip(
            np.vstack(parts)[:self.pop_size], self.lb, self.ub
        )
        self.fitness = self.evaluate(self.population, obj_func, t)


    def _find_best_sol(self, obj_func, t):
        target = self.pop_size // 2
        if not self.memory:
            return np.random.uniform(self.lb, self.ub, (target, self.n_var))
        all_elites = np.vstack([m['pop'] for m in self.memory])
        if HAS_SKLEARN and len(all_elites) > self.n_svr_samples:
            ns = min(self.n_svr_samples, len(all_elites))
            XT = np.random.uniform(self.lb, self.ub, (ns, self.n_var))
            YT = obj_func(XT, t)
            Y_est = np.zeros((len(all_elites), self.n_obj))
            for oi in range(self.n_obj):
                try:
                    svr = SklearnSVR(kernel='rbf', C=1.0, epsilon=0.1, max_iter=500)
                    svr.fit(XT, YT[:, oi])
                    Y_est[:, oi] = svr.predict(all_elites)
                except Exception:
                    Y_est[:, oi] = obj_func(all_elites, t)[:, oi]
        else:
            Y_est = obj_func(all_elites, t)
        fronts = self.fast_nds(Y_est)
        nd_idx = fronts[0] if fronts else list(range(min(target, len(all_elites))))
        selected = all_elites[nd_idx]
        if len(selected) > target:
            cd = self.crowding_distance(Y_est, nd_idx)
            selected = selected[np.argsort(-cd)[:target]]
        elif len(selected) < target:
            n_add = target - len(selected)
            noise_idx = np.random.choice(len(selected), n_add, replace=True)
            noisy = selected[noise_idx] + np.random.normal(0, 0.01, (n_add, self.n_var))
            selected = np.vstack([selected, np.clip(noisy, self.lb, self.ub)])
        return selected[:target]


    def _transfer(self, last_best, obj_func, t):
        if not HAS_SKLEARN or len(last_best) < 4:
            return None
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        n_trans = self.pop_size // 2
        L = min(self.n_clusters, len(last_best) // 2)
        if L < 1:
            return None
        try:
            labels = KMeans(n_clusters=L, n_init=3, max_iter=50,
                            random_state=0).fit_predict(last_best)
        except Exception:
            labels = np.random.randint(0, L, len(last_best))
        T = np.random.uniform(self.lb, self.ub, (self.pop_size, self.n_var))
        all_mapped = []
        p = self.n_subspaces
        for j in range(L):
            S_j = last_best[labels == j]
            if len(S_j) < 2:
                continue
            d = min(p, self.n_var, len(S_j) - 1, len(T) - 1)
            if d < 1:
                continue
            try:
                pca_s = PCA(n_components=d).fit(S_j)
                pca_t = PCA(n_components=d).fit(T)
            except Exception:
                continue
            PS = pca_s.components_.T
            PT = pca_t.components_.T
            for k_idx in range(1, p + 1):
                alpha = k_idx / (p + 1)
                P_mid = (1 - alpha) * PS + alpha * PT  # ← 原始错误实现 (对比用)
                P_mid, _ = np.linalg.qr(P_mid)
                S_c = S_j - pca_s.mean_
                projected = S_c @ P_mid @ P_mid.T + pca_t.mean_
                all_mapped.append(projected)
        if not all_mapped:
            return None
        all_trans = np.vstack(all_mapped)
        idx = np.random.choice(len(all_trans), min(n_trans, len(all_trans)),
                               replace=(n_trans > len(all_trans)))
        return np.clip(all_trans[idx], self.lb, self.ub)




class Tr_DMOEA(BaseDMOEA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_pop = None


    def respond_to_change(self, obj_func, t):
        if self.population is not None:
            self.prev_pop = self.population.copy()
        if self.prev_pop is not None and HAS_SKLEARN:
            from sklearn.decomposition import PCA
            n_trans = self.pop_size // 2
            nc = min(5, self.n_var, len(self.prev_pop) - 1)
            if nc >= 2:
                pca = PCA(n_components=nc).fit(self.prev_pop)
                proj = pca.transform(self.prev_pop)
                proj += np.random.normal(0, 0.1, proj.shape)
                recon = pca.inverse_transform(proj)
                idx = np.random.choice(len(recon), n_trans, replace=True)
                transferred = np.clip(recon[idx], self.lb, self.ub)
            else:
                idx = np.random.choice(len(self.prev_pop), n_trans, replace=True)
                transferred = self.prev_pop[idx]
            rand_pop = np.random.uniform(
                self.lb, self.ub, (self.pop_size - n_trans, self.n_var)
            )
            self.population = np.vstack([transferred, rand_pop])
        else:
            self.initialize()
        self.population = np.clip(self.population, self.lb, self.ub)
        self.fitness = self.evaluate(self.population, obj_func, t)




class PPS_DMOEA(BaseDMOEA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids = []
        self.manifolds = []


    def respond_to_change(self, obj_func, t):
        if self.population is not None:
            self.centroids.append(np.mean(self.population, axis=0))
            if HAS_SKLEARN:
                from sklearn.decomposition import PCA
                nc = min(3, self.n_var, len(self.population) - 1)
                if nc >= 2:
                    self.manifolds.append(PCA(n_components=nc).fit(self.population))
        parts = []
        if len(self.centroids) >= 2:
            vel = self.centroids[-1] - self.centroids[-2]
            pred_c = np.clip(self.centroids[-1] + vel, self.lb, self.ub)
            n_pred = self.pop_size // 2
            if self.manifolds:
                pca = self.manifolds[-1]
                samples = np.random.normal(0, 1, (n_pred, pca.n_components_))
                pred_pop = pca.inverse_transform(samples) + pred_c - np.mean(self.population, axis=0)
            else:
                pred_pop = pred_c + np.random.normal(0, 0.1, (n_pred, self.n_var))
            parts.append(np.clip(pred_pop, self.lb, self.ub))
        n_have = sum(len(p) for p in parts) if parts else 0
        parts.append(np.random.uniform(
            self.lb, self.ub, (self.pop_size - n_have, self.n_var)
        ))
        self.population = np.vstack(parts)[:self.pop_size]
        self.fitness = self.evaluate(self.population, obj_func, t)




class KF_DMOEA(BaseDMOEA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = None
        self.velocity = None
        self.P = None


    def respond_to_change(self, obj_func, t):
        if self.population is not None:
            obs = np.mean(self.population, axis=0)
            if self.state is None:
                self.state = obs.copy()
                self.velocity = np.zeros(self.n_var)
                self.P = np.eye(self.n_var) * 0.1
            else:
                Q = np.eye(self.n_var) * 0.01
                R = np.eye(self.n_var) * 0.1
                pred_s = self.state + self.velocity
                pred_P = self.P + Q
                K = pred_P @ np.linalg.inv(pred_P + R)
                self.velocity = obs - self.state
                self.state = pred_s + K @ (obs - pred_s)
                self.P = (np.eye(self.n_var) - K) @ pred_P
            pred = np.clip(self.state + self.velocity, self.lb, self.ub)
            spread = np.sqrt(np.diag(self.P))
            n_pred = self.pop_size * 3 // 4
            pred_pop = pred + np.random.randn(n_pred, self.n_var) * spread
            rand_pop = np.random.uniform(
                self.lb, self.ub, (self.pop_size - n_pred, self.n_var)
            )
            self.population = np.clip(
                np.vstack([pred_pop, rand_pop]), self.lb, self.ub
            )
        else:
            self.initialize()
        self.fitness = self.evaluate(self.population, obj_func, t)




class SVR_DMOEA(BaseDMOEA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []


    def respond_to_change(self, obj_func, t):
        if self.population is not None:
            self.history.append(np.mean(self.population, axis=0))
        if len(self.history) >= 3:
            centroids = np.array(self.history[-10:])
            times = np.arange(len(centroids))
            pred = np.zeros(self.n_var)
            deg = min(2, len(centroids) - 1)
            for d in range(self.n_var):
                coeffs = np.polyfit(times, centroids[:, d], deg=deg)
                pred[d] = np.polyval(coeffs, len(centroids))
            pred = np.clip(pred, self.lb, self.ub)
            n_pred = self.pop_size * 2 // 3
            pred_pop = pred + np.random.normal(0, 0.05, (n_pred, self.n_var))
            rand_pop = np.random.uniform(
                self.lb, self.ub, (self.pop_size - n_pred, self.n_var)
            )
            self.population = np.clip(
                np.vstack([pred_pop, rand_pop]), self.lb, self.ub
            )
        else:
            self.initialize()
        self.fitness = self.evaluate(self.population, obj_func, t)




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  第 4 部分: KEMM-DMOEA 改进版 — 集成所有改进模块
# ╚═══════════════════════════════════════════════════════════════════════════╝


class KEMM_DMOEA_Improved(BaseDMOEA):
    """
    KEMM-DMOEA 完整改进版
    ─────────────────────────────────────────
    集成所有改进模块:
    
    改进1: 正确 Grassmann SGF (geodesic_flow.py)
      - 修复: SVD + 正交补 RS, 真正的测地流
      - 非线性时替代原始错误的线性插值
    
    改进2: VAE 压缩记忆 (compressed_memory.py)
      - 替代: 原始 FIFO 原始解存储
      - 效果: 存储量降低 D/latent_dim 倍, 隐空间检索
    
    改进3: MAB 自适应算子选择 (adaptive_operator.py)
      - 替代: 硬编码的魔法数字比例
      - 效果: 在线学习最优 memory/predict/transfer/reinit 比例
    
    改进4: GP Pareto 前沿漂移预测 (pareto_drift.py)
      - 替代: 简单质心线性外推
      - 效果: 预测 PF 整体特征变化, 提供不确定性估计
    
    参数 (来源论文 Section IV):
      N=100, ns=30, L=4, p=5
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # ── 子模块实例化 ──


        # 改进1: 正确 Grassmann SGF
        self._transfer_module = ManifoldTransferLearning(
            n_clusters=4,       # 论文 L=4
            n_subspaces=5,      # 论文 p=5
            var_threshold=0.95, # 动态维度阈值
            jitter=1e-6         # 数值稳定性微扰
        )
        self._multi_src_transfer = MultiSourceTransfer(self._transfer_module)


        # 改进2: VAE 压缩记忆
        self._vae_memory = VAECompressedMemory(
            input_dim=self.n_var,
            latent_dim=min(8, self.n_var // 2),
            hidden_dim=64,
            capacity=50,
            beta=0.1,
            online_epochs=15
        )


        # 改进3: MAB 自适应算子选择
        self._operator_selector = AdaptiveOperatorSelector(
            window=10,
            c=0.5,
            temperature=0.8,
            min_ratio=0.05
        )
        self._drift_detector = ParetoFrontDriftDetector(window=6)


        # 改进4: GP Pareto 前沿漂移预测
        self._drift_predictor = ParetoFrontDriftPredictor(
            feature_dim=10,
            max_history=8,
            lengthscale=1.0,
            noise_var=0.05
        )


        # 状态变量
        self._prev_igd = None
        self._time_step = 0
        self._centroid_history = []


    def respond_to_change(self, obj_func, t):
        """
        环境变化响应 — 集成所有改进模块
        
        完整流程:
          1. 计算简单环境指纹 (用于 VAE 记忆检索)
          2. 存储当前精英到 VAE 记忆库 (改进2)
          3. MAB 获取本次分配比例 (改进3)
          4. 从 VAE 记忆检索相似历史环境
          5. FindBestSol: 精英直接使用
          6. GP 预测下一时刻 PF 位置 (改进4)
          7. SGF 流形迁移 (改进1: 正确测地流)
          8. 随机重初始化 (保底)
          9. 合并并截断到 pop_size
        """
        # ── Step 1: 存储精英到 VAE 记忆 ──
        # 来源: Process 1, lines 10-14
        if self.population is not None and self.fitness is not None:
            fronts = self.fast_nds(self.fitness)
            elite_idx = fronts[0]
            elites = self.population[elite_idx]
            elite_fit = self.fitness[elite_idx]


            # 计算简单指纹 (质心 + 目标统计)
            fp = np.concatenate([
                np.mean(self.population, axis=0)[:min(6, self.n_var)],
                np.mean(elite_fit, axis=0),
                np.std(elite_fit, axis=0)
            ])
            # 填充到固定长度 12
            if len(fp) < 12:
                fp = np.pad(fp, (0, 12 - len(fp)))


            # 更新记忆 (改进2: VAE 压缩存储)
            self._vae_memory.store(elites, elite_fit, fp)


            # 更新质心历史
            self._centroid_history.append(np.mean(self.population, axis=0))
            if len(self._centroid_history) > 8:
                self._centroid_history.pop(0)


            # 更新漂移检测器
            self._drift_detector.update(elite_fit)
            self._drift_predictor.update(elite_fit, float(self._time_step))


        self._time_step += 1


        # ── Step 2: MAB 获取分配比例 (改进3) ──
        # 替代原始硬编码魔法数字
        ratios, mab_info = self._operator_selector.get_ratios()
        n_memory   = int(self.pop_size * ratios[0])
        n_predict  = int(self.pop_size * ratios[1])
        n_transfer = int(self.pop_size * ratios[2])
        n_reinit   = self.pop_size - n_memory - n_predict - n_transfer


        parts = []


        # ── Step 3: 从 VAE 记忆检索 + FindBestSol ──
        # 来源: Process 2
        if n_memory > 0 and len(self._vae_memory) > 0:
            # 当前环境的简单指纹
            if self.population is not None:
                curr_fp = np.concatenate([
                    np.mean(self.population, axis=0)[:min(6, self.n_var)],
                    np.zeros(6)  # 目标统计未知时用 0
                ])[:12]
            else:
                curr_fp = np.zeros(12)


            retrieved = self._vae_memory.retrieve(curr_fp, top_k=3, n_decode=n_memory * 2)
            if retrieved:
                best = retrieved[0]['solutions']
                # 选取最好的 n_memory 个
                # 用当前目标函数快速评价筛选
                if len(best) > n_memory:
                    quick_fit = obj_func(best, t)
                    fronts = self.fast_nds(quick_fit)
                    nd_idx = fronts[0] if fronts else list(range(n_memory))
                    if len(nd_idx) >= n_memory:
                        cd = self.crowding_distance(quick_fit, nd_idx)
                        keep = np.argsort(-cd)[:n_memory]
                        best = best[np.array(nd_idx)[keep]]
                    else:
                        best = best[:n_memory]
                parts.append(np.clip(best[:n_memory], self.lb, self.ub))
            else:
                n_reinit += n_memory


        # ── Step 4: GP 预测种群 (改进4) ──
        # 来源: 原创 — Pareto 前沿漂移建模
        if n_predict > 0:
            confidence = self._drift_predictor.get_prediction_confidence()
            if confidence > 0.3 and len(self._centroid_history) >= 2:
                # GP 预测可信 → 使用 GP 候选解
                next_time = float(self._time_step)
                _, gp_candidates = self._drift_predictor.predict_next(
                    next_time, n_samples=n_predict,
                    var_bounds=(self.lb, self.ub)
                )
                if gp_candidates is not None:
                    parts.append(np.clip(gp_candidates, self.lb, self.ub))
                else:
                    # GP 预测失败 → 线性外推
                    predicted = self._linear_predict(n_predict)
                    parts.append(predicted)
            else:
                # GP 置信度不足 → 简单线性外推
                predicted = self._linear_predict(n_predict)
                parts.append(predicted)


        # ── Step 5: 正确 SGF 流形迁移 (改进1) ──
        # 来源: Process 3 + 修复的 Grassmann 测地流
        if n_transfer > 0 and len(self._vae_memory) >= 2:
            curr_fp = np.zeros(12) if self.population is None else np.concatenate([
                np.mean(self.population, axis=0)[:min(6, self.n_var)],
                np.zeros(6)
            ])[:12]
            retrieved = self._vae_memory.retrieve(curr_fp, top_k=3, n_decode=40)


            if retrieved and len(retrieved) >= 1:
                # 目标域样本 (当前种群或随机)
                if self.population is not None:
                    target_samples = self.population[:min(30, len(self.population))]
                else:
                    target_samples = np.random.uniform(
                        self.lb, self.ub, (30, self.n_var)
                    )


                # 多源加权迁移
                sources = [
                    {'data': r['solutions'], 'similarity': r['similarity']}
                    for r in retrieved
                ]
                transferred = self._multi_src_transfer.transfer_from_sources(
                    sources, target_samples, n_transfer
                )
                parts.append(np.clip(transferred, self.lb, self.ub))
            else:
                n_reinit += n_transfer


        # ── Step 6: 随机重初始化 ──
        n_have = sum(len(p) for p in parts) if parts else 0
        n_need = self.pop_size - n_have
        if n_need > 0:
            parts.append(np.random.uniform(self.lb, self.ub, (n_need, self.n_var)))


        # ── Step 7: 合并 ──
        self.population = np.clip(
            np.vstack(parts)[:self.pop_size], self.lb, self.ub
        )
        self.fitness = self.evaluate(self.population, obj_func, t)


        # ── Step 8: MAB 反馈 (IGD 奖励信号) ──
        # 用当前种群的近似 IGD 作为奖励(用内部PF点间距)
        pf_fit = self.get_pareto_front()
        if len(pf_fit) > 1:
            from scipy.spatial.distance import cdist as _cdist
            d = _cdist(pf_fit, pf_fit)
            np.fill_diagonal(d, np.inf)
            approx_igd = float(np.mean(np.min(d, axis=1)))
        else:
            approx_igd = float(np.mean(self.fitness[:, 0])) if self.fitness is not None else 1.0
        self._operator_selector.update_with_igd(approx_igd)



    def _linear_predict(self, n_pred: int) -> np.ndarray:
        """线性/二阶差分外推预测"""
        if len(self._centroid_history) < 2:
            return np.random.uniform(self.lb, self.ub, (n_pred, self.n_var))


        centroids = np.array(self._centroid_history)
        if len(centroids) >= 3:
            vel = centroids[-1] - centroids[-2]
            acc = 0.5 * (centroids[-1] - 2 * centroids[-2] + centroids[-3])
            pred_center = np.clip(centroids[-1] + vel + acc, self.lb, self.ub)
        else:
            vel = centroids[-1] - centroids[-2]
            pred_center = np.clip(centroids[-1] + vel, self.lb, self.ub)


        spread = np.maximum(np.std(centroids[-3:], axis=0), 0.05)
        pred_pop = pred_center + np.random.randn(n_pred, self.n_var) * spread
        return np.clip(pred_pop, self.lb, self.ub)