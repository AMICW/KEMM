"""Baseline algorithms preserved from the original benchmark module."""

from __future__ import annotations

import numpy as np

from kemm.algorithms.base import BaseDMOEA

try:
    from sklearn.svm import SVR as SklearnSVR

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class RI_DMOEA(BaseDMOEA):
    """随机重初始化基线。"""

    def respond_to_change(self, obj_func, t):
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.n_var))
        self.fitness = self.evaluate(self.population, obj_func, t)


class MMTL_DMOEA(BaseDMOEA):
    """原始 MMTL-DMOEA 对比基线。"""

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
            self.memory.append({"pop": elites, "fitness": elite_fit})
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
            parts.append(np.random.uniform(self.lb, self.ub, (n_rand, self.n_var)))
        self.population = np.clip(np.vstack(parts)[: self.pop_size], self.lb, self.ub)
        self.fitness = self.evaluate(self.population, obj_func, t)

    def _find_best_sol(self, obj_func, t):
        target = self.pop_size // 2
        if not self.memory:
            return np.random.uniform(self.lb, self.ub, (target, self.n_var))
        all_elites = np.vstack([m["pop"] for m in self.memory])
        if HAS_SKLEARN and len(all_elites) > self.n_svr_samples:
            ns = min(self.n_svr_samples, len(all_elites))
            XT = np.random.uniform(self.lb, self.ub, (ns, self.n_var))
            YT = obj_func(XT, t)
            Y_est = np.zeros((len(all_elites), self.n_obj))
            for oi in range(self.n_obj):
                try:
                    svr = SklearnSVR(kernel="rbf", C=1.0, epsilon=0.1, max_iter=500)
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
            labels = KMeans(n_clusters=L, n_init=3, max_iter=50, random_state=0).fit_predict(last_best)
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
                P_mid = (1 - alpha) * PS + alpha * PT
                P_mid, _ = np.linalg.qr(P_mid)
                S_c = S_j - pca_s.mean_
                projected = S_c @ P_mid @ P_mid.T + pca_t.mean_
                all_mapped.append(projected)
        if not all_mapped:
            return None
        all_trans = np.vstack(all_mapped)
        idx = np.random.choice(len(all_trans), min(n_trans, len(all_trans)), replace=(n_trans > len(all_trans)))
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
            rand_pop = np.random.uniform(self.lb, self.ub, (self.pop_size - n_trans, self.n_var))
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
        parts.append(np.random.uniform(self.lb, self.ub, (self.pop_size - n_have, self.n_var)))
        self.population = np.vstack(parts)[: self.pop_size]
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
            rand_pop = np.random.uniform(self.lb, self.ub, (self.pop_size - n_pred, self.n_var))
            self.population = np.clip(np.vstack([pred_pop, rand_pop]), self.lb, self.ub)
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
            rand_pop = np.random.uniform(self.lb, self.ub, (self.pop_size - n_pred, self.n_var))
            self.population = np.clip(np.vstack([pred_pop, rand_pop]), self.lb, self.ub)
        else:
            self.initialize()
        self.fitness = self.evaluate(self.population, obj_func, t)


__all__ = ["RI_DMOEA", "MMTL_DMOEA", "Tr_DMOEA", "PPS_DMOEA", "KF_DMOEA", "SVR_DMOEA"]
