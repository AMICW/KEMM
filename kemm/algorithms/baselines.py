"""Baseline algorithms preserved from the original benchmark module."""

from __future__ import annotations

import numpy as np

from kemm.algorithms.base import BaseDMOEA
from kemm.core.transfer import ManifoldTransferLearning

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


class _MMTLTransferLearning(ManifoldTransferLearning):
    """MMTL-specific transfer module with an LPCA-style clustering approximation."""

    def __init__(
        self,
        *args,
        cluster_subspace_dim: int = 1,
        lpca_iterations: int = 8,
        transfer_dim: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cluster_subspace_dim = cluster_subspace_dim
        self.lpca_iterations = lpca_iterations
        self.transfer_dim = max(1, int(transfer_dim))

    def _cluster(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        if len(data) <= n_clusters or n_clusters <= 1:
            return np.arange(len(data)) % max(1, n_clusters)

        centers = self._kmeans_plus_plus_init(data, n_clusters)
        bases = [self._fit_local_basis(data[np.argmin(np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2), axis=1) == idx])
                 for idx in range(n_clusters)]

        labels = np.argmin(np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2), axis=1)
        for _ in range(self.lpca_iterations):
            distances = np.column_stack([
                self._subspace_distance(data, centers[idx], bases[idx])
                for idx in range(n_clusters)
            ])
            labels = np.argmin(distances, axis=1)
            moved = False
            for idx in range(n_clusters):
                cluster_data = data[labels == idx]
                if len(cluster_data) == 0:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    centers[idx] = data[farthest]
                    bases[idx] = None
                    moved = True
                    continue
                new_center = np.mean(cluster_data, axis=0)
                if not np.allclose(new_center, centers[idx]):
                    moved = True
                centers[idx] = new_center
                bases[idx] = self._fit_local_basis(cluster_data)
            if not moved:
                break
        return labels

    def _kmeans_plus_plus_init(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        centers = [data[np.random.randint(len(data))]]
        while len(centers) < n_clusters:
            center_array = np.asarray(centers)
            sq_dist = np.min(np.sum((data[:, None, :] - center_array[None, :, :]) ** 2, axis=2), axis=1)
            total = float(np.sum(sq_dist))
            if total <= 1e-12:
                centers.append(data[np.random.randint(len(data))])
                continue
            probs = sq_dist / total
            centers.append(data[np.random.choice(len(data), p=probs)])
        return np.asarray(centers, dtype=float)

    def _fit_local_basis(self, cluster_data: np.ndarray) -> np.ndarray | None:
        if cluster_data is None or len(cluster_data) < 2:
            return None
        centered = cluster_data - np.mean(cluster_data, axis=0)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        dim = min(self.cluster_subspace_dim, vt.shape[0], cluster_data.shape[1])
        if dim < 1:
            return None
        return vt[:dim].T

    def _subspace_distance(self, data: np.ndarray, center: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
        residual = data - center
        if basis is None or basis.size == 0:
            return np.sum(residual**2, axis=1)
        projection = residual @ basis @ basis.T
        orthogonal = residual - projection
        return np.sum(orthogonal**2, axis=1)

    def _adaptive_dim(self, pca_model, max_d: int) -> int:
        return max(1, min(self.transfer_dim, max_d))


class MMTL_DMOEA(BaseDMOEA):
    """原始 MMTL-DMOEA 对比基线。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive = np.empty((0, self.n_var), dtype=float)
        self.archive_capacity = 10 * self.pop_size
        self.n_svr_samples = 30
        self.n_clusters = 4
        self.n_subspaces = 5
        self.neighborhood_size = min(20, self.pop_size)
        self.de_F = 0.5
        self.de_CR = 1.0
        self.neighbor_mating_prob = 0.9
        self.max_replacements = 2
        self.transfer_module = _MMTLTransferLearning(
            n_clusters=self.n_clusters,
            n_subspaces=self.n_subspaces,
            cluster_subspace_dim=max(1, self.n_obj - 1),
            transfer_dim=max(1, self.n_obj - 1),
        )
        self.weight_vectors = None
        self.neighbors = None
        self.ideal_point = None

    def initialize(self):
        super().initialize()
        self.archive = np.empty((0, self.n_var), dtype=float)
        self.ideal_point = None
        self.weight_vectors = self._build_weight_vectors()
        self.neighbors = self._build_neighbors(self.weight_vectors)

    def respond_to_change(self, obj_func, t):
        if self.population is not None and self.fitness is not None:
            self._store_solutions(self.population)

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
        self._refresh_ideal_point()

    def _store_solutions(self, solutions: np.ndarray) -> None:
        """Store previous environment solutions using FIFO replacement."""

        if solutions is None or len(solutions) == 0:
            return
        stored = np.asarray(solutions, dtype=float)
        if stored.ndim == 1:
            stored = stored.reshape(1, -1)
        if len(self.archive) == 0:
            self.archive = stored.copy()
        else:
            self.archive = np.vstack([self.archive, stored])
        if len(self.archive) > self.archive_capacity:
            self.archive = self.archive[-self.archive_capacity :].copy()

    def _find_best_sol(self, obj_func, t):
        target = self.pop_size // 2
        if len(self.archive) == 0:
            return np.random.uniform(self.lb, self.ub, (target, self.n_var))
        archive_pop = self.archive
        if HAS_SKLEARN and len(archive_pop) >= 4:
            ns = min(self.n_svr_samples, max(target, 4))
            XT = np.random.uniform(self.lb, self.ub, (ns, self.n_var))
            YT = obj_func(XT, t)
            Y_est = np.zeros((len(archive_pop), self.n_obj))
            for oi in range(self.n_obj):
                try:
                    svr = SklearnSVR(kernel="rbf", C=1.0, epsilon=0.1, max_iter=500)
                    svr.fit(XT, YT[:, oi])
                    Y_est[:, oi] = svr.predict(archive_pop)
                except Exception:
                    Y_est[:, oi] = obj_func(archive_pop, t)[:, oi]
        else:
            Y_est = obj_func(archive_pop, t)
        fronts = self.fast_nds(Y_est)
        nd_idx = fronts[0] if fronts else list(range(min(target, len(archive_pop))))
        selected = archive_pop[nd_idx]
        if len(selected) > target:
            cd = self.crowding_distance(Y_est, nd_idx)
            selected = selected[np.argsort(-cd)[:target]]
        elif len(selected) < target:
            if len(selected) == 0:
                selected = np.random.uniform(self.lb, self.ub, (target, self.n_var))
            else:
                n_add = target - len(selected)
                noise_idx = np.random.choice(len(selected), n_add, replace=True)
                noisy = selected[noise_idx] + np.random.normal(0.0, 0.01, (n_add, self.n_var))
                selected = np.vstack([selected, np.clip(noisy, self.lb, self.ub)])
        return selected[:target]

    def _transfer(self, last_best, obj_func, t):
        n_trans = self.pop_size // 2
        if len(last_best) == 0:
            return np.random.uniform(self.lb, self.ub, (n_trans, self.n_var))
        target_samples = np.random.uniform(self.lb, self.ub, (self.pop_size, self.n_var))
        transferred = self.transfer_module.transfer(last_best, target_samples, transfer_size=n_trans)
        if transferred is None or len(transferred) == 0:
            idx = np.random.choice(len(last_best), n_trans, replace=(n_trans > len(last_best)))
            transferred = last_best[idx] + np.random.normal(0.0, 0.02, (n_trans, self.n_var))
        return np.clip(np.asarray(transferred, dtype=float)[:n_trans], self.lb, self.ub)

    def evolve_one_gen(self, obj_func, t):
        self._ensure_moead_state()
        for subproblem in range(self.pop_size):
            child = self._generate_offspring(subproblem)
            child_fit = self.evaluate(child[None, :], obj_func, t)[0]
            self.ideal_point = np.minimum(self.ideal_point, child_fit)
            self._update_neighborhood(subproblem, child, child_fit)

    def get_pareto_front(self):
        if self.fitness is None:
            return None
        fronts = self.fast_nds(self.fitness)
        return self.fitness[fronts[0]] if fronts else self.fitness

    def _ensure_moead_state(self) -> None:
        if self.weight_vectors is None or self.neighbors is None:
            self.weight_vectors = self._build_weight_vectors()
            self.neighbors = self._build_neighbors(self.weight_vectors)
        self._refresh_ideal_point()

    def _refresh_ideal_point(self) -> None:
        if self.fitness is None:
            return
        current_min = np.min(self.fitness, axis=0)
        if self.ideal_point is None:
            self.ideal_point = current_min
        else:
            self.ideal_point = np.minimum(self.ideal_point, current_min)

    def _build_weight_vectors(self) -> np.ndarray:
        if self.n_obj == 2:
            if self.pop_size == 1:
                return np.array([[0.5, 0.5]], dtype=float)
            grid = np.linspace(0.0, 1.0, self.pop_size)
            weights = np.column_stack([grid, 1.0 - grid])
            return np.clip(weights, 1e-6, 1.0)
        rng = np.random.default_rng(0)
        weights = rng.dirichlet(np.ones(self.n_obj), size=self.pop_size)
        return np.clip(weights, 1e-6, 1.0)

    def _build_neighbors(self, weights: np.ndarray) -> np.ndarray:
        pairwise = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        order = np.argsort(pairwise, axis=1)
        return order[:, : max(2, min(self.neighborhood_size, self.pop_size))]

    def _select_mating_pool(self, subproblem: int) -> np.ndarray:
        if np.random.rand() < self.neighbor_mating_prob:
            pool = self.neighbors[subproblem]
        else:
            pool = np.arange(self.pop_size)
        if len(pool) < 3:
            return np.arange(self.pop_size)
        return pool

    def _generate_offspring(self, subproblem: int) -> np.ndarray:
        pool = np.asarray(self._select_mating_pool(subproblem), dtype=int)
        pool = np.unique(pool)
        pool = pool[pool != subproblem]
        if len(pool) < 2:
            pool = np.setdiff1d(np.arange(self.pop_size), np.array([subproblem]), assume_unique=False)

        r1, r2 = np.random.choice(pool, 2, replace=False)
        target = self.population[subproblem]
        mutant = target + self.de_F * (self.population[r1] - self.population[r2])
        mutant = np.clip(mutant, self.lb, self.ub)

        trial = target.copy()
        j_rand = np.random.randint(self.n_var)
        crossover_mask = (np.random.rand(self.n_var) < self.de_CR)
        crossover_mask[j_rand] = True
        trial[crossover_mask] = mutant[crossover_mask]
        trial = self._polynomial_mutation(trial)
        return np.clip(trial, self.lb, self.ub)

    def _polynomial_mutation(self, vector: np.ndarray, eta_m: float = 20.0) -> np.ndarray:
        mutant = vector.copy()
        pm = 1.0 / max(1, self.n_var)
        for dim in range(self.n_var):
            if np.random.rand() >= pm:
                continue
            y = mutant[dim]
            yl = self.lb[dim]
            yu = self.ub[dim]
            if yu - yl <= 1e-14:
                continue
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            rnd = np.random.rand()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            mutant[dim] = np.clip(y + deltaq * (yu - yl), yl, yu)
        return mutant

    def _scalarize(self, fitness: np.ndarray, weight: np.ndarray) -> float:
        weight = np.maximum(weight, 1e-6)
        return float(np.max(weight * np.abs(fitness - self.ideal_point)))

    def _update_neighborhood(self, subproblem: int, child: np.ndarray, child_fit: np.ndarray) -> None:
        replacement_order = np.random.permutation(self.neighbors[subproblem])
        replaced = 0
        for neighbor_idx in replacement_order:
            current_value = self._scalarize(self.fitness[neighbor_idx], self.weight_vectors[neighbor_idx])
            child_value = self._scalarize(child_fit, self.weight_vectors[neighbor_idx])
            if child_value <= current_value:
                self.population[neighbor_idx] = child
                self.fitness[neighbor_idx] = child_fit
                replaced += 1
                if replaced >= self.max_replacements:
                    break


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
