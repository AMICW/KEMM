"""Base NSGA-II style dynamic multi-objective optimizer."""

from __future__ import annotations

import numpy as np


class BaseDMOEA:
    """Shared base class used by the benchmark algorithms."""

    def __init__(self, pop_size, n_var, n_obj, var_bounds):
        self.pop_size = pop_size
        self.n_var = n_var
        self.n_obj = n_obj
        self.lb, self.ub = var_bounds
        self.population = None
        self.fitness = None

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.n_var))

    def evaluate(self, pop, obj_func, t):
        result = obj_func(pop, t)
        return result if result.ndim > 1 else result.reshape(1, -1)

    def fast_nds(self, fitness):
        n = len(fitness)
        if n == 0:
            return []
            
        if fitness.shape[1] > self.n_obj:
            F = fitness[:, :self.n_obj]
            cv = fitness[:, self.n_obj:]
            cv_sum = np.sum(cv, axis=1)
        else:
            F = fitness
            cv_sum = np.zeros(n)

        leq = F[:, None, :] <= F[None, :, :]
        lt = F[:, None, :] < F[None, :, :]
        obj_dom = np.all(leq, axis=2) & np.any(lt, axis=2)
        
        cv_less = cv_sum[:, None] < cv_sum[None, :]
        cv_eq = np.abs(cv_sum[:, None] - cv_sum[None, :]) < 1e-9
        
        dom_matrix = cv_less | (cv_eq & obj_dom)
        
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
        f = fitness[front, :self.n_obj]
        dist = np.zeros(n)
        for m in range(f.shape[1]):
            order = np.argsort(f[:, m])
            dist[order[0]] = dist[order[-1]] = np.inf
            rng = f[order[-1], m] - f[order[0], m]
            if rng < 1e-14:
                continue
            dist[order[1:-1]] += (f[order[2:], m] - f[order[:-2], m]) / rng
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
        p1 = pop[idx[: N // 2 * 2 : 2]]
        p2 = pop[idx[1 : N // 2 * 2 : 2]]
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
                (1.0 / (2.0 - u * alpha + 1e-30)) ** (1.0 / (eta_c + 1)),
            )
            child1 = np.clip(0.5 * ((y1 + y2) - bq * (y2 - y1)), lb, ub)
            child2 = np.clip(0.5 * ((y1 + y2) + bq * (y2 - y1)), lb, ub)
            c1 = np.where(active, child1, p1)
            c2 = np.where(active, child2, p2)

        offspring = np.vstack([c1, c2])
        if len(offspring) < N:
            offspring = np.vstack([offspring, pop[np.random.choice(N, N - len(offspring))]])
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
                dq[left] = (2 * u_m[left] + (1 - 2 * u_m[left]) * (xy_l ** (eta_m + 1))) ** (
                    1 / (eta_m + 1)
                ) - 1.0
            if np.any(~left):
                xy_r = 1.0 - d2[~left]
                dq[~left] = 1.0 - (
                    2 * (1 - u_m[~left]) + 2 * (u_m[~left] - 0.5) * (xy_r ** (eta_m + 1))
                ) ** (1 / (eta_m + 1))
            rng = ub[cols] - lb[cols]
            offspring[mut_mask] = np.clip(val + dq * rng, lb[cols], ub[cols])

        return offspring

    def evolve_one_gen(self, obj_func, t):
        offspring = self.sbx_pm_batch(self.population)
        off_fit = self.evaluate(offspring, obj_func, t)
        merged = np.vstack([self.population, offspring])
        merged_f = np.vstack([self.fitness, off_fit])
        self.population, self.fitness = self.env_selection(merged, merged_f, self.pop_size)

    def get_pareto_front(self):
        fronts = self.fast_nds(self.fitness)
        return self.fitness[fronts[0]] if fronts else self.fitness

    def respond_to_change(self, obj_func, t):
        self.initialize()
        self.fitness = self.evaluate(self.population, obj_func, t)


__all__ = ["BaseDMOEA"]
