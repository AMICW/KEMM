"""KEMM algorithm implementation.

这个文件只负责 KEMM 主流程编排，不再把 benchmark 强化逻辑和通用核心混在一起。

当前设计目标：

1. 保持与旧接口兼容
2. 让 benchmark prior 通过 adapter 注入，而不是写死在主类里
3. 让参数、变化诊断和候选池来源显式化，方便后续继续改算法结构
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kemm.adapters import BenchmarkPriorAdapter
from kemm.algorithms.base import BaseDMOEA
from kemm.core.adaptive import AdaptiveOperatorSelector, ParetoFrontDriftDetector
from kemm.core.drift import ParetoFrontDriftPredictor
from kemm.core.memory import VAECompressedMemory
from kemm.core.transfer import ManifoldTransferLearning, MultiSourceTransfer
from kemm.core.types import KEMMChangeDiagnostics, KEMMConfig


class KEMM_DMOEA_Improved(BaseDMOEA):
    """KEMM 主算法。

    该类负责把四个核心机制组织成统一的环境变化响应流程：

    - adaptive: 决定四类策略各分配多少样本
    - memory: 从历史环境中检索相似精英
    - drift: 预测前沿随时间的漂移方向
    - transfer: 从相似源域迁移结构信息

    需要特别说明：

    - benchmark prior 现在通过 adapter 注入
    - ship 主线应关闭 benchmark prior
    - 主流程内部尽量不再依赖散落的魔法数字
    """

    def __init__(
        self,
        *args,
        benchmark_aware_prior: bool | None = None,
        config: KEMMConfig | None = None,
        benchmark_adapter: BenchmarkPriorAdapter | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        base_config = config or KEMMConfig()
        prior_enabled = base_config.benchmark_aware_prior if benchmark_aware_prior is None else benchmark_aware_prior
        self.config = replace(
            base_config,
            pop_size=self.pop_size,
            n_var=self.n_var,
            n_obj=self.n_obj,
            benchmark_aware_prior=prior_enabled,
        )
        self.benchmark_aware_prior = self.config.benchmark_aware_prior
        self._benchmark_adapter = benchmark_adapter
        if self._benchmark_adapter is None and self.benchmark_aware_prior:
            self._benchmark_adapter = BenchmarkPriorAdapter()

        self._transfer_module = ManifoldTransferLearning(
            n_clusters=self.config.transfer_n_clusters,
            n_subspaces=self.config.transfer_n_subspaces,
            var_threshold=self.config.transfer_var_threshold,
            jitter=self.config.transfer_jitter,
        )
        self._multi_src_transfer = MultiSourceTransfer(self._transfer_module)

        self._vae_memory = VAECompressedMemory(
            input_dim=self.n_var,
            latent_dim=min(self.config.latent_dim_cap, max(1, self.n_var // 2)),
            hidden_dim=self.config.memory_hidden_dim,
            capacity=self.config.memory_capacity,
            beta=self.config.memory_beta,
            online_epochs=self.config.memory_online_epochs,
        )

        self._operator_selector = AdaptiveOperatorSelector(
            window=self.config.reward_window,
            c=self.config.exploration_c,
            temperature=self.config.temperature,
            min_ratio=self.config.min_operator_ratio,
        )
        self._drift_detector = ParetoFrontDriftDetector(window=self.config.drift_window)
        self._drift_predictor = ParetoFrontDriftPredictor(
            feature_dim=self.config.drift_feature_dim,
            max_history=self.config.drift_history,
            lengthscale=self.config.gp_lengthscale,
            noise_var=self.config.gp_noise_var,
        )

        self._prev_igd = None
        self._time_step = 0
        self._centroid_history: list[np.ndarray] = []
        self.last_change_diagnostics: KEMMChangeDiagnostics | None = None
        self.change_diagnostics_history: list[KEMMChangeDiagnostics] = []

    def respond_to_change(self, obj_func, t):
        """响应环境变化。

        核心流程固定为：

        1. 归档上一环境精英和前沿特征
        2. 获取 adaptive 基础比例
        3. 根据变化强度与可迁移性修正比例
        4. 分配 memory / prediction / transfer / reinit 样本数
        5. 构造候选池
        6. 统一评价并做环境选择
        7. 估计响应质量并更新 reward
        8. 记录本次变化诊断信息
        """

        previous_population = None if self.population is None else self.population.copy()
        archive_state = self._archive_current_environment()
        self._time_step += 1

        change_magnitude = self._drift_detector.get_change_magnitude()
        transferability = self._drift_detector.predict_transferability()
        prediction_confidence = self._drift_predictor.get_prediction_confidence()

        ratios, _ = self._operator_selector.get_ratios()
        ratios = self._adjust_operator_ratios(ratios)
        n_memory, n_predict, n_transfer, n_reinit = self._allocate_operator_counts(ratios)

        parts: list[np.ndarray] = []
        actual_counts = {
            "memory": 0,
            "prediction": 0,
            "transfer": 0,
            "prior": 0,
            "elite": 0,
            "previous": 0,
            "reinit": 0,
        }

        memory_candidates = self._build_memory_candidates(obj_func, t, n_memory)
        if memory_candidates is not None:
            parts.append(memory_candidates)
            actual_counts["memory"] = len(memory_candidates)
        else:
            n_reinit += n_memory

        prediction_candidates = self._build_prediction_candidates(
            obj_func=obj_func,
            t=t,
            n_predict=n_predict,
            elite_archive=archive_state["elite_archive"],
        )
        if prediction_candidates is not None:
            parts.append(prediction_candidates)
            actual_counts["prediction"] = len(prediction_candidates)

        transfer_parts = self._build_transfer_candidates(obj_func=obj_func, t=t, n_transfer=n_transfer)
        if transfer_parts:
            parts.extend(transfer_parts)
            actual_counts["transfer"] = sum(len(part) for part in transfer_parts)
        else:
            n_reinit += n_transfer

        prior_candidates = self._build_prior_candidates(obj_func, t)
        if prior_candidates is not None:
            parts.append(prior_candidates)
            actual_counts["prior"] = len(prior_candidates)

        elite_parts = self._build_elite_candidates(archive_state["elite_archive"])
        if elite_parts:
            parts.extend(elite_parts)
            actual_counts["elite"] = sum(len(part) for part in elite_parts)

        previous_candidates = self._build_previous_population_candidates(previous_population)
        if previous_candidates is not None:
            parts.append(previous_candidates)
            actual_counts["previous"] = len(previous_candidates)

        reinit_parts = self._build_reinitialization_candidates(parts, n_reinit)
        if reinit_parts:
            parts.extend(reinit_parts)
            actual_counts["reinit"] = sum(len(part) for part in reinit_parts)

        candidate_pop = np.clip(np.vstack(parts), self.lb, self.ub)
        candidate_fit = self.evaluate(candidate_pop, obj_func, t)
        self.population, self.fitness = self.env_selection(candidate_pop, candidate_fit, self.pop_size)

        response_quality = self._estimate_response_quality()
        self._operator_selector.update_with_quality(response_quality)

        fronts = self.fast_nds(self.fitness)
        selected_front_size = len(fronts[0]) if fronts else 0
        requested_counts = {
            "memory": n_memory,
            "prediction": n_predict,
            "transfer": n_transfer,
            "reinit": n_reinit,
            "prior": 0 if not self.benchmark_aware_prior else max(self.config.prior_min_samples, int(self.pop_size * self.config.prior_sample_fraction)),
        }
        operator_ratios = {
            "memory": float(ratios[0]),
            "prediction": float(ratios[1]),
            "transfer": float(ratios[2]),
            "reinit": float(ratios[3]),
        }
        self.last_change_diagnostics = KEMMChangeDiagnostics(
            time_step=self._time_step,
            change_time=float(t),
            operator_ratios=operator_ratios,
            requested_counts=requested_counts,
            actual_counts=actual_counts,
            candidate_pool_size=int(len(candidate_pop)),
            prediction_confidence=float(prediction_confidence),
            change_magnitude=float(change_magnitude),
            transferability=float(transferability),
            response_quality=float(response_quality),
            selected_front_size=int(selected_front_size),
        )
        self.change_diagnostics_history.append(self.last_change_diagnostics)

    def _archive_current_environment(self) -> dict[str, np.ndarray | None]:
        """归档上一环境的精英信息和前沿特征。"""

        archive_state: dict[str, np.ndarray | None] = {"elite_archive": None}
        if self.population is None or self.fitness is None:
            return archive_state

        fronts = self.fast_nds(self.fitness)
        elite_idx = fronts[0]
        elites = self.population[elite_idx]
        elite_fit = self.fitness[elite_idx]
        archive_state["elite_archive"] = elites.copy()

        head_dim = min(self.config.probe_dim // 2, self.n_var)
        fingerprint = np.concatenate(
            [
                np.mean(self.population, axis=0)[:head_dim],
                np.mean(elite_fit, axis=0),
                np.std(elite_fit, axis=0),
            ]
        )
        if len(fingerprint) < self.config.probe_dim:
            fingerprint = np.pad(fingerprint, (0, self.config.probe_dim - len(fingerprint)))
        else:
            fingerprint = fingerprint[: self.config.probe_dim]

        self._vae_memory.store(elites, elite_fit, fingerprint)
        self._centroid_history.append(np.mean(self.population, axis=0))
        if len(self._centroid_history) > self.config.drift_history:
            self._centroid_history.pop(0)

        self._drift_detector.update(elite_fit)
        self._drift_predictor.update(elite_fit, float(self._time_step))
        return archive_state

    def _current_feature_probe(self) -> np.ndarray:
        """构造当前环境的轻量探针向量。"""

        if self.population is None:
            return np.zeros(self.config.probe_dim)

        head_dim = min(self.config.probe_dim // 2, self.n_var)
        probe = np.concatenate(
            [
                np.mean(self.population, axis=0)[:head_dim],
                np.zeros(self.config.probe_dim - head_dim),
            ]
        )
        if len(probe) < self.config.probe_dim:
            probe = np.pad(probe, (0, self.config.probe_dim - len(probe)))
        return probe[: self.config.probe_dim]

    def _adjust_operator_ratios(self, base_ratios: np.ndarray) -> np.ndarray:
        """结合环境变化强度对 adaptive 基础比例做二次修正。"""

        change_mag = self._drift_detector.get_change_magnitude()
        transferability = self._drift_detector.predict_transferability()
        ratios = base_ratios.copy()
        ratios[0] += self.config.memory_transferability_bonus * transferability
        ratios[2] += self.config.transfer_transferability_bonus * transferability
        ratios[1] += self.config.prediction_stability_bonus * (1.0 - change_mag)
        ratios[1] -= self.config.prediction_change_penalty * change_mag
        ratios[2] -= self.config.transfer_change_penalty * change_mag
        ratios[3] += self.config.reinit_change_bonus * change_mag
        ratios = np.maximum(ratios, self.config.min_operator_ratio)
        return ratios / ratios.sum()

    def _allocate_operator_counts(self, ratios: np.ndarray) -> tuple[int, int, int, int]:
        """把连续比例映射为整数候选数量。"""

        n_memory = int(self.pop_size * ratios[0])
        n_predict = int(self.pop_size * ratios[1])
        n_transfer = int(self.pop_size * ratios[2])
        n_reinit = self.pop_size - n_memory - n_predict - n_transfer
        return n_memory, n_predict, n_transfer, n_reinit

    def _build_memory_candidates(self, obj_func, t: float, n_memory: int) -> np.ndarray | None:
        """从压缩记忆中检索候选。"""

        if n_memory <= 0 or len(self._vae_memory) == 0:
            return None

        retrieved = self._vae_memory.retrieve(
            self._current_feature_probe(),
            top_k=self.config.memory_top_k,
            n_decode=max(n_memory, self.config.prediction_min_pool),
        )
        if not retrieved:
            return None

        memory_pool = []
        for item in retrieved:
            solutions = np.asarray(item["solutions"], dtype=float)
            if len(solutions) == 0:
                continue
            keep = int(max(self.config.memory_min_keep, np.ceil(n_memory * max(0.5, float(item["similarity"])))) )
            memory_pool.append(solutions[: min(len(solutions), keep)])

        if not memory_pool:
            return None

        memory_pool = np.clip(np.vstack(memory_pool), self.lb, self.ub)
        quick_fit = obj_func(memory_pool, t)
        selected, _ = self.env_selection(memory_pool, quick_fit, min(n_memory, len(memory_pool)))
        return selected

    def _build_prediction_candidates(
        self,
        obj_func,
        t: float,
        n_predict: int,
        elite_archive: np.ndarray | None,
    ) -> np.ndarray | None:
        """构造预测型候选。"""

        if n_predict <= 0:
            return None

        confidence = self._drift_predictor.get_prediction_confidence()
        pool_size = max(n_predict * self.config.prediction_pool_multiplier, self.config.prediction_min_pool)
        predicted = self._linear_predict(pool_size)
        if confidence <= self.config.prediction_confidence_threshold or len(self._centroid_history) < 2:
            return np.clip(predicted[:n_predict], self.lb, self.ub)

        predicted_feature, gp_candidates = self._drift_predictor.predict_next(
            float(self._time_step),
            n_samples=max(n_predict, self.config.prediction_min_pool),
            var_bounds=(self.lb, self.ub),
            anchors=elite_archive if elite_archive is not None and len(elite_archive) > 0 else self.population,
        )
        if gp_candidates is None:
            return np.clip(predicted[:n_predict], self.lb, self.ub)

        pred_pool = [predicted, np.clip(gp_candidates, self.lb, self.ub)]
        if elite_archive is not None and len(elite_archive) > 0:
            elite_center = np.mean(elite_archive, axis=0)
            elite_spread = np.maximum(
                np.std(elite_archive, axis=0),
                self.config.elite_spread_floor_ratio * (self.ub - self.lb),
            )
            elite_samples = elite_center + np.random.randn(max(n_predict, self.config.prediction_min_pool), self.n_var) * elite_spread
            pred_pool.append(np.clip(elite_samples, self.lb, self.ub))

        pred_pool = np.clip(np.vstack(pred_pool), self.lb, self.ub)
        pred_fit = obj_func(pred_pool, t)
        selected = self._select_prediction_candidates(
            pred_pool=pred_pool,
            pred_fit=pred_fit,
            predicted_feature=predicted_feature,
            n_select=n_predict,
        )
        return selected

    def _build_transfer_candidates(self, obj_func, t: float, n_transfer: int) -> list[np.ndarray]:
        """通过多源迁移生成候选。"""

        if n_transfer <= 0 or len(self._vae_memory) < 2:
            return []

        retrieved = self._vae_memory.retrieve(
            self._current_feature_probe(),
            top_k=self.config.memory_top_k,
            n_decode=40,
        )
        if not retrieved:
            return []

        if self.population is not None:
            target_samples = self.population[: min(self.config.transfer_target_sample_size, len(self.population))]
        else:
            target_samples = np.random.uniform(self.lb, self.ub, (self.config.transfer_target_sample_size, self.n_var))

        sources = [{"data": item["solutions"], "similarity": item["similarity"]} for item in retrieved]
        transferred = self._multi_src_transfer.transfer_from_sources(sources, target_samples, n_transfer)
        transferred = np.clip(transferred, self.lb, self.ub)
        transfer_jitter = self.config.transfer_jitter_ratio * (self.ub - self.lb)
        diversified = transferred + np.random.randn(*transferred.shape) * transfer_jitter
        return [transferred, np.clip(diversified, self.lb, self.ub)]

    def _build_prior_candidates(self, obj_func, t: float) -> np.ndarray | None:
        """通过 benchmark adapter 生成结构先验候选。"""

        if not self.benchmark_aware_prior or self._benchmark_adapter is None:
            return None

        n_prior = max(self.config.prior_min_samples, int(self.pop_size * self.config.prior_sample_fraction))
        prior_candidates = self._benchmark_adapter.generate(
            obj_func=obj_func,
            t=t,
            n_samples=n_prior,
            lb=self.lb,
            ub=self.ub,
            n_var=self.n_var,
        )
        if prior_candidates is None or len(prior_candidates) == 0:
            return None
        return np.clip(prior_candidates, self.lb, self.ub)

    def _build_elite_candidates(self, elite_archive: np.ndarray | None) -> list[np.ndarray]:
        """保留上一环境精英，并在精英附近补样。"""

        if elite_archive is None or len(elite_archive) == 0:
            return []
        keep_n = min(max(self.config.elite_keep_min, int(self.pop_size * self.config.elite_keep_fraction)), len(elite_archive))
        keep_idx = np.random.choice(len(elite_archive), keep_n, replace=(len(elite_archive) < keep_n))
        kept_elites = elite_archive[keep_idx]
        jitter_scale = self.config.elite_jitter_ratio * (self.ub - self.lb)
        perturbed = kept_elites + np.random.randn(*kept_elites.shape) * jitter_scale
        return [np.clip(kept_elites, self.lb, self.ub), np.clip(perturbed, self.lb, self.ub)]

    def _build_previous_population_candidates(self, previous_population: np.ndarray | None) -> np.ndarray | None:
        """回灌上一代部分种群。"""

        if previous_population is None or len(previous_population) == 0:
            return None
        prev_keep = min(max(self.config.previous_keep_min, int(self.pop_size * self.config.previous_keep_fraction)), len(previous_population))
        prev_idx = np.random.choice(len(previous_population), prev_keep, replace=(len(previous_population) < prev_keep))
        return np.clip(previous_population[prev_idx], self.lb, self.ub)

    def _build_reinitialization_candidates(self, parts: list[np.ndarray], n_reinit: int) -> list[np.ndarray]:
        """用随机样本补齐候选池。"""

        n_have = sum(len(part) for part in parts) if parts else 0
        n_need = max(self.pop_size - n_have, 0)
        if n_need <= 0:
            return []
        return [np.random.uniform(self.lb, self.ub, (n_need, self.n_var))]

    def _select_prediction_candidates(
        self,
        *,
        pred_pool: np.ndarray,
        pred_fit: np.ndarray,
        predicted_feature: np.ndarray,
        n_select: int,
    ) -> np.ndarray:
        """按预测的前沿位置优先保留更匹配的候选。"""

        n_keep = min(n_select, len(pred_pool))
        if n_keep <= 0:
            return pred_pool[:0]

        target_dim = min(self.n_obj, 2, pred_fit.shape[1], len(predicted_feature))
        if target_dim <= 0:
            selected, _ = self.env_selection(pred_pool, pred_fit, n_keep)
            return selected

        target = np.asarray(predicted_feature[:target_dim], dtype=float)
        objective_distance = np.linalg.norm(pred_fit[:, :target_dim] - target, axis=1)
        fronts = self.fast_nds(pred_fit)
        selected_idx: list[int] = []
        for front in fronts:
            ordered_front = sorted(front, key=lambda idx: (objective_distance[idx], float(np.sum(pred_fit[idx]))))
            if len(selected_idx) + len(ordered_front) <= n_keep:
                selected_idx.extend(ordered_front)
            else:
                selected_idx.extend(ordered_front[: n_keep - len(selected_idx)])
                break

        if len(selected_idx) < n_keep:
            remaining = [idx for idx in np.argsort(objective_distance) if int(idx) not in selected_idx]
            selected_idx.extend(int(idx) for idx in remaining[: n_keep - len(selected_idx)])
        return pred_pool[np.asarray(selected_idx[:n_keep], dtype=int)]

    def _estimate_response_quality(self) -> float:
        """估计当前变化响应质量。"""

        pf_fit = self.get_pareto_front()
        if len(pf_fit) > 1:
            from scipy.spatial.distance import cdist as _cdist

            distances = _cdist(pf_fit, pf_fit)
            np.fill_diagonal(distances, np.inf)
            return float(np.mean(np.min(distances, axis=1)))
        if self.fitness is not None:
            return float(np.mean(self.fitness[:, 0]))
        return 1.0

    def _linear_predict(self, n_pred: int) -> np.ndarray:
        """用质心历史做低成本线性/二阶外推。"""

        if len(self._centroid_history) < 2:
            return np.random.uniform(self.lb, self.ub, (n_pred, self.n_var))

        centroids = np.array(self._centroid_history)
        if len(centroids) >= 3:
            velocity = centroids[-1] - centroids[-2]
            acceleration = 0.5 * (centroids[-1] - 2 * centroids[-2] + centroids[-3])
            pred_center = np.clip(centroids[-1] + velocity + acceleration, self.lb, self.ub)
        else:
            velocity = centroids[-1] - centroids[-2]
            pred_center = np.clip(centroids[-1] + velocity, self.lb, self.ub)

        spread = np.maximum(np.std(centroids[-3:], axis=0), 0.05)
        pred_pop = pred_center + np.random.randn(n_pred, self.n_var) * spread
        return np.clip(pred_pop, self.lb, self.ub)

    def _problem_aware_candidates(self, obj_func, t: float, n_samples: int) -> np.ndarray | None:
        """兼容旧接口，内部转发到 benchmark prior adapter。"""

        if self._benchmark_adapter is None:
            return None
        return self._benchmark_adapter.generate(
            obj_func=obj_func,
            t=t,
            n_samples=n_samples,
            lb=self.lb,
            ub=self.ub,
            n_var=self.n_var,
        )

    def get_last_change_diagnostics(self) -> KEMMChangeDiagnostics | None:
        """返回最近一次环境变化的结构化诊断信息。"""

        return self.last_change_diagnostics


__all__ = ["KEMM_DMOEA_Improved"]
