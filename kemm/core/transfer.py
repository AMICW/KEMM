"""Subspace transfer operators used by KEMM after environment changes.

The implementations here build Grassmann geodesic flows and related transfer
mechanisms for moving candidate solutions across dynamic environments.
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


warnings.filterwarnings('ignore')


class GrassmannGeodesicFlow:
    """Constructs intermediate subspaces along a Grassmann geodesic between two fronts."""


    def __init__(self, n_subspaces: int = 5, jitter: float = 1e-6):
        """
        Args:
            n_subspaces: 中间子空间数量 p (论文默认 p=5)
            jitter: PCA 前注入的微扰量, 防止数值坍塌
        """
        self.n_subspaces = n_subspaces   # 论文 p=5
        self.jitter = jitter              # 数值稳定性微扰


    def compute_geodesic_bases(
        self,
        PS: np.ndarray,
        PT: np.ndarray
    ) -> List[np.ndarray]:
        """
        计算 Grassmann 测地线上的 p 个中间子空间基底
        
        理论基础:
            两子空间 PS, PT ∈ G(d, D) 之间的测地线为:
            φ(t) = PS·U·cos(t·Θ) + RS·V·sin(t·Θ)
            
            其中:
              - PS^T·PT = U·Σ·V^T   (SVD 分解)
              - Θ = arccos(clip(Σ, -1, 1))  (主角向量)
              - RS = PS 的正交补, 满足 PS^T·RS = 0, RS^T·RS = I
        
        Args:
            PS: 源子空间基底, shape (D, d), 列向量已正交归一化
            PT: 目标子空间基底, shape (D, d), 同上
        
        Returns:
            p 个中间子空间基底的列表, 每个 shape (D, d)
        
        来源: 论文 Process 3, Step 7
          "Construct the geodesic flow φ(k) = PS U1 Γ(k) − RS U2 Σ(k)"
        """
        D, d = PS.shape


        # ── Step 1: 计算子空间内积矩阵 ──
        # M = PS^T · PT,  shape (d, d)
        # M 的奇异值 σ_i = cos(θ_i), 其中 θ_i 是第 i 个主角
        M = PS.T @ PT  # (d, d)


        # ── Step 2: SVD 分解 — 核心步骤 ──
        # 原始代码完全跳过此步骤！
        # U, sigma, Vt 分别对应论文中的 U1, Σ (对角元素), U2^T
        try:
            U, sigma, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            # SVD 失败时的安全回退
            return self._fallback_bases(PS, PT)


        # 数值裁剪: cos(θ) ∈ [-1, 1]
        sigma = np.clip(sigma[:d], -1.0 + 1e-7, 1.0 - 1e-7)


        # ── Step 3: 计算主角 θ_i = arccos(σ_i) ──
        # θ_i 表示两子空间在第 i 个方向上的"夹角"
        # θ_i = 0 → 完全对齐; θ_i = π/2 → 正交
        theta = np.arccos(sigma)  # shape (d,)


        # ── Step 4: 计算正交补 RS — 原始代码缺失！ ──
        # RS 是 PT 中"无法被 PS 解释"的分量
        # 满足: PS^T·RS = 0 (与PS正交)
        # 来源: 论文 "RS is the orthogonal complement of PS in PT"
        RS = self._compute_orthogonal_complement(PS, PT, U, Vt, d, D)


        # ── Step 5: 生成 p 个中间子空间 ──
        # φ(t) = PS·U·diag(cos(t·θ)) + RS·V^T·diag(sin(t·θ))
        # t 均匀分布在 (0, 1) 之间: t_k = k/(p+1), k=1,...,p
        geodesic_bases = []
        V = Vt.T  # (d, d) → 对应论文中的 U2


        for k in range(1, self.n_subspaces + 1):
            t = k / (self.n_subspaces + 1)  # t ∈ (0, 1)


            # 对角矩阵 cos(t·θ) 和 sin(t·θ)
            cos_t = np.diag(np.cos(t * theta))   # (d, d)
            sin_t = np.diag(np.sin(t * theta))   # (d, d)


            # Grassmann 测地线上的中间基底
            # 公式: φ(t) = PS·U·cos(t·Θ) + RS·V·sin(t·Θ)
            phi_t = PS @ U[:, :d] @ cos_t + RS @ V[:d, :] @ sin_t  # (D, d)


            # 正交归一化确保是合法子空间基底
            phi_t, _ = np.linalg.qr(phi_t)
            phi_t = phi_t[:, :d]


            geodesic_bases.append(phi_t)


        return geodesic_bases


    def _compute_orthogonal_complement(
        self,
        PS: np.ndarray,
        PT: np.ndarray,
        U: np.ndarray,
        Vt: np.ndarray,
        d: int,
        D: int
    ) -> np.ndarray:
        """
        计算 PS 相对于 PT 的正交补空间 RS
        
        数学定义:
            RS 是 PT 中被 PS 正交化后的残差子空间
            RS = QR(PT - PS·(PS^T·PT))[0]  →  RS^T·PS = 0
        
        来源: 论文 Process 3, Step 7
          "RS is the othorgonal complement of PS"
          Gong et al. CVPR2012 Eq.(3)
        
        Args:
            PS: 源子空间基底 (D, d)
            PT: 目标子空间基底 (D, d)
            U, Vt: SVD 分解结果
            d: 子空间维度
            D: 环境空间维度
        
        Returns:
            RS: 正交补基底 (D, d), 满足 RS^T·PS ≈ 0
        """
        # 方法: 从 PT 中减去在 PS 上的投影
        # PT_residual = PT - PS·(PS^T·PT)
        PT_proj = PS @ (PS.T @ PT)    # PS 在 PT 方向的投影
        PT_residual = PT - PT_proj    # 残差 (D, d)


        # QR 分解正交化残差
        if PT_residual.shape[0] >= PT_residual.shape[1]:
            RS, _ = np.linalg.qr(PT_residual)
            RS = RS[:, :d]
        else:
            # 维度不足时用随机正交基补全
            RS = np.random.randn(D, d)
            RS = RS - PS @ (PS.T @ RS)  # 投影到 PS 的正交补
            RS, _ = np.linalg.qr(RS)
            RS = RS[:, :d]


        # 验证正交性 (调试用)
        # assert np.allclose(RS.T @ PS, 0, atol=1e-4), "RS 不满足正交补条件"


        return RS


    def _fallback_bases(
        self,
        PS: np.ndarray,
        PT: np.ndarray
    ) -> List[np.ndarray]:
        """
        SVD 失败时的安全回退: 使用线性插值 + QR
        仅作为极端数值情况下的保险措施
        """
        D, d = PS.shape
        bases = []
        for k in range(1, self.n_subspaces + 1):
            alpha = k / (self.n_subspaces + 1)
            P_mid = (1 - alpha) * PS + alpha * PT
            if P_mid.shape[0] >= P_mid.shape[1]:
                P_mid, _ = np.linalg.qr(P_mid)
                P_mid = P_mid[:, :d]
            bases.append(P_mid)
        return bases


    def project_and_transfer(
        self,
        source_data: np.ndarray,
        geodesic_bases: List[np.ndarray],
        source_mean: np.ndarray,
        target_mean: np.ndarray
    ) -> np.ndarray:
        """
        将源域数据沿测地流迁移到目标域
        
        实现论文 Process 3, Steps 8-11:
          "8: for x ∈ LastBestSolj do
           9:   Project x to φ(·) and get x̄;
           10:  x̂ = arg min_x ||x^T φ(·) − x̄||;
           11:  TransSol = x̂ ∪ TransSol;"
        
        数学形式:
          投影: x̄_k = (x - μ_s)^T · φ(t_k)   (降维)
          重建: x̂_k = x̄_k · φ(t_k)^T + μ_t   (升维, 偏移到目标域)
        
        Args:
            source_data: 源域数据 (N, D)
            geodesic_bases: p 个中间子空间基底列表
            source_mean: 源域均值 (D,)
            target_mean: 目标域均值 (D,)
        
        Returns:
            迁移后的数据 (N*p, D) — 所有中间子空间的映射合集
        """
        all_mapped = []
        S_centered = source_data - source_mean  # 中心化


        for phi_t in geodesic_bases:
            # Step 9: 投影到中间子空间 (降维)
            # x̄ = (x - μ_s)^T · φ(t),  shape (N, d)
            projected = S_centered @ phi_t


            # Step 10: 重建到目标空间 (升维 + 偏移)
            # x̂ = projected · φ(t)^T + μ_t,  shape (N, D)
            reconstructed = projected @ phi_t.T + target_mean


            all_mapped.append(reconstructed)


        return np.vstack(all_mapped)  # (N*p, D)




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  完整 Transfer 模块: 聚类 + PCA + SGF
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ManifoldTransferLearning:
    """Transfers archived elite solutions through learned subspace mappings."""


    def __init__(
        self,
        n_clusters: int = 4,
        n_subspaces: int = 5,
        var_threshold: float = 0.95,
        jitter: float = 1e-6
    ):
        """
        Args:
            n_clusters: LPCA 聚类数 L (论文 L=4)
            n_subspaces: 中间子空间数 p (论文 p=5)
            var_threshold: PCA 累积贡献率阈值, 用于动态确定子空间维度 d
            jitter: PCA 前注入的数值微扰量
        """
        self.n_clusters = n_clusters       # 论文 L=4
        self.n_subspaces = n_subspaces     # 论文 p=5
        self.var_threshold = var_threshold # 动态维度控制
        self.jitter = jitter               # 数值稳定性


        # 子模块
        self.sgf = GrassmannGeodesicFlow(
            n_subspaces=n_subspaces,
            jitter=jitter
        )


    def transfer(
        self,
        source_pop: np.ndarray,
        target_samples: np.ndarray,
        transfer_size: int
    ) -> Optional[np.ndarray]:
        """
        执行完整的流形迁移 (对应论文 Process 3 全流程)
        
        Args:
            source_pop: 源域精英解, shape (N_src, D)
                       - 在标准测试问题中 D = n_var
                       - 在船舶规划中 D = n_pts * 2 (展平的路径)
            target_samples: 目标域随机采样, shape (N_tgt, D)
            transfer_size: 需要迁移的个体数量 (论文中为 N/2)
        
        Returns:
            迁移后的个体, shape (transfer_size, D), 或 None (失败时)
        """
        try:
            return self._transfer_impl(source_pop, target_samples, transfer_size)
        except Exception as e:
            # 任何异常时安全回退到高斯微扰
            return self._gaussian_perturbation(source_pop, transfer_size)


    def _transfer_impl(self, source_pop, target_samples, transfer_size):
        """Transfer 的核心实现"""
        N_src, D = source_pop.shape


        # 安全检查
        if N_src < 4 or D < 2:
            return self._gaussian_perturbation(source_pop, transfer_size)


        # ── Step 2: LPCA 聚类 (论文用 LPCA, 此处用 KMeans 近似) ──
        # 来源: Process 3 Step 2
        #   "Clustering LastBestSol into L segments by LPCA"
        L = min(self.n_clusters, N_src // 2)
        if L < 1:
            L = 1


        labels = self._cluster(source_pop, L)


        all_transferred = []


        # ── Step 3-12: 对每个聚类执行 SGF ──
        for j in range(L):
            cluster_mask = (labels == j)
            S_j = source_pop[cluster_mask]  # 第 j 个聚类的源域数据


            if len(S_j) < 2:
                continue


            transferred_j = self._transfer_single_cluster(S_j, target_samples, D)
            if transferred_j is not None:
                all_transferred.append(transferred_j)


        if not all_transferred:
            return self._gaussian_perturbation(source_pop, transfer_size)


        # 合并所有聚类的迁移结果
        all_trans = np.vstack(all_transferred)


        # 按需采样 transfer_size 个个体
        replace = (transfer_size > len(all_trans))
        idx = np.random.choice(len(all_trans), transfer_size, replace=replace)
        return all_trans[idx]


    def _transfer_single_cluster(
        self,
        S_j: np.ndarray,
        T: np.ndarray,
        D: int
    ) -> Optional[np.ndarray]:
        """
        对单个聚类执行完整 SGF 迁移
        
        包含:
          - 动态 PCA 维度选择 (按累积贡献率 95%)
          - PCA 前注入微扰 (防止数值坍塌)
          - 正确的 Grassmann 测地流
          - d < 2 时退化为高斯微扰
        
        来源: Process 3 Steps 4-11
        """
        try:
            from sklearn.decomposition import PCA as SklearnPCA
        except ImportError:
            return self._gaussian_perturbation(S_j, len(S_j))


        N_j = len(S_j)
        max_d = min(self.n_subspaces, D - 1, N_j - 1, len(T) - 1)


        if max_d < 1:
            return self._gaussian_perturbation(S_j, N_j)


        # ── Step 4: PCA for source cluster → PS ──
        # 论文: "Use PCA for LastBestSolj to get PS"
        # 改进: PCA 前注入微扰, 防止奇异值过小导致的数值问题
        S_j_jittered = S_j + np.random.normal(0, self.jitter, S_j.shape)
        T_jittered = T + np.random.normal(0, self.jitter, T.shape)


        try:
            pca_s = SklearnPCA(n_components=max_d).fit(S_j_jittered)
            pca_t = SklearnPCA(n_components=max_d).fit(T_jittered)
        except Exception:
            return self._gaussian_perturbation(S_j, N_j)


        # ── 动态子空间维度: 按累积贡献率 95% 确定 d ──
        # 改进点: 原始代码使用固定 p=5, 可能对低维数据过拟合
        d_s = self._adaptive_dim(pca_s, max_d)
        d_t = self._adaptive_dim(pca_t, max_d)
        d = min(d_s, d_t, max_d)


        # ── d < 1 时退化为高斯微扰 ──
        # 一维子空间在动态 Pareto 前沿迁移中很常见，仍应允许 SGF 正常执行。
        if d < 1:
            return self._gaussian_perturbation(S_j, N_j * self.n_subspaces)


        # ── Step 4 (续): 提取子空间基底 ──
        PS = pca_s.components_[:d].T  # (D, d) — 源子空间, 列正交
        PT = pca_t.components_[:d].T  # (D, d) — 目标子空间


        # ── Step 5: 生成目标域采样 T (已由参数提供) ──


        # ── Step 7: 构建正确的 Grassmann 测地流 ──
        # 核心修复: 使用 SVD + 正交补计算真正的测地线
        geodesic_bases = self.sgf.compute_geodesic_bases(PS, PT)


        if not geodesic_bases:
            return self._gaussian_perturbation(S_j, N_j)


        # ── Steps 8-11: 投影并迁移 ──
        transferred = self.sgf.project_and_transfer(
            source_data=S_j,
            geodesic_bases=geodesic_bases,
            source_mean=pca_s.mean_,
            target_mean=pca_t.mean_
        )


        return transferred


    def _adaptive_dim(self, pca_model, max_d: int) -> int:
        """
        动态确定有效子空间维度 d
        
        策略: 找到最小的 d 使得累积贡献率 ≥ var_threshold
        
        改进点: 原始代码固定使用 p=5, 对低维数据可能过拟合。
               动态维度更适应不同问题的内在维度。
        
        Args:
            pca_model: 已拟合的 PCA 模型
            max_d: 最大允许维度
        
        Returns:
            有效维度 d (至少为 1)
        """
        if not hasattr(pca_model, 'explained_variance_ratio_'):
            return max_d


        cumvar = np.cumsum(pca_model.explained_variance_ratio_)
        # 找到第一个满足阈值的维度索引
        d = int(np.searchsorted(cumvar, self.var_threshold) + 1)
        return max(1, min(d, max_d))


    def _cluster(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        LPCA 聚类 (论文) / KMeans 近似实现
        
        来源: Process 3 Step 2
          "Clustering LastBestSol into L segments by LPCA"
        
        当 sklearn 可用时用 KMeans, 否则用随机分配
        """
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=3,
                max_iter=50,
                random_state=0
            )
            return kmeans.fit_predict(data)
        except Exception:
            # 安全回退: 随机分配
            return np.random.randint(0, n_clusters, len(data))


    @staticmethod
    def _gaussian_perturbation(data: np.ndarray, n: int, sigma: float = 0.05) -> np.ndarray:
        """
        高斯微扰作为安全回退
        当 SGF 失败或数据维度不足时使用
        """
        idx = np.random.choice(len(data), n, replace=(n > len(data)))
        result = data[idx].copy()
        noise_scale = sigma * (np.std(data) + 1e-8)
        result += np.random.normal(0, noise_scale, result.shape)
        return result




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  多源加权迁移: 从多个历史环境加权迁移 (KEMM 改进)
# ╚═══════════════════════════════════════════════════════════════════════════╝


class MultiSourceTransfer:
    """Ranks and combines multiple historical environments for transfer reuse."""


    def __init__(self, transfer_module: ManifoldTransferLearning):
        """
        Args:
            transfer_module: 底层 ManifoldTransferLearning 实例
        """
        self.transfer = transfer_module


    def transfer_from_sources(
        self,
        sources: List[dict],
        target_samples: np.ndarray,
        total_transfer: int
    ) -> np.ndarray:
        """
        从多个源域按相似度加权迁移
        
        Args:
            sources: 源域列表, 每个元素为:
                     {'data': np.ndarray(N, D), 
                      'similarity': float}
                     similarity 越大表示与当前环境越相似
            target_samples: 目标域采样 (N_tgt, D)
            total_transfer: 总迁移数量
        
        Returns:
            迁移结果 (total_transfer, D)
        """
        if not sources:
            return target_samples[:total_transfer]


        # 归一化相似度为权重
        sims = np.array([s.get('similarity', 1.0) for s in sources])
        sims = np.maximum(sims, 1e-8)
        weights = sims / sims.sum()


        all_transferred = []


        for src, w in zip(sources, weights):
            n_from = max(1, int(total_transfer * w))
            trans = self.transfer.transfer(
                source_pop=src['data'],
                target_samples=target_samples,
                transfer_size=n_from
            )
            if trans is not None:
                all_transferred.append(trans)


        if not all_transferred:
            # 全部失败时用目标域采样
            idx = np.random.choice(len(target_samples), total_transfer,
                                   replace=(total_transfer > len(target_samples)))
            return target_samples[idx]


        result = np.vstack(all_transferred)


        # 精确调整到目标数量
        replace = (total_transfer > len(result))
        idx = np.random.choice(len(result), total_transfer, replace=replace)
        return result[idx]
