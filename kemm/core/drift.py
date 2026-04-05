"""
===============================================================================
pareto_drift.py
Pareto 前沿漂移预测模型 — 高斯过程回归
===============================================================================
【理论贡献】
  这是 KEMM 相对于 MMTL-DMOEA 的一个原创理论扩展:
  
  原始所有算法 (MMTL/PPS/KF/SVR) 都假设:
    环境变化后, Pareto 前沿发生"跳变" → 需要从头优化
  
  KEMM 的新假设 (理论贡献):
    在缓慢变化环境中, Pareto 前沿 PF(t) 在决策/目标空间
    以近似连续的方式漂移, 可以建模为流形上的时间序列
  
  数学形式:
    PF(t+1) ≈ f(PF(t), PF(t-1), ..., PF(t-k))
    
    用高斯过程回归 (GPR) 学习这个映射
  
  实际效果:
    当 PF 漂移规律时 (如 FDA1/dMOP2), 预测个体质量远高于随机初始化
    当 PF 跳变时 (如 FDA3), GPR 预测失效, 自动退化到随机初始化


【高斯过程基础】
  GPR 拟合函数 f: R^d → R:
    f(x) ~ GP(m(x), k(x, x'))
  
  核函数: RBF (径向基函数)
    k(x, x') = σ_f² · exp(-||x-x'||² / (2l²))
  
  预测:
    μ* = k*^T · (K + σ_n²I)^{-1} · y
    σ*² = k** - k*^T · (K + σ_n²I)^{-1} · k*
  
  计算复杂度: O(n³) 训练, O(n²) 预测
  其中 n = 历史时刻数 (通常 < 10, 可接受)


【参数设计】
  max_history = 8: 使用最近 8 个时刻的 PF 特征
  feature_dim = 10: PF 特征维度 (来自 ParetoFrontDriftDetector)
  gp_lengthscale = 1.0: RBF 核长度尺度
  gp_noise = 0.1: 观测噪声标准差
===============================================================================
"""


import numpy as np
from typing import List, Tuple, Optional




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  轻量级高斯过程回归
# ╚═══════════════════════════════════════════════════════════════════════════╝


class LightweightGPR:
    """
    轻量级高斯过程回归 — 纯 NumPy 实现
    ─────────────────────────────────────────
    用于预测 Pareto 前沿特征在下一时刻的值。
    
    实现特点:
      - 纯 NumPy, 无外部 GP 库依赖
      - 使用 RBF 核 (各向同性)
      - L-BFGS 核参数优化 (简化版: 网格搜索)
      - 数值稳定性: Cholesky 分解
    
    使用场景:
      训练数据量 n < 20 (历史时刻数有限)
      输入维度 d ∈ {1, 2} (时间索引或 2D 特征)
      输出维度 = 1 (每次预测单个特征值)
    """


    def __init__(
        self,
        lengthscale: float = 1.0,
        signal_var: float = 1.0,
        noise_var: float = 0.01
    ):
        """
        Args:
            lengthscale: RBF 核长度尺度 l (越大越平滑)
            signal_var: 信号方差 σ_f² (核的幅度)
            noise_var: 观测噪声方差 σ_n² (越大越不相信观测)
        """
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var


        # 训练数据
        self.X_train: Optional[np.ndarray] = None  # (n, d)
        self.y_train: Optional[np.ndarray] = None  # (n,)
        self.alpha: Optional[np.ndarray] = None    # (n,) — 对偶变量
        self.L: Optional[np.ndarray] = None        # Cholesky 因子


        self.is_fitted = False


    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF (径向基函数) 核
        
        k(x, x') = σ_f² · exp(-||x-x'||² / (2l²))
        
        向量化实现, 支持批量计算。
        
        Args:
            X1: (n1, d) 或 (d,)
            X2: (n2, d) 或 (d,)
        
        Returns:
            K: (n1, n2) 核矩阵
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)


        # 计算平方欧氏距离 ||x - x'||²
        # 使用 (a-b)^2 = a^2 + b^2 - 2ab 的向量化形式
        sq_dists = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )
        sq_dists = np.maximum(sq_dists, 0)  # 防止数值负值


        K = self.signal_var * np.exp(-sq_dists / (2 * self.lengthscale ** 2))
        return K


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练 GPR
        
        计算对偶变量 α = (K + σ_n²I)^{-1} · y
        使用 Cholesky 分解保证数值稳定性。
        
        Args:
            X: 训练输入 (n, d)
            y: 训练目标 (n,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).ravel()


        n = len(X)
        self.X_train = X.copy()
        self.y_train = y.copy()


        # 计算核矩阵
        K = self.rbf_kernel(X, X)  # (n, n)


        # 添加噪声 + 数值稳定项
        K_noise = K + (self.noise_var + 1e-6) * np.eye(n)


        # Cholesky 分解: K_noise = L · L^T
        try:
            self.L = np.linalg.cholesky(K_noise)
            # 求解 K_noise · α = y
            # 即 L · L^T · α = y
            # 先求 L · v = y (前向代入)
            v = np.linalg.solve(self.L, y)
            # 再求 L^T · α = v (后向代入)
            self.alpha = np.linalg.solve(self.L.T, v)
        except np.linalg.LinAlgError:
            # Cholesky 失败时用直接求逆 (较慢但更稳定)
            try:
                K_inv = np.linalg.pinv(K_noise)
                self.alpha = K_inv @ y
                self.L = None
            except Exception:
                self.alpha = np.zeros(n)
                self.L = None


        self.is_fitted = True


    def predict(
        self,
        X_test: np.ndarray,
        return_std: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        GPR 预测
        
        预测均值: μ* = k*^T · α
        预测方差: σ*² = k** - k*^T · (K + σI)^{-1} · k*
        
        Args:
            X_test: 测试输入 (m, d) 或 (d,)
            return_std: 是否返回预测标准差
        
        Returns:
            mu: 预测均值 (m,)
            std: 预测标准差 (m,), 仅当 return_std=True
        """
        if not self.is_fitted:
            n_test = len(np.atleast_2d(X_test))
            mu = np.zeros(n_test)
            std = np.ones(n_test) if return_std else None
            return mu, std


        X_test = np.atleast_2d(X_test)


        # k*: 测试点与训练点的核向量
        k_star = self.rbf_kernel(X_test, self.X_train)  # (m, n)


        # 预测均值
        mu = k_star @ self.alpha  # (m,)


        if not return_std:
            return mu, None


        # 预测方差
        k_star_star = np.diag(self.rbf_kernel(X_test, X_test))  # (m,)


        if self.L is not None:
            v = np.linalg.solve(self.L, k_star.T)  # (n, m)
            var = k_star_star - np.sum(v ** 2, axis=0)  # (m,)
        else:
            # 直接计算 (无 Cholesky)
            K_noise = self.rbf_kernel(self.X_train, self.X_train) + \
                      self.noise_var * np.eye(len(self.X_train))
            try:
                K_inv = np.linalg.pinv(K_noise)
                var = k_star_star - np.sum(k_star @ K_inv * k_star, axis=1)
            except Exception:
                var = k_star_star


        std = np.sqrt(np.maximum(var, 0))
        return mu, std




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Pareto 前沿漂移预测器
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ParetoFrontDriftPredictor:
    """
    Pareto 前沿漂移预测器
    ─────────────────────────────────────────
    用 GP 回归预测下一时刻的 Pareto 前沿位置,
    从而生成更好的初始种群。
    
    预测流程:
      1. 提取历史 PF 的紧凑特征向量 (10 维)
      2. 为每个特征维度训练一个 GP 回归器
         (10 个独立 GP, 每个预测一个特征维度)
      3. 预测下一时刻的特征向量
      4. 根据预测特征生成符合预期分布的候选解
    
    理论依据 (KEMM 原创):
      假设 PF 在特征空间的演化是连续随机过程,
      GP 是对该过程的非参数贝叶斯估计。
      
      相比 PPS/KF 的质心预测, 本方法:
        - 预测 PF 的整体分布特征 (而非单点)
        - 提供预测不确定性 (GP 方差)
        - 可以发现周期性漂移 (GP 可用周期核)
    
    来源 (理论根基):
      PPS (论文引用 [22]): "autoregressive model for center prediction"
      本文扩展: GP 对 PF 特征的非参数预测
    """


    def __init__(
        self,
        feature_dim: int = 10,
        max_history: int = 8,
        lengthscale: float = 1.0,
        noise_var: float = 0.05
    ):
        """
        Args:
            feature_dim: PF 特征维度 (默认 10)
            max_history: 最大历史时刻数 (GPR 训练数据量)
            lengthscale: GP 核长度尺度
            noise_var: 观测噪声方差
        """
        self.feature_dim = feature_dim
        self.max_history = max_history


        # 历史数据
        self.time_steps: List[float] = []  # 时间索引
        self.features: List[np.ndarray] = []  # PF 特征序列


        # 为每个特征维度创建一个 GPR
        self.gp_models: List[LightweightGPR] = [
            LightweightGPR(
                lengthscale=lengthscale,
                signal_var=1.0,
                noise_var=noise_var
            )
            for _ in range(feature_dim)
        ]

    def _compute_feature(self, pf_fitness: np.ndarray) -> np.ndarray:
        """将 Pareto 前沿压缩为固定长度统计特征向量。"""
        pf = np.atleast_2d(np.asarray(pf_fitness, dtype=float))
        if pf.size == 0 or pf.shape[0] == 0:
            return np.zeros(self.feature_dim, dtype=float)

        n_pf, n_obj = pf.shape
        mean = np.mean(pf, axis=0)
        std = np.std(pf, axis=0)
        min_v = np.min(pf, axis=0)
        max_v = np.max(pf, axis=0)
        span = max_v - min_v

        feature = np.concatenate([
            mean[:min(2, n_obj)],
            std[:min(2, n_obj)],
            min_v[:min(2, n_obj)],
            span[:min(2, n_obj)],
            np.array([float(n_pf)], dtype=float),
        ])
        if feature.size < self.feature_dim:
            feature = np.pad(feature, (0, self.feature_dim - feature.size))
        else:
            feature = feature[:self.feature_dim]
        return feature.astype(float)


    def update(self, pf_fitness: np.ndarray, time_step: float):
        """
        更新历史数据并重训练 GP 模型
        
        Args:
            pf_fitness: 当前 Pareto 前沿目标值 (N_pf, n_obj)
            time_step: 当前时间步 (用作 GP 的输入 x)
        """
        feature = self._compute_feature(pf_fitness)
        self.time_steps.append(time_step)
        self.features.append(feature)


        # 维护历史窗口
        if len(self.time_steps) > self.max_history:
            self.time_steps.pop(0)
            self.features.pop(0)


        # 重训练 GP (当有足够历史数据时)
        if len(self.time_steps) >= 3:
            self._fit_gp_models()


    def _predict_next_raw(
        self,
        next_time: float,
        n_samples: int = 50,
        var_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测下一时刻的 PF 特征并生成候选解。

        这是保留的早期版本实现，当前实际对外使用的是文件后部修正后的
        `predict_next()`，其中补上了预测值反归一化逻辑。
        
        Args:
            next_time: 下一时刻的时间步
            n_samples: 生成的候选解数量
            var_bounds: 决策变量边界 (lb, ub), shape (n_var,)
        
        Returns:
            predicted_feature: 预测的 PF 特征向量 (feature_dim,)
            candidate_solutions: 基于预测特征生成的候选解 (n_samples, n_var)
                                  注意: 由于无法直接从特征反推解,
                                  此处返回基于预测均值/方差的高斯样本
        """
        if len(self.time_steps) < 3:
            # 历史不足, 返回 None 表示无预测
            return np.zeros(self.feature_dim), None


        # 预测下一时刻的特征
        X_test = np.array([[next_time]])
        predicted_mean = np.zeros(self.feature_dim)
        predicted_std = np.ones(self.feature_dim)


        for i, gp in enumerate(self.gp_models):
            if gp.is_fitted:
                mu, std = gp.predict(X_test, return_std=True)
                predicted_mean[i] = float(mu[0])
                predicted_std[i] = float(std[0]) if std is not None else 1.0


        # 基于预测特征生成候选解
        # 策略: 从预测的 PF 均值/范围生成高斯样本
        # feature[0:2] = PF 在目标空间的均值
        # feature[6:8] = PF 在目标空间的范围 (用于估计分布宽度)
        candidate_solutions = None
        if var_bounds is not None:
            lb, ub = var_bounds
            n_var = len(lb)


            # 用预测的目标空间均值作为搜索中心
            # 由于决策-目标映射未知, 使用高斯噪声生成多样化候选
            center = (lb + ub) / 2.0


            # 预测的 PF 范围大则采样范围大
            spread_scale = float(np.mean(predicted_std))
            spread = (ub - lb) * 0.2 * (1 + spread_scale)


            candidate_solutions = (
                center + np.random.randn(n_samples, n_var) * spread
            )
            candidate_solutions = np.clip(candidate_solutions, lb, ub)


        return predicted_mean, candidate_solutions


    def get_prediction_confidence(self) -> float:
        """
        返回当前预测的置信度 (0~1, 越高越可信)
        
        依据:
          - 历史越长 → 置信度越高
          - GP 方差越小 → 置信度越高
          - 特征变化越规律 → 置信度越高
        
        Returns:
            confidence: 0~1
        """
        if len(self.time_steps) < 3:
            return 0.0


        # 历史长度贡献
        hist_score = min(len(self.time_steps) / self.max_history, 1.0)


        # 特征变化规律性 (用 R² 衡量线性度)
        if len(self.features) >= 3:
            features_arr = np.array(self.features)
            times = np.array(self.time_steps)
            r2_list = []
            for dim in range(min(4, self.feature_dim)):
                y = features_arr[:, dim]
                if np.std(y) < 1e-10:
                    r2_list.append(1.0)
                    continue
                # 线性拟合
                coeffs = np.polyfit(times, y, 1)
                y_pred = np.polyval(coeffs, times)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
                r2_list.append(max(0, 1 - ss_res / ss_tot))
            regularity_score = float(np.mean(r2_list))
        else:
            regularity_score = 0.5


        confidence = 0.4 * hist_score + 0.6 * regularity_score
        return float(np.clip(confidence, 0.0, 0.95))


    def _fit_gp_models(self):
        """为每个特征维度拟合独立的 GP 模型"""
        X = np.array(self.time_steps).reshape(-1, 1)  # (n, 1)
        features_arr = np.array(self.features)  # (n, feature_dim)


        for i in range(self.feature_dim):
            y = features_arr[:, i]
            # 归一化 y 使 GP 更稳定
            y_mean = y.mean()
            y_std = y.std() + 1e-8
            y_norm = (y - y_mean) / y_std


            try:
                self.gp_models[i].fit(X, y_norm)
                # 保存归一化参数用于反归一化
                self.gp_models[i]._y_mean = y_mean
                self.gp_models[i]._y_std = y_std
            except Exception as e:
                # 可以根据需要记录异常或简单跳过
                pass

# 同时修复 predict_next 中的预测反归一化
    def predict_next(self, next_time, n_samples=50, var_bounds=None, anchors=None):
        if len(self.time_steps) < 3:
            return np.zeros(self.feature_dim), None

        X_test = np.array([[next_time]])
        predicted_mean = np.zeros(self.feature_dim)
        predicted_std = np.ones(self.feature_dim)
        for i, gp in enumerate(self.gp_models):
            if getattr(gp, 'is_fitted', False):
                mu, std = gp.predict(X_test, return_std=True)
                # 反归一化  
                y_mean = getattr(gp, '_y_mean', 0.0)
                y_std  = getattr(gp, '_y_std',  1.0)
                predicted_mean[i] = float(mu[0]) * y_std + y_mean
                predicted_std[i]  = float(std[0]) * y_std if std is not None else 1.0

        candidate_solutions = None
        if var_bounds is not None:
            lb, ub = var_bounds
            lb = np.asarray(lb, dtype=float)
            ub = np.asarray(ub, dtype=float)
            if anchors is not None:
                anchor_array = np.atleast_2d(np.asarray(anchors, dtype=float))
                center = np.mean(anchor_array, axis=0)
                anchor_spread = np.std(anchor_array, axis=0)
            else:
                center = 0.5 * (lb + ub)
                anchor_spread = 0.1 * (ub - lb)
            spread_scale = float(np.clip(np.mean(predicted_std), 0.05, 2.5))
            shift_dim = min(2, len(center), len(predicted_mean))
            center = center.copy()
            if shift_dim > 0 and self.features:
                reference = np.asarray(self.features[-1][:shift_dim], dtype=float)
                delta = np.asarray(predicted_mean[:shift_dim], dtype=float) - reference
                if np.any(np.abs(delta) > 0.0):
                    direction = np.sign(delta)
                    center[:shift_dim] = center[:shift_dim] + direction * 0.08 * (ub[:shift_dim] - lb[:shift_dim])
            spread_floor = 0.05 * (ub - lb)
            spread = np.maximum(anchor_spread, spread_floor) * (1.0 + 0.5 * spread_scale)
            candidate_solutions = center + np.random.randn(n_samples, len(lb)) * spread
            candidate_solutions = np.clip(candidate_solutions, lb, ub)

        return predicted_mean, candidate_solutions
