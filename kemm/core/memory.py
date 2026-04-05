"""Compressed elite memory built on a lightweight NumPy VAE implementation.

This module stores historical Pareto-set structure in a compact latent space so
KEMM can reuse past experience without keeping every raw population snapshot.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


warnings.filterwarnings('ignore')


class LightweightVAE:
    """Small NumPy-only beta-VAE used to encode and reconstruct elite-set structure."""


    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        beta: float = 0.1,
        lr: float = 1e-3
    ):
        """
        Args:
            input_dim: 输入维度 D (n_var 或 n_pts*2)
            latent_dim: 隐空间维度 (默认 8, 适合典型 DMOP 问题)
            hidden_dim: 隐层宽度 H (默认 64)
            beta: KL 散度权重 (β-VAE, 越小重建越准)
            lr: Adam 学习率
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.lr = lr


        # ── 初始化网络参数 ──
        # 编码器: D → H
        scale_enc = np.sqrt(2.0 / input_dim)
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * scale_enc
        self.b_enc1 = np.zeros(hidden_dim)


        # 编码器: H → μ (均值头)
        scale_h = np.sqrt(2.0 / hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * scale_h
        self.b_mu = np.zeros(latent_dim)


        # 编码器: H → log σ² (方差头)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * scale_h * 0.1
        self.b_logvar = np.full(latent_dim, -2.0)  # 初始化为小方差


        # 解码器: latent_dim → H
        scale_dec = np.sqrt(2.0 / latent_dim)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * scale_dec
        self.b_dec1 = np.zeros(hidden_dim)


        # 解码器: H → D
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * scale_h
        self.b_dec2 = np.zeros(input_dim)


        # ── Adam 优化器状态 ──
        self._init_adam_states()


        # ── 数据归一化统计 ──
        self.data_mean = None
        self.data_std = None


        # ── 训练历史 ──
        self.loss_history = []
        self.n_updates = 0


    def _init_adam_states(self):
        """初始化 Adam 一阶/二阶矩"""
        params = self._get_params()
        self.m = [np.zeros_like(p) for p in params]  # 一阶矩
        self.v = [np.zeros_like(p) for p in params]  # 二阶矩
        self.adam_t = 0  # 时间步


    def _get_params(self) -> list:
        """返回所有参数的列表 (按固定顺序)"""
        return [
            self.W_enc1, self.b_enc1,
            self.W_mu, self.b_mu,
            self.W_logvar, self.b_logvar,
            self.W_dec1, self.b_dec1,
            self.W_dec2, self.b_dec2
        ]


    # ══════════════════════════════════════
    #  前向传播
    # ══════════════════════════════════════


    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        编码器前向传播
        
        Args:
            x: 输入数据 (N, D), 已归一化
        
        Returns:
            mu: 隐均值 (N, latent_dim)
            logvar: 隐对数方差 (N, latent_dim)
        """
        # 第一层: D → H, tanh 激活
        h = np.tanh(x @ self.W_enc1 + self.b_enc1)  # (N, H)


        # 输出层: 均值和对数方差
        mu = h @ self.W_mu + self.b_mu          # (N, latent_dim)
        logvar = h @ self.W_logvar + self.b_logvar  # (N, latent_dim)
        logvar = np.clip(logvar, -10, 2)         # 防止数值溢出


        return mu, logvar


    def reparameterize(
        self,
        mu: np.ndarray,
        logvar: np.ndarray
    ) -> np.ndarray:
        """
        重参数化采样: z = μ + σ·ε,  ε ~ N(0, I)
        
        这是 VAE 的关键技巧: 将随机性分离出来,
        使得梯度可以通过 μ 和 σ 反传。
        
        来源: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
        """
        std = np.exp(0.5 * logvar)      # σ = exp(logσ²/2)
        eps = np.random.randn(*mu.shape)  # ε ~ N(0, I)
        return mu + std * eps             # 重参数化


    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        解码器前向传播
        
        Args:
            z: 隐变量 (N, latent_dim)
        
        Returns:
            x_recon: 重建数据 (N, D)
        """
        # 第一层: latent_dim → H, tanh 激活
        h = np.tanh(z @ self.W_dec1 + self.b_dec1)  # (N, H)


        # 输出层: H → D, 线性输出 (回归)
        x_recon = h @ self.W_dec2 + self.b_dec2  # (N, D)


        return x_recon


    def forward(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整前向传播
        
        Returns:
            x_recon: 重建 (N, D)
            z: 采样隐变量 (N, latent_dim)
            mu: 隐均值 (N, latent_dim)
            logvar: 隐对数方差 (N, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar


    # ══════════════════════════════════════
    #  损失函数
    # ══════════════════════════════════════


    def elbo_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        ELBO 损失函数
        
        L = -E[log p(x|z)] + β·KL(q(z|x)||p(z))
        
        其中:
          重建损失 = MSE(x, x_recon)  (假设高斯似然)
          KL 散度  = -0.5·Σ(1 + logσ² - μ² - σ²)
        
        Args:
            x: 原始输入 (N, D)
            x_recon: 重建输出 (N, D)
            mu: 隐均值 (N, latent_dim)
            logvar: 隐对数方差 (N, latent_dim)
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        N = len(x)


        # 重建损失: 均方误差 (对应高斯解码器)
        recon_loss = float(np.mean(np.sum((x - x_recon) ** 2, axis=1)))


        # KL 散度: KL(N(μ,σ²) || N(0,1))
        # 解析解: -0.5 * (1 + logσ² - μ² - σ²)
        kl_loss = float(-0.5 * np.mean(
            np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=1)
        ))


        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss


    # ══════════════════════════════════════
    #  梯度计算 (手动反传)
    # ══════════════════════════════════════


    def compute_gradients(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        z: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray
    ) -> list:
        """
        手动计算反向传播梯度
        
        计算顺序: 解码器 → 隐采样 → 编码器
        使用链式法则手动推导所有参数的梯度。
        
        Notes:
            由于纯 NumPy 实现无自动微分, 此处手动推导。
            对于复杂网络建议使用 PyTorch/JAX 替代。
        """
        N = len(x)


        # ── 解码器梯度 ──
        # ∂L/∂x_recon = 2*(x_recon - x)/N  (MSE 梯度)
        dL_dxr = 2.0 * (x_recon - x) / N  # (N, D)


        # 解码器第二层: x_recon = h_dec @ W_dec2 + b_dec2
        dL_dW_dec2 = self._h_dec.T @ dL_dxr      # (H, D)
        dL_db_dec2 = dL_dxr.mean(axis=0)          # (D,)
        dL_dh_dec = dL_dxr @ self.W_dec2.T        # (N, H)


        # tanh 梯度: d tanh(a)/da = 1 - tanh²(a) = 1 - h²
        dL_da_dec = dL_dh_dec * (1 - self._h_dec ** 2)  # (N, H)
        dL_dW_dec1 = z.T @ dL_da_dec              # (latent_dim, H)
        dL_db_dec1 = dL_da_dec.mean(axis=0)       # (H,)
        dL_dz = dL_da_dec @ self.W_dec1.T         # (N, latent_dim)


        # ── 重参数化梯度 ──
        # z = mu + std * eps, std = exp(0.5*logvar)
        std = np.exp(0.5 * logvar)
        eps = (z - mu) / (std + 1e-8)             # 还原 ε


        # KL 梯度
        # ∂KL/∂μ = μ/N
        # ∂KL/∂logvar = 0.5*(exp(logvar) - 1)/N
        dL_dmu = dL_dz + self.beta * mu / N
        dL_dlogvar = (0.5 * dL_dz * eps * std +
                      self.beta * 0.5 * (np.exp(logvar) - 1) / N)


        # ── 编码器梯度 ──
        # μ 路径: H → latent_dim
        dL_dW_mu = self._h_enc.T @ dL_dmu         # (H, latent_dim)
        dL_db_mu = dL_dmu.mean(axis=0)            # (latent_dim,)
        dL_dh_enc_mu = dL_dmu @ self.W_mu.T       # (N, H)


        # logvar 路径: H → latent_dim
        dL_dW_logvar = self._h_enc.T @ dL_dlogvar  # (H, latent_dim)
        dL_db_logvar = dL_dlogvar.mean(axis=0)     # (latent_dim,)
        dL_dh_enc_lv = dL_dlogvar @ self.W_logvar.T  # (N, H)


        # 合并两个路径
        dL_dh_enc = dL_dh_enc_mu + dL_dh_enc_lv   # (N, H)


        # 编码器第一层
        dL_da_enc = dL_dh_enc * (1 - self._h_enc ** 2)  # (N, H), tanh 梯度
        dL_dW_enc1 = x.T @ dL_da_enc              # (D, H)
        dL_db_enc1 = dL_da_enc.mean(axis=0)       # (H,)


        return [
            dL_dW_enc1, dL_db_enc1,
            dL_dW_mu, dL_db_mu,
            dL_dW_logvar, dL_db_logvar,
            dL_dW_dec1, dL_db_dec1,
            dL_dW_dec2, dL_db_dec2
        ]


    # ══════════════════════════════════════
    #  Adam 优化器步骤
    # ══════════════════════════════════════


    def adam_step(self, grads: list, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam 优化器更新步骤
        
        来源: Kingma & Ba, "Adam: A Method for Stochastic Optimization",
              ICLR 2015
        
        更新规则:
          m_t = β₁·m_{t-1} + (1-β₁)·g_t
          v_t = β₂·v_{t-1} + (1-β₂)·g_t²
          m̂_t = m_t / (1-β₁^t)
          v̂_t = v_t / (1-β₂^t)
          θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)
        """
        self.adam_t += 1
        t = self.adam_t
        params = self._get_params()


        for i, (p, g) in enumerate(zip(params, grads)):
            # 梯度裁剪防止梯度爆炸
            g = np.clip(g, -5.0, 5.0)


            # 更新矩估计
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g ** 2


            # 偏差修正
            m_hat = self.m[i] / (1 - beta1 ** t)
            v_hat = self.v[i] / (1 - beta2 ** t)


            # 参数更新 (in-place)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


    # ══════════════════════════════════════
    #  训练接口
    # ══════════════════════════════════════


    def fit(
        self,
        data: np.ndarray,
        n_epochs: int = 20,
        batch_size: int = 32,
        verbose: bool = False
    ):
        """
        在线训练 VAE
        
        在每个时刻调用此函数微调 VAE, 使其适应当前 Pareto 前沿的分布。
        
        Args:
            data: 训练数据 (N, D) — 精英解的决策变量
            n_epochs: 训练轮数 (在线训练, 通常 10-30 轮)
            batch_size: mini-batch 大小
            verbose: 是否打印训练损失
        """
        N, D = data.shape
        assert D == self.input_dim, f"维度不匹配: {D} != {self.input_dim}"


        # ── 数据归一化 ──
        if self.data_mean is None:
            self.data_mean = data.mean(axis=0)
            self.data_std = data.std(axis=0) + 1e-8
        else:
            # 在线更新归一化统计 (指数移动平均)
            alpha = 0.1
            self.data_mean = (1 - alpha) * self.data_mean + alpha * data.mean(axis=0)
            self.data_std = (1 - alpha) * self.data_std + alpha * (data.std(axis=0) + 1e-8)


        x_norm = (data - self.data_mean) / self.data_std  # 归一化到 ~N(0,1)


        # ── Mini-batch 训练 ──
        epoch_losses = []
        for epoch in range(n_epochs):
            idx = np.random.permutation(N)
            batch_losses = []


            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                x_batch = x_norm[batch_idx]


                # 缓存中间激活 (用于反传)  先手动计算并缓存隐层激活
                self._h_enc = np.tanh(x_batch @ self.W_enc1 + self.b_enc1)
                
                #再调用encode（内部也会算h_enc，但我们用缓存版）
                mu     = self._h_enc @ self.W_mu     + self.b_mu
                logvar = self._h_enc @ self.W_logvar + self.b_logvar
                logvar = np.clip(logvar, -10, 2)


                # 重参数化
                z = self.reparameterize(mu, logvar)


                # 解码
                self._h_dec = np.tanh(z @ self.W_dec1 + self.b_dec1)
                x_recon = self._h_dec @ self.W_dec2 + self.b_dec2
    

                # 计算损失
                loss, recon, kl = self.elbo_loss(x_batch, x_recon, mu, logvar)
                batch_losses.append(loss)


                # 反传 + Adam 更新
                grads = self.compute_gradients(x_batch, x_recon, z, mu, logvar)
                self.adam_step(grads)


            epoch_losses.append(np.mean(batch_losses))


        self.loss_history.extend(epoch_losses)
        self.n_updates += 1


        if verbose:
            print(f"    VAE 训练 {n_epochs} 轮, 最终 Loss: {epoch_losses[-1]:.4f}")


    def encode_data(self, data: np.ndarray) -> np.ndarray:
        """
        编码数据到隐空间 (仅返回均值, 不采样)
        
        Args:
            data: (N, D)
        
        Returns:
            z_mean: (N, latent_dim)
        """
        if self.data_mean is None:
            return np.random.randn(len(data), self.latent_dim)
        x_norm = (data - self.data_mean) / self.data_std
        mu, _ = self.encode(x_norm)
        return mu


    def decode_latent(self, z: np.ndarray) -> np.ndarray:
        """
        从隐空间解码到决策空间
        
        Args:
            z: (N, latent_dim)
        
        Returns:
            x: (N, D), 已反归一化
        """
        if self.data_mean is None:
            return np.random.randn(len(z), self.input_dim)
        x_norm = self.decode(z)
        return x_norm * self.data_std + self.data_mean


    def sample(self, n: int, temperature: float = 1.0) -> np.ndarray:
        """
        从先验 z ~ N(0, temperature²·I) 采样并解码
        
        Args:
            n: 采样数量
            temperature: 采样温度 (越高越多样化)
        
        Returns:
            samples: (n, D)
        """
        z = np.random.randn(n, self.latent_dim) * temperature
        return self.decode_latent(z)




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  VAE 增强的记忆库
# ╚═══════════════════════════════════════════════════════════════════════════╝


class VAECompressedMemory:
    """Historical elite archive that retrieves and decodes latent memories for KEMM."""


    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        capacity: int = 50,
        beta: float = 0.1,
        online_epochs: int = 15
    ):
        """
        Args:
            input_dim: 决策变量维度 D
            latent_dim: VAE 隐空间维度 (默认 8)
            hidden_dim: VAE 隐层宽度 (默认 64)
            capacity: 记忆库容量 (论文 C, 默认 50)
            beta: VAE 的 KL 权重 (β-VAE)
            online_epochs: 每次更新时的训练轮数
        """
        self.capacity = capacity
        self.online_epochs = online_epochs
        self.memory: List[Dict] = []
        self._age = 0


        # VAE 模型
        self.vae = LightweightVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            beta=beta
        )


        # 是否已预热 (初期直接存原始解, 累积足够数据后启用 VAE)
        self._warmup_done = False
        self._warmup_threshold = 50  # 至少看过 50 个样本才启用 VAE


    def store(
        self,
        solutions: np.ndarray,
        fitness: np.ndarray,
        fingerprint: np.ndarray
    ):
        """
        存储当前时刻的精英解
        
        流程:
          1. 用新数据微调 VAE (在线更新)
          2. 将解编码为隐向量
          3. 存储隐向量 + 统计摘要 (非原始解)
          4. 若记忆库满, 替换最旧的条目 (FIFO)
        
        来源: 论文 Process 1, lines 10-14
          "if size(P ∪ Solutionst) ≤ C then Store;
           else Replace earliest stored"
        
        Args:
            solutions: 精英解 (N, D)
            fitness: 对应适应度 (N, n_obj)
            fingerprint: 环境指纹 (fp_dim,)
        """
        self._age += 1
        N = len(solutions)


        # ── Step 1: 在线更新 VAE ──
        if N >= 4:  # 至少需要 4 个样本
            n_epochs = self.online_epochs if self._warmup_done else self.online_epochs * 2
            self.vae.fit(solutions, n_epochs=n_epochs, batch_size=min(32, N))


            if self.vae.n_updates >= 3:
                self._warmup_done = True


        # ── Step 2: 编码到隐空间 ──
        if self._warmup_done:
            z = self.vae.encode_data(solutions)  # (N, latent_dim)
            z_mean = z.mean(axis=0)               # (latent_dim,)
            z_std = z.std(axis=0)                 # (latent_dim,)
        else:
            # 预热期: 用 PCA 近似
            z_mean = solutions.mean(axis=0)[:self.vae.latent_dim]
            z_std = solutions.std(axis=0)[:self.vae.latent_dim]


        # ── Step 3: 构建记忆条目 ──
        # 保留少量原始精英 (用于直接使用, 不仅用于解码)
        n_raw = min(20, N)  # 最多保留 20 个原始解
        raw_idx = np.random.choice(N, n_raw, replace=False)


        entry = {
            'latent_mean': z_mean,
            'latent_std': z_std,
            'fitness_mean': fitness.mean(axis=0),
            'fitness_std': fitness.std(axis=0),
            'fingerprint': fingerprint.copy(),
            'age': self._age,
            'raw_elites': solutions[raw_idx].copy(),  # 少量原始解
            'raw_fitness': fitness[raw_idx].copy(),
            'n_original': N  # 记录原始数量
        }


        # ── Step 4: 存储 (FIFO 溢出替换) ──
        self.memory.append(entry)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)  # 替换最早存储的


    def retrieve(
        self,
        query_fingerprint: np.ndarray,
        top_k: int = 3,
        n_decode: int = 50
    ) -> List[Dict]:
        """
        从记忆库检索最相似的历史环境并解码
        
        检索策略:
          1. 在归一化后的指纹空间计算欧氏距离
          2. 返回 top-K 最相似条目
          3. 用 VAE 解码器从隐向量采样还原解
        
        Args:
            query_fingerprint: 当前环境指纹
            top_k: 检索数量
            n_decode: 每个条目解码的样本数
        
        Returns:
            检索结果列表, 每个元素为:
            {'solutions': np.ndarray, 'fitness': np.ndarray,
             'similarity': float, 'age': int}
        """
        if not self.memory:
            return []


        # ── 计算指纹相似度 ──
        fps = np.array([m['fingerprint'] for m in self.memory])
        query = query_fingerprint


        # 归一化避免量纲影响
        fp_scale = np.std(fps, axis=0) + 1e-8
        dists = np.linalg.norm(
            (fps - query) / fp_scale, axis=1
        )  # (M,)


        # 相似度 = 1 / (1 + 距离)
        similarities = 1.0 / (1.0 + dists)


        # Top-K 索引
        k = min(top_k, len(dists))
        top_idx = np.argsort(-similarities)[:k]


        results = []
        for i in top_idx:
            entry = self.memory[i]
            sim = float(similarities[i])


            # ── 解码: 从隐向量采样解 ──
            if self._warmup_done:
                z_mean = entry['latent_mean']     # (latent_dim,)
                z_std = entry['latent_std']       # (latent_dim,)
                # 从以 z_mean 为中心的分布采样
                z_samples = z_mean + np.random.randn(n_decode, len(z_mean)) * z_std
                decoded = self.vae.decode_latent(z_samples)  # (n_decode, D)
                # 混合: 50% 解码样本 + 50% 原始精英 (加噪)
                n_raw_use = min(len(entry['raw_elites']), n_decode // 2)
                raw_noisy = entry['raw_elites'][:n_raw_use].copy()
                raw_noisy += np.random.normal(0, 0.02, raw_noisy.shape)
                solutions = np.vstack([decoded[:n_decode - n_raw_use], raw_noisy])
            else:
                # 预热期直接用原始精英 (加噪扩充)
                n_add = max(0, n_decode - len(entry['raw_elites']))
                solutions = entry['raw_elites'].copy()
                if n_add > 0:
                    idx = np.random.choice(len(solutions), n_add, replace=True)
                    noisy = solutions[idx] + np.random.normal(0, 0.05, (n_add, solutions.shape[1]))
                    solutions = np.vstack([solutions, noisy])
                solutions = solutions[:n_decode]


            results.append({
                'solutions': solutions,
                'fitness': np.tile(entry['fitness_mean'], (len(solutions), 1)),
                'similarity': sim,
                'age': entry['age']
            })


        return results


    def __len__(self) -> int:
        return len(self.memory)


    @property
    def is_warmed_up(self) -> bool:
        return self._warmup_done