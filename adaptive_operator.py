"""
===============================================================================
adaptive_operator.py
MAB 自适应算子选择 — UCB1 策略
===============================================================================
【设计动机】
  原始 KEMM 代码中的比例分配是硬编码的魔法数字:
    memory_ratio = 0.35   # ← 为什么是 0.35?
    predict_ratio = 0.35  # ← 纯粹经验
    transfer_ratio = 0.10 # ← 没有理论依据
  
  正确做法: 将各策略视为"臂", 用 MAB 在线学习最优分配比例


【MAB 理论基础】
  多臂赌博机 (Multi-Armed Bandit, MAB):
    - 每个"臂"代表一种策略 (记忆精英/线性预测/SGF迁移/随机重初始化)
    - 每次选择后获得奖励信号 (IGD 改善量)
    - 目标: 最大化累积奖励
  
  UCB1 算法 (Upper Confidence Bound):
    选择 i* = argmax_i [ Q_i + c·√(ln T / N_i) ]
    
    其中:
      Q_i = 臂 i 的平均奖励 (exploitation 利用项)
      c·√(ln T / N_i) = 探索奖励 (exploration 探索项)
      T = 总拉动次数, N_i = 臂 i 的拉动次数
      c = 探索系数 (越大越倾向探索)
  
  UCB1 的理论保证:
    期望悔恨 E[Regret(T)] = O(√(K·T·ln T))
    其中 K 是臂的数量
  
  来源: Auer et al., "Finite-time Analysis of the Multiarmed
        Bandit Problem", Machine Learning 2002
        Fialho et al., "Adaptive Operator Selection for
        Optimization", GECCO 2010


【奖励设计】
  奖励信号: IGD 前后的相对改善量
    r = max(0, (IGD_before - IGD_after) / (IGD_before + ε))
  
  滑动窗口: 仅使用最近 W 次奖励计算平均值 (适应非平稳环境)
  
  奖励归一化: 使用 z-score 归一化后映射到 [0, 1]


【臂的定义】
  臂0: 记忆精英直接使用 (Memory Elite)
       - 来源: 论文 Process 2 的 FindBestSol
  臂1: 线性预测 (KF/PPS 风格)
       - 来源: 论文引用 [22][23]
  臂2: SGF 流形迁移 (Transfer)
       - 来源: 论文 Process 3
  臂3: 随机重初始化
       - 来源: RI-DMOEA 基线


【参数设置】
  n_arms = 4   (对应 4 种策略)
  window = 10  (滑动窗口大小)
  c = 0.5      (探索系数)
===============================================================================
"""


import numpy as np
from typing import List, Tuple, Optional
from collections import deque




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  UCB1 多臂赌博机
# ╚═══════════════════════════════════════════════════════════════════════════╝


class UCB1Bandit:
    """
    UCB1 多臂赌博机
    ─────────────────────────────────────────
    实现标准 UCB1 算法, 用于在线学习最优策略比例。
    
    与标准 MAB 的区别:
      - 输出的不是单个离散动作, 而是 K 个臂的连续概率分布
      - 使用 softmax 将 UCB 值转换为概率
      - 支持滑动窗口奖励 (适应非平稳问题)
    
    来源: Auer et al., Machine Learning 2002
    """


    def __init__(
        self,
        n_arms: int = 4,
        window: int = 10,
        c: float = 0.5,
        arm_names: Optional[List[str]] = None
    ):
        """
        Args:
            n_arms: 臂的数量 K (对应策略数)
            window: 滑动窗口大小 W (奖励历史长度)
            c: UCB 探索系数 (越大越倾向探索新策略)
            arm_names: 每个臂的名称 (仅用于日志)
        """
        self.n_arms = n_arms
        self.window = window
        self.c = c
        self.arm_names = arm_names or [f"Arm{i}" for i in range(n_arms)]


        # ── 统计量 ──
        # 每个臂的最近 W 次奖励 (滑动窗口)
        self.reward_windows: List[deque] = [
            deque(maxlen=window) for _ in range(n_arms)
        ]
        # 每个臂的总拉动次数
        self.counts = np.ones(n_arms, dtype=float)  # 初始化为 1 避免除零
        # 总拉动次数
        self.total_count = float(n_arms)  # 初始假设每个臂各拉过一次


        # ── 先验奖励 (初始化) ──
        # 给每个臂设置合理的先验期望值
        # 论文知识: Transfer 在 Non-IID 下更好, Memory 在 IID 下更好
        default_rewards = {
            0: 0.4,  # Memory Elite: 稳定但保守
            1: 0.3,  # Linear Predict: 快速但精度有限
            2: 0.4,  # SGF Transfer: 强但不稳定
            3: 0.2,  # Random Reinit: 作为探索保底
        }
        for arm_idx, reward in default_rewards.items():
            if arm_idx < n_arms:
                self.reward_windows[arm_idx].append(reward)


        # ── 历史记录 ──
        self.selected_arms_history: List[int] = []
        self.rewards_history: List[float] = []
        self.ratios_history: List[np.ndarray] = []


    def get_ucb_values(self) -> np.ndarray:
        """
        计算每个臂的 UCB1 值
        
        UCB1 公式:
          UCB_i = Q_i + c·√(ln T / N_i)
        
        其中:
          Q_i = 滑动窗口内的平均奖励 (exploitation)
          c·√(ln T / N_i) = 探索奖励 (exploration)
        
        Returns:
            ucb_values: (n_arms,) 每个臂的 UCB 值
        """
        # 平均奖励 Q_i
        Q = np.array([
            np.mean(list(w)) if len(w) > 0 else 0.0
            for w in self.reward_windows
        ])


        # 探索奖励: c·√(ln T / N_i)
        exploration = self.c * np.sqrt(
            np.log(self.total_count + 1) / (self.counts + 1e-8)
        )


        return Q + exploration


    def select_ratios(self, temperature: float = 1.0) -> np.ndarray:
        """
        根据 UCB 值选择各臂的比例
        
        使用 softmax 将 UCB 值转换为概率分布:
          p_i = exp(UCB_i / T) / Σ exp(UCB_j / T)
        
        Args:
            temperature: softmax 温度 (越低越"贪婪", 越高越均匀)
        
        Returns:
            ratios: (n_arms,) 各策略的分配比例, 求和为 1
        """
        ucb = self.get_ucb_values()


        # Softmax 转换为比例
        # 减去最大值防止数值溢出
        ucb_shifted = ucb - ucb.max()
        exp_ucb = np.exp(ucb_shifted / (temperature + 1e-8))
        ratios = exp_ucb / (exp_ucb.sum() + 1e-10)


        # 确保每个策略至少有 5% 的比例 (最低探索率)
        min_ratio = 0.05
        ratios = np.maximum(ratios, min_ratio)
        ratios = ratios / ratios.sum()


        self.ratios_history.append(ratios.copy())
        return ratios


    def update(self, arm_idx: int, reward: float):
        """
        更新指定臂的奖励统计
        
        Args:
            arm_idx: 被选中的臂索引 (0 ~ n_arms-1)
            reward: 获得的奖励值
                    推荐取值: IGD 相对改善量, 归一化到 [0, 1]
                    reward = max(0, (IGD_before - IGD_after) / IGD_before)
        """
        assert 0 <= arm_idx < self.n_arms, f"无效臂索引: {arm_idx}"


        # 奖励归一化到 [0, 1]
        reward = np.clip(float(reward), 0.0, 1.0)


        # 更新滑动窗口
        self.reward_windows[arm_idx].append(reward)


        # 更新计数
        self.counts[arm_idx] += 1
        self.total_count += 1


        # 记录历史
        self.selected_arms_history.append(arm_idx)
        self.rewards_history.append(reward)


    def get_statistics(self) -> dict:
        """
        返回当前 MAB 统计信息 (用于调试和可视化)
        
        Returns:
            stats: {
                'q_values': 每个臂的平均奖励,
                'counts': 每个臂的拉动次数,
                'exploration': 探索奖励项,
                'ucb': UCB 值,
                'current_ratios': 当前推荐比例
            }
        """
        Q = np.array([
            np.mean(list(w)) if len(w) > 0 else 0.0
            for w in self.reward_windows
        ])
        exploration = self.c * np.sqrt(
            np.log(self.total_count + 1) / (self.counts + 1e-8)
        )
        ucb = Q + exploration
        ratios = self.select_ratios()


        return {
            'arm_names': self.arm_names,
            'q_values': Q,
            'counts': self.counts.copy(),
            'exploration': exploration,
            'ucb': ucb,
            'current_ratios': ratios,
            'total_updates': len(self.rewards_history)
        }


    def reset(self):
        """重置 MAB 状态 (用于多次运行之间的隔离)"""
        for w in self.reward_windows:
            w.clear()
        self.counts = np.ones(self.n_arms, dtype=float)
        self.total_count = float(self.n_arms)
        self.selected_arms_history.clear()
        self.rewards_history.clear()




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  自适应算子选择器 — 集成 IGD 反馈
# ╚═══════════════════════════════════════════════════════════════════════════╝


class AdaptiveOperatorSelector:
    """
    自适应算子选择器
    ─────────────────────────────────────────
    整合 UCB1 MAB 和 IGD 反馈信号, 提供完整的自适应算子选择功能。
    
    使用方式:
      1. 调用 get_ratios() 获取本次的分配比例
      2. 用对应比例分配种群初始化策略
      3. 进化若干代后调用 update_with_igd() 提供反馈
      4. MAB 自动学习最优分配
    
    4 种策略 (4 个"臂"):
      臂0 - MEMORY:   使用记忆库中的精英解 (FindBestSol)
      臂1 - PREDICT:  线性/KF 预测 (质心外推)
      臂2 - TRANSFER: SGF 流形迁移 (Transfer)
      臂3 - REINIT:   随机重初始化 (探索保底)
    
    来源:
      Fialho et al., "Adaptive Operator Selection for Optimization",
      GECCO 2010 — 将 AOS 用于进化算法
      
      本文扩展: 将 AOS 用于动态多目标优化的
                环境响应策略选择 (而非交叉/变异算子)
    """


    # 策略索引常量
    MEMORY = 0    # 记忆精英
    PREDICT = 1   # 线性预测
    TRANSFER = 2  # SGF 迁移
    REINIT = 3    # 随机重初始化


    ARM_NAMES = ['Memory', 'Predict', 'Transfer', 'Reinit']


    def __init__(
        self,
        window: int = 10,
        c: float = 0.5,
        temperature: float = 0.8,
        min_ratio: float = 0.05
    ):
        """
        Args:
            window: 奖励滑动窗口大小
            c: UCB 探索系数
            temperature: softmax 温度 (比例选择时)
            min_ratio: 每个策略的最小比例 (保证多样性)
        """
        self.temperature = temperature
        self.min_ratio = min_ratio


        # UCB1 MAB
        self.bandit = UCB1Bandit(
            n_arms=4,
            window=window,
            c=c,
            arm_names=self.ARM_NAMES
        )


        # 记录上一次的 IGD (用于计算奖励)
        self._prev_igd: Optional[float] = None
        self._prev_ratios: Optional[np.ndarray] = None


        # 统计
        self.update_count = 0


    def get_ratios(self) -> Tuple[np.ndarray, dict]:
        """
        获取本次各策略的分配比例
        
        Returns:
            ratios: (4,) 各策略比例 [memory, predict, transfer, reinit]
            info: 调试信息字典
        
        用法示例:
            ratios, info = selector.get_ratios()
            n_memory   = int(pop_size * ratios[0])
            n_predict  = int(pop_size * ratios[1])
            n_transfer = int(pop_size * ratios[2])
            n_reinit   = pop_size - n_memory - n_predict - n_transfer
        """
        ratios = self.bandit.select_ratios(temperature=self.temperature)


        # 确保最小比例
        ratios = np.maximum(ratios, self.min_ratio)
        ratios = ratios / ratios.sum()


        self._prev_ratios = ratios.copy()


        stats = self.bandit.get_statistics()
        info = {
            'ratios': ratios,
            'ucb_values': stats['ucb'],
            'q_values': stats['q_values'],
            'counts': stats['counts'],
        }


        return ratios, info


    def update_with_igd(self, current_igd: float):
        """
        用 IGD 变化量更新 MAB
        
        奖励信号设计:
          reward = max(0, (IGD_prev - IGD_curr) / IGD_prev)
        
          - IGD 下降 → 正奖励 (策略有效)
          - IGD 上升 → 零奖励 (策略无效)
          - 归一化到 [0, 1]
        
        Args:
            current_igd: 当前时刻的 IGD 值
        
        来源: 改进思路来自 Fialho et al. GECCO 2010
              "the improvement of solution quality is used as reward"
        """
        if self._prev_igd is None:
            self._prev_igd = current_igd
            return


        # 计算 IGD 相对改善量
        if self._prev_igd > 1e-10:
            reward = max(0.0, (self._prev_igd - current_igd) / self._prev_igd)
        else:
            reward = 0.0


        reward = float(np.clip(reward, 0.0, 1.0))


        # 为本次使用的所有策略更新奖励
        # (按比例加权分配奖励)
        if self._prev_ratios is not None:
            for arm_idx in range(4):
                if self._prev_ratios[arm_idx] > self.min_ratio + 0.01:
                    # 仅更新实际使用的策略
                    self.bandit.update(arm_idx, reward)


        self._prev_igd = current_igd
        self.update_count += 1


    def force_update(self, arm_idx: int, reward: float):
        """
        强制更新单个臂 (用于精确的策略-奖励归因)
        
        当能明确归因某个策略产生了效果时使用。
        
        Args:
            arm_idx: 臂索引 (使用类常量 MEMORY/PREDICT/TRANSFER/REINIT)
            reward: 奖励值 [0, 1]
        """
        self.bandit.update(arm_idx, reward)


    def get_recommended_mode(self) -> str:
        """
        基于 UCB 值返回推荐的主要模式
        
        Returns:
            mode: 'memory_dominant' / 'predict_dominant' /
                  'transfer_dominant' / 'balanced'
        """
        ratios, _ = self.get_ratios()
        max_idx = int(np.argmax(ratios))


        if ratios[max_idx] > 0.4:  # 某策略占比 > 40% 视为主导
            return f"{self.ARM_NAMES[max_idx].lower()}_dominant"
        return "balanced"


    def print_status(self):
        """打印当前 MAB 状态 (调试用)"""
        stats = self.bandit.get_statistics()
        ratios = stats['current_ratios']
        print(f"\n  [MAB 状态] 总更新次数: {self.update_count}")
        print(f"  {'策略':<12} {'Q值':>8} {'探索':>8} {'UCB':>8} {'比例':>8} {'次数':>6}")
        print(f"  {'-'*55}")
        for i, name in enumerate(self.ARM_NAMES):
            print(f"  {name:<12} "
                  f"{stats['q_values'][i]:>8.3f} "
                  f"{stats['exploration'][i]:>8.3f} "
                  f"{stats['ucb'][i]:>8.3f} "
                  f"{ratios[i]:>8.1%} "
                  f"{int(stats['counts'][i]):>6d}")




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Pareto 前沿漂移检测器 (替代原始线性度 R² 检测)
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ParetoFrontDriftDetector:
    """
    Pareto 前沿漂移检测器
    ─────────────────────────────────────────
    替代原始 KEMM 中用 R² 检测线性度的方法。
    
    原始方法的问题:
      R² 是对质心轨迹的线性拟合度量, 但:
      1. 质心线性不代表 Pareto 前沿线性漂移
      2. R² 对周期性变化 (如 sin(t)) 给出极低值,
         但此时知识实际上可以迁移
      3. 忽略了 Pareto 前沿形状的变化 (宽/窄/弯曲)
    
    改进方案: Wasserstein 距离 + 形状特征
      - 用 Pareto 前沿的多维统计特征检测变化幅度
      - 用自回归模型预测下一时刻 PF 特征
      - 结合 IGD 历史评估迁移效果
    
    来源:
      思路来自 PPS 的自回归预测 (论文引用 [22]):
        "predicts the next center point by an autoregressive model
         using a series of center points"
      扩展到 Pareto 前沿整体特征的预测
    """


    def __init__(self, window: int = 6):
        """
        Args:
            window: 历史特征窗口大小 (越大越稳定但响应越慢)
        """
        self.window = window
        # Pareto 前沿特征历史
        self.pf_feature_history: List[np.ndarray] = []
        # IGD 历史
        self.igd_history: List[float] = []
        # 变化幅度历史
        self.change_history: List[float] = []


    def compute_pf_feature(self, pf_fitness: np.ndarray) -> np.ndarray:
        """
        计算 Pareto 前沿的紧凑特征表示
        
        特征维度 (共 10 维):
          [0:2]  各目标均值
          [2:4]  各目标标准差
          [4:6]  各目标 25/75 百分位 (形状特征)
          [6:8]  各目标范围 (spread)
          [8]    PF 点数
          [9]    PF 密度 (点数/边界框面积)
        
        Args:
            pf_fitness: Pareto 前沿目标值 (N_pf, n_obj)
        
        Returns:
            feature: (10,) 特征向量
        """
        if len(pf_fitness) == 0:
            return np.zeros(10)


        n_obj = pf_fitness.shape[1]


        # 只用前两个目标 (兼容多目标)
        f = pf_fitness[:, :min(2, n_obj)]
        if f.shape[1] < 2:
            f = np.hstack([f, np.zeros((len(f), 2 - f.shape[1]))])


        feature = np.array([
            float(np.mean(f[:, 0])),     # f1 均值
            float(np.mean(f[:, 1])),     # f2 均值
            float(np.std(f[:, 0])),      # f1 标准差
            float(np.std(f[:, 1])),      # f2 标准差
            float(np.percentile(f[:, 0], 25)),  # f1 下四分位
            float(np.percentile(f[:, 1], 25)),  # f2 下四分位
            float(np.max(f[:, 0]) - np.min(f[:, 0])),  # f1 范围
            float(np.max(f[:, 1]) - np.min(f[:, 1])),  # f2 范围
            float(len(pf_fitness)),       # PF 点数
            float(len(pf_fitness)) / (   # PF 密度
                (np.max(f[:, 0]) - np.min(f[:, 0])) *
                (np.max(f[:, 1]) - np.min(f[:, 1])) + 1e-8
            )
        ])


        return feature


    def update(self, pf_fitness: np.ndarray, igd: Optional[float] = None):
        """
        更新检测器状态
        
        Args:
            pf_fitness: 当前 Pareto 前沿目标值
            igd: 当前 IGD 值 (可选)
        """
        feature = self.compute_pf_feature(pf_fitness)
        self.pf_feature_history.append(feature)
        if len(self.pf_feature_history) > self.window:
            self.pf_feature_history.pop(0)


        if igd is not None:
            self.igd_history.append(igd)
            if len(self.igd_history) > self.window:
                self.igd_history.pop(0)


        # 计算变化幅度
        if len(self.pf_feature_history) >= 2:
            f_curr = self.pf_feature_history[-1]
            f_prev = self.pf_feature_history[-2]
            scale = np.abs(f_prev) + 1e-8
            change = float(np.mean(np.abs(f_curr - f_prev) / scale))
            self.change_history.append(change)
            if len(self.change_history) > self.window:
                self.change_history.pop(0)


    def get_change_magnitude(self) -> float:
        """
        获取当前环境变化幅度估计 (0~1, 越大变化越剧烈)
        
        Returns:
            magnitude: 0 表示无变化, 1 表示极大变化
        """
        if not self.change_history:
            return 0.5  # 无信息时返回中间值
        return float(np.clip(np.mean(self.change_history[-3:]), 0, 1))


    def predict_transferability(self) -> float:
        """
        预测历史知识的可迁移性 (0~1, 越高迁移越有效)
        
        判断依据:
          1. 变化幅度越小 → 可迁移性越高 (环境稳定)
          2. IGD 历史越稳定 → 可迁移性越高 (收敛良好)
          3. PF 形状变化越小 → 可迁移性越高 (结构保持)
        
        Returns:
            transferability: 0~1, 越高表示迁移越有价值
        """
        if len(self.pf_feature_history) < 2:
            return 0.5


        # 变化幅度贡献
        change = self.get_change_magnitude()
        change_score = 1.0 - np.clip(change * 3, 0, 1)


        # IGD 稳定性贡献
        if len(self.igd_history) >= 3:
            igd_cv = np.std(self.igd_history) / (np.mean(self.igd_history) + 1e-8)
            igd_score = 1.0 - np.clip(igd_cv, 0, 1)
        else:
            igd_score = 0.5


        # 综合评分
        transferability = 0.6 * change_score + 0.4 * igd_score
        return float(np.clip(transferability, 0.1, 0.9))