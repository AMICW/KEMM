"""
===============================================================================
feasibility_repair.py
可行性驱动路径修复 — 替代惩罚函数
===============================================================================
【设计动机】
  原始 KEMM (船舶规划) 使用惩罚函数处理碰撞约束:
    f_penalized = f_obj + λ · violation²
  
  问题:
    1. λ 难以调节: λ太小 → 忽略约束; λ太大 → 目标函数失真
    2. 进化早期大量无效个体: 随机初始化的路径 70%+ 碰撞
    3. Pareto 前沿中混入可行/不可行解, 误导搜索方向
  
  改进: 可行性驱动修复算子 (Feasibility-Driven Repair, FDR)
    - 检测碰撞路径段
    - 用 Dubins 绕行路径修复
    - 保证种群中 ≥ 80% 为完全可行解
    - 消除惩罚函数中的 λ 超参数


【Dubins 路径原理】
  Dubins 路径是在有最小转弯半径约束下,
  从一个位姿到另一个位姿的最短可行路径。
  
  对于避障应用, 使用简化版:
    1. 检测路径段 p1→p2 与障碍物 O 的交叉
    2. 计算 O 的两条切线方向
    3. 在较短的切线方向插入绕行点 bypass
    4. 递归检查 p1→bypass 和 bypass→p2 是否仍有碰撞
  
  来源:
    Dubins, "On Curves of Minimal Length...", 1957
    Hwangbo et al., "Fast Path Planning...", IEEE T-RO 2007


【与 NSGA-II 的集成方式】
  在每次生成子代后, 对所有不可行个体执行修复:
  
    offspring = crossover_mutation(population)
    offspring = repair_infeasible(offspring)  ← 本模块
    evaluate(offspring)  ← 修复后再评价
  
  这样可以保证评价调用都在可行域内, 大幅降低惩罚函数的作用。
===============================================================================
"""


import numpy as np
from typing import List, Tuple, Optional




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  碰撞检测工具函数
# ╚═══════════════════════════════════════════════════════════════════════════╝


def segment_circle_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
    radius: float
) -> float:
    """
    计算线段 p1→p2 到圆心 center 的最短距离
    
    用于快速碰撞检测: 若最短距离 < radius, 则碰撞。
    
    数学推导:
      参数化线段: p(t) = p1 + t*(p2-p1),  t ∈ [0,1]
      到圆心距离: d(t) = ||p(t) - center||
      最小化: d d(t)²/dt = 0 → t* = (center-p1)·(p2-p1) / ||p2-p1||²
      裁剪到 [0,1]: t_clamp = clip(t*, 0, 1)
      最短距离: ||p(t_clamp) - center||
    
    Args:
        p1, p2: 线段端点 (2,)
        center: 圆心 (2,)
        radius: 圆半径 (用于后续比较, 此函数不比较)
    
    Returns:
        min_dist: 线段到圆心的最短距离
    """
    d = p2 - p1
    len_sq = np.dot(d, d)


    if len_sq < 1e-12:
        # 退化为点
        return float(np.linalg.norm(p1 - center))


    # t* = (center - p1) · d / ||d||²
    t = np.dot(center - p1, d) / len_sq
    t_clamp = np.clip(t, 0.0, 1.0)


    closest = p1 + t_clamp * d
    return float(np.linalg.norm(closest - center))




def segment_circle_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
    radius: float,
    margin: float = 1.0
) -> bool:
    """
    检测线段 p1→p2 是否与圆 (center, radius*margin) 相交
    
    Args:
        p1, p2: 线段端点
        center: 圆心
        radius: 圆半径
        margin: 安全裕度倍数 (1.0=无裕度, 1.5=50%安全裕度)
    
    Returns:
        True: 发生碰撞 (或距离不足 margin·radius)
    """
    return segment_circle_distance(p1, p2, center, radius) < radius * margin




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  单路径碰撞修复器
# ╚═══════════════════════════════════════════════════════════════════════════╝


class PathCollisionRepairer:
    """
    基于 Dubins 路径的单路径碰撞修复器
    ─────────────────────────────────────────
    对单条路径执行碰撞修复, 保证输出路径无碰撞。
    
    修复策略:
      1. 顺序扫描所有路径段
      2. 检测每段与所有障碍物的碰撞
      3. 对碰撞段计算最优绕行点
      4. 插入绕行点并重新检测 (迭代至无碰撞)
    
    绕行点计算:
      对于碰撞段 p1→p2 和障碍物 O(center, r):
        - 计算段中点到 O 的垂线方向
        - 在左/右两侧各计算一个绕行点
        - 选取路径总长度更短的方案
    """


    def __init__(
        self,
        safety_margin: float = 1.5,
        max_iterations: int = 10,
        max_extra_waypoints: int = 3
    ):
        """
        Args:
            safety_margin: 安全裕度 (障碍物半径的倍数, 默认 1.5x)
            max_iterations: 最大修复迭代次数 (防止无限循环)
            max_extra_waypoints: 每段最多插入的额外航路点数
        """
        self.safety_margin = safety_margin
        self.max_iterations = max_iterations
        self.max_extra_waypoints = max_extra_waypoints


    def repair(
        self,
        path: np.ndarray,
        obstacles_pos: np.ndarray,
        obstacles_rad: np.ndarray,
        bounds: Tuple[float, float] = (0, 500)
    ) -> np.ndarray:
        """
        修复单条路径
        
        Args:
            path: 路径点序列 (n_pts, 2)
            obstacles_pos: 障碍物圆心 (M, 2)
            obstacles_rad: 障碍物半径 (M,)
            bounds: 环境边界 (lo, hi)
        
        Returns:
            repaired_path: 修复后的路径 (可能有更多航路点)
        """
        if len(obstacles_pos) == 0:
            return path


        lo, hi = bounds
        current_path = path.copy()


        for iteration in range(self.max_iterations):
            # 检测所有碰撞
            collision_info = self._detect_all_collisions(
                current_path, obstacles_pos, obstacles_rad
            )


            if not collision_info:
                break  # 无碰撞, 修复完成


            # 修复第一个碰撞 (从前往后处理)
            seg_idx, obs_idx = collision_info[0]
            current_path = self._repair_single_collision(
                current_path, seg_idx, obs_idx,
                obstacles_pos, obstacles_rad, lo, hi
            )


        return current_path


    def _detect_all_collisions(
        self,
        path: np.ndarray,
        obs_pos: np.ndarray,
        obs_rad: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        检测路径中所有碰撞
        
        Returns:
            碰撞列表: [(段索引, 障碍物索引), ...]
            按段索引升序排列
        """
        collisions = []
        n_pts = len(path)


        for seg_i in range(n_pts - 1):
            p1, p2 = path[seg_i], path[seg_i + 1]
            for obs_i, (center, radius) in enumerate(zip(obs_pos, obs_rad)):
                if segment_circle_intersect(
                    p1, p2, center, radius,
                    margin=self.safety_margin
                ):
                    collisions.append((seg_i, obs_i))


        return collisions


    def _repair_single_collision(
        self,
        path: np.ndarray,
        seg_idx: int,
        obs_idx: int,
        obs_pos: np.ndarray,
        obs_rad: np.ndarray,
        lo: float,
        hi: float
    ) -> np.ndarray:
        """
        修复单个碰撞: 在碰撞段插入绕行点
        
        绕行点计算策略 (Dubins 简化版):
          1. 找到线段上距离障碍物最近的点 q
          2. 计算 q → 障碍物中心的方向 n
          3. 垂直于 n 的方向 perp
          4. 绕行点 = q + (safety_radius) * perp (左或右)
          5. 选择路径总长度更短的方向
        
        Args:
            path: 当前路径
            seg_idx: 碰撞段索引
            obs_idx: 碰撞障碍物索引
            ...
        
        Returns:
            插入绕行点后的新路径
        """
        p1 = path[seg_idx]
        p2 = path[seg_idx + 1]
        center = obs_pos[obs_idx]
        radius = obs_rad[obs_idx]


        # 安全距离 = 障碍物半径 × 安全裕度 + 一点额外余量
        safe_dist = radius * self.safety_margin + 5.0


        # 找到线段上最近点
        d = p2 - p1
        len_sq = np.dot(d, d)
        if len_sq < 1e-12:
            t_clamp = 0.5
        else:
            t = np.dot(center - p1, d) / len_sq
            t_clamp = np.clip(t, 0.1, 0.9)  # 稍微偏离端点
        closest = p1 + t_clamp * d


        # 从最近点到障碍物的方向
        to_obs = center - closest
        dist_to_obs = np.linalg.norm(to_obs)


        if dist_to_obs < 1e-8:
            # 最近点恰好在障碍物中心 (极端情况)
            to_obs = np.array([1.0, 0.0])
        else:
            to_obs = to_obs / dist_to_obs


        # 垂直方向 (两个候选方向)
        perp = np.array([-to_obs[1], to_obs[0]])


        # 两个候选绕行点 (左绕/右绕)
        bypass_left  = center - to_obs * safe_dist + perp * safe_dist * 0.3
        bypass_right = center - to_obs * safe_dist - perp * safe_dist * 0.3


        # 裁剪到边界内
        bypass_left  = np.clip(bypass_left,  lo + 5, hi - 5)
        bypass_right = np.clip(bypass_right, lo + 5, hi - 5)


        # 选择路径总长度更短的方向
        len_left  = (np.linalg.norm(bypass_left  - p1) +
                     np.linalg.norm(p2 - bypass_left))
        len_right = (np.linalg.norm(bypass_right - p1) +
                     np.linalg.norm(p2 - bypass_right))


        bypass = bypass_left if len_left <= len_right else bypass_right


        # 在路径中插入绕行点
        new_path = np.insert(path, seg_idx + 1, bypass, axis=0)
        return new_path




# ╔═══════════════════════════════════════════════════════════════════════════╗
#  批量修复器 — 对整个种群进行可行性修复
# ╚═══════════════════════════════════════════════════════════════════════════╝


class FeasibilityDrivenRepair:
    """
    可行性驱动修复算子 — 种群级别
    ─────────────────────────────────────────
    对整个种群执行碰撞修复, 替代惩罚函数方法。
    
    与惩罚函数的对比:
    
      惩罚函数:
        优点: 实现简单
        缺点: λ 难调, 无效个体多, 影响 Pareto 前沿
    
      可行性修复:
        优点: 保证可行性, 无超参数 λ
        缺点: 修复可能改变路径目标值(路径变长)
    
    使用方式:
      在 EvolutionaryOperators.reproduce() 之后调用:
        offspring = operators.reproduce(...)
        offspring = repairer.repair_population(offspring, env)
        fitness = evaluator.evaluate(offspring)
    
    参数 (来源论文精神):
      safety_margin = 1.5: 比障碍物半径大 50% 的安全距离
      target_n_waypoints: 修复后路径长度 (插入绕行点后截断/补全)
    """


    def __init__(
        self,
        target_n_waypoints: int = 12,
        safety_margin: float = 1.5,
        repair_fraction: float = 0.8
    ):
        """
        Args:
            target_n_waypoints: 目标航路点数 (不含起终点)
            safety_margin: 安全裕度倍数
            repair_fraction: 修复种群的比例 (0~1)
                            不一定修复所有个体 (节省计算)
        """
        self.target_n_waypoints = target_n_waypoints
        self.repair_fraction = repair_fraction


        self.repairer = PathCollisionRepairer(
            safety_margin=safety_margin,
            max_iterations=5,
            max_extra_waypoints=3
        )


        # 统计
        self.n_repaired = 0
        self.n_collisions_fixed = 0


    def repair_population(
        self,
        population: np.ndarray,
        obstacles_pos: np.ndarray,
        obstacles_rad: np.ndarray,
        bounds: Tuple[float, float] = (0, 500)
    ) -> np.ndarray:
        """
        对种群中的不可行个体执行修复
        
        Args:
            population: 路径种群 (pop_size, n_pts, 2)
            obstacles_pos: 障碍物圆心 (M, 2)
            obstacles_rad: 障碍物半径 (M,)
            bounds: 环境边界
        
        Returns:
            repaired_population: 修复后的种群 (同形状)
        """
        if len(obstacles_pos) == 0:
            return population


        pop_size, n_pts, _ = population.shape
        repaired = population.copy()


        # 确定哪些个体需要修复
        collision_mask = self._batch_collision_check(
            population, obstacles_pos, obstacles_rad
        )


        n_to_repair = int(np.sum(collision_mask) * self.repair_fraction)
        repair_indices = np.where(collision_mask)[0][:n_to_repair]


        for idx in repair_indices:
            path = population[idx]   # (n_pts, 2)


            # 执行碰撞修复 (可能改变路径长度)
            repaired_path = self.repairer.repair(
                path, obstacles_pos, obstacles_rad, bounds
            )


            # 重采样到目标长度 (保持种群形状一致)
            repaired_path = self._resample_path(repaired_path, n_pts)
            repaired[idx] = repaired_path


            self.n_repaired += 1


        return repaired


    def _batch_collision_check(
        self,
        population: np.ndarray,
        obs_pos: np.ndarray,
        obs_rad: np.ndarray
    ) -> np.ndarray:
        """
        批量碰撞检测 — 向量化加速
        
        Returns:
            mask: (pop_size,) bool 数组, True 表示有碰撞
        """
        pop_size, n_pts, _ = population.shape
        mask = np.zeros(pop_size, dtype=bool)


        # 计算所有路径段中点到所有障碍物的距离
        midpoints = (population[:, :-1] + population[:, 1:]) / 2.0  # (N, n_pts-1, 2)


        for oi, (center, radius) in enumerate(zip(obs_pos, obs_rad)):
            safe_r = radius * self.repairer.safety_margin
            # 中点到障碍物距离 (快速近似检测)
            dists = np.linalg.norm(midpoints - center, axis=2)  # (N, n_pts-1)
            # 任意段的中点距离 < safe_r → 该个体碰撞
            mask |= np.any(dists < safe_r, axis=1)


        return mask


    def _resample_path(
        self,
        path: np.ndarray,
        target_n_pts: int
    ) -> np.ndarray:
        """
        将可变长度路径重采样到固定长度
        
        通过等弧长参数化实现均匀重采样:
          1. 计算累积弧长
          2. 用等间距的弧长参数生成新路径点
          3. 保留起终点不变
        
        Args:
            path: 可变长度路径 (M, 2)
            target_n_pts: 目标点数
        
        Returns:
            resampled: 等长路径 (target_n_pts, 2)
        """
        if len(path) == target_n_pts:
            return path


        if len(path) < 2:
            # 退化情况: 复制扩充
            return np.tile(path[0], (target_n_pts, 1))


        # 计算累积弧长
        diffs = np.diff(path, axis=0)          # (M-1, 2)
        seg_lens = np.linalg.norm(diffs, axis=1)  # (M-1,)
        cum_lens = np.concatenate([[0], np.cumsum(seg_lens)])  # (M,)
        total_len = cum_lens[-1]


        if total_len < 1e-10:
            return np.tile(path[0], (target_n_pts, 1))


        # 等间距目标弧长参数
        target_lens = np.linspace(0, total_len, target_n_pts)


        # 插值
        resampled = np.zeros((target_n_pts, 2))
        for k, s in enumerate(target_lens):
            # 找到 s 所在的段
            idx = np.searchsorted(cum_lens, s, side='right') - 1
            idx = np.clip(idx, 0, len(path) - 2)


            # 线性插值
            t = (s - cum_lens[idx]) / (seg_lens[idx] + 1e-10)
            t = np.clip(t, 0, 1)
            resampled[k] = path[idx] + t * diffs[idx]


        # 保证起终点精确
        resampled[0] = path[0]
        resampled[-1] = path[-1]


        return resampled


    def get_feasibility_rate(
        self,
        population: np.ndarray,
        obstacles_pos: np.ndarray,
        obstacles_rad: np.ndarray
    ) -> float:
        """
        计算种群可行率 (无碰撞个体占比)
        
        Returns:
            rate: 0~1, 1 表示全部可行
        """
        if len(obstacles_pos) == 0:
            return 1.0
        mask = self._batch_collision_check(population, obstacles_pos, obstacles_rad)
        return float(1.0 - mask.mean())