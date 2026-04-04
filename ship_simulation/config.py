"""ship_simulation/config.py

本文件负责集中管理整个船舶轨迹规划仿真系统的配置参数。

设计目的：
1. 将船舶参数、环境参数、碰撞风险参数、仿真参数统一收口。
2. 让优化问题、场景生成、可视化模块共享同一套配置来源。
3. 为后续从 Nomoto 一阶模型升级到 MMG、从恒定流场升级到网格场预留扩展位。

建议使用方式：
- 绝大多数模块只依赖 `ProblemConfig`
- 细分模块内部再读取 `ship / environment / domain / simulation` 子配置
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ShipConfig:
    """船舶物理参数与控制参数。

    这里使用的是 MVP 阶段的简化参数集合，既能驱动 Nomoto 一阶模型，
    也方便未来替换为更复杂的动力学模型。
    """

    # 船体尺度参数，主要用于运动学尺度和船舶域风险计算
    length: float = 120.0
    beam: float = 20.0
    draft: float = 7.0

    # Nomoto 一阶模型参数
    # T: 航向响应时间常数，越大表示转向响应越慢
    # K: 转向增益，影响舵令对角速度的作用强弱
    nomoto_k: float = 1.0
    nomoto_t: float = 25.0

    # 简化航迹跟踪控制器参数
    heading_gain: float = 1.6
    max_turn_rate_deg: float = 3.0

    # 速度动态参数
    speed_time_constant: float = 35.0
    max_speed: float = 9.0
    min_speed: float = 2.0


@dataclass
class EnvironmentConfig:
    """环境配置。

    当前版本只支持恒定海流和恒定风场。
    这种建模方式足够用于算法早期验证，同时接口上也容易替换成时空变化环境。
    """

    # 海流速度与方向（世界坐标系，角度制）
    current_speed: float = 0.6
    current_direction_deg: float = 35.0

    # 风速与方向（世界坐标系，角度制）
    wind_speed: float = 6.0
    wind_direction_deg: float = 75.0

    # 风导致的等效漂移比例系数
    # 这里不直接做空气动力学建模，而是用一个轻量代理参数表示风对地面航迹的影响
    wind_drift_coefficient: float = 0.015


@dataclass
class DomainConfig:
    """船舶域碰撞风险模型参数。

    本项目采用椭圆船舶域思想，并通过前后左右不同尺度因子体现非对称性：
    - 前方一般需要更大安全距离
    - 左右舷安全要求可不同
    """

    # 前、后、右、左方向上的尺度放大因子
    forward_factor: float = 3.5
    aft_factor: float = 1.6
    starboard_factor: float = 1.8
    port_factor: float = 1.4

    # soft_margin 用于把“硬边界船舶域”变成“连续风险场”
    # 数值越大，风险衰减越柔和
    soft_margin: float = 1.35


@dataclass
class SimulationConfig:
    """仿真积分参数与场景空间范围。"""

    # 积分步长，单位秒
    dt: float = 5.0

    # 单次仿真最大时长，单位秒
    horizon: float = 1800.0

    # 判定到达目标点的容许半径
    arrival_tolerance: float = 80.0

    # 场景二维范围：(xmin, xmax, ymin, ymax)
    area: Tuple[float, float, float, float] = (-1000.0, 9000.0, -4000.0, 4000.0)


@dataclass
class ProblemConfig:
    """面向优化器的总配置对象。

    它是整个系统对外最重要的配置入口：
    - 决策变量规模由它决定
    - 速度边界、惩罚项规模由它决定
    - 子模块配置通过它层层分发
    """

    # 中间航路点数量
    # 决策向量长度 = num_intermediate_waypoints * 3
    num_intermediate_waypoints: int = 3

    # 为后续加权求和或偏好引导预留的目标权重
    objective_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # 每段速度的取值边界
    speed_bounds: Tuple[float, float] = (3.0, 8.5)

    # 当优化器给出越界变量时，用于提高目标值的惩罚项
    # 这样做有利于与无约束多目标算法直接对接
    penalty_out_of_bounds: float = 1e5

    # 若轨迹在仿真终止时仍未到达目标点，则按剩余距离追加惩罚
    # 这可以避免优化器把“没到终点但碰撞风险低”的解误判为好解
    terminal_fuel_penalty_per_meter: float = 2.0
    terminal_time_penalty_per_meter: float = 0.18
    terminal_risk_penalty_per_meter: float = 0.002
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ship: ShipConfig = field(default_factory=ShipConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)


@dataclass
class KEMMConfig:
    """KEMM 在船舶仿真问题中的运行参数。

    这里不改动 benchmark 侧算法主体，而是在 ship_simulation 内部补一层
    面向工程使用的配置，控制：
    - 种群规模与迭代代数
    - 是否把直线初始解注入初代
    - 是否周期性触发 KEMM 的 memory/predict/transfer 响应机制
    """

    pop_size: int = 80
    generations: int = 60
    refresh_interval: int = 8
    seed: int = 42
    inject_initial_guess: bool = True
    initial_guess_copies: int = 6
    initial_guess_jitter_ratio: float = 0.04


@dataclass
class DemoConfig:
    """演示脚本层的配置。

    单独拆出这一层，是为了避免把“仿真物理参数”和“示例运行参数”
    混在一起，方便以后扩展为批量实验脚本。
    """

    optimizer_name: str = "kemm"
    random_search_samples: int = 80
    random_search_seed: int = 42
    kemm: KEMMConfig = field(default_factory=KEMMConfig)


def build_default_config() -> ProblemConfig:
    """构造一套默认配置。

    如果用户暂时不想关心参数调优，可以直接调用这个函数开始仿真。
    """

    return ProblemConfig()


def build_default_demo_config() -> DemoConfig:
    """构造演示脚本的默认配置。"""

    return DemoConfig()
