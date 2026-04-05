"""ship_simulation 配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ShipConfig:
    """船舶物理参数与简化控制参数。"""

    length: float = 120.0
    beam: float = 20.0
    draft: float = 7.0
    nomoto_k: float = 1.0
    nomoto_t: float = 25.0
    heading_gain: float = 1.6
    max_turn_rate_deg: float = 3.0
    speed_time_constant: float = 35.0
    max_speed: float = 9.0
    min_speed: float = 2.0
    max_commanded_yaw_rate_deg: float = 4.0


@dataclass
class EnvironmentConfig:
    """基础环境参数。"""

    current_speed: float = 0.6
    current_direction_deg: float = 35.0
    wind_speed: float = 6.0
    wind_direction_deg: float = 75.0
    wind_drift_coefficient: float = 0.015


@dataclass
class DomainConfig:
    """船舶域参数。"""

    forward_factor: float = 3.5
    aft_factor: float = 1.6
    starboard_factor: float = 1.8
    port_factor: float = 1.4
    soft_margin: float = 1.35


@dataclass
class SimulationConfig:
    """积分与空间范围参数。"""

    dt: float = 5.0
    horizon: float = 1800.0
    arrival_tolerance: float = 80.0
    area: Tuple[float, float, float, float] = (-1000.0, 9000.0, -4000.0, 4000.0)


@dataclass
class EpisodeConfig:
    """滚动重规划参数。"""

    enabled: bool = True
    local_horizon: float = 360.0
    execution_horizon: float = 180.0
    max_replans: int = 8
    stop_on_high_risk: float = 2.8
    snapshot_count: int = 4


@dataclass
class ProblemConfig:
    """优化问题总配置。"""

    num_intermediate_waypoints: int = 3
    objective_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    speed_bounds: Tuple[float, float] = (3.0, 8.5)
    penalty_out_of_bounds: float = 1e5
    terminal_fuel_penalty_per_meter: float = 2.0
    terminal_time_penalty_per_meter: float = 0.18
    terminal_risk_penalty_per_meter: float = 0.002
    soft_clearance_penalty_per_meter: float = 2.0
    hard_clearance_penalty_per_meter: float = 16.0
    intrusion_risk_penalty_per_second: float = 0.02
    domain_risk_weight: float = 0.55
    dcpa_risk_weight: float = 0.2
    obstacle_risk_weight: float = 0.15
    environment_risk_weight: float = 0.1
    tcpa_decay_seconds: float = 480.0
    safety_clearance: float = 180.0
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ship: ShipConfig = field(default_factory=ShipConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)


@dataclass
class KEMMConfig:
    """ship 主线使用的 KEMM 运行参数。"""

    pop_size: int = 48
    generations: int = 24
    refresh_interval: int = 8
    seed: int = 42
    inject_initial_guess: bool = True
    initial_guess_copies: int = 4
    initial_guess_jitter_ratio: float = 0.04
    use_change_response: bool = False


@dataclass
class DemoConfig:
    """ship 演示与批量报告配置。"""

    optimizer_name: str = "kemm"
    random_search_samples: int = 48
    random_search_seed: int = 42
    evolutionary_baseline_pop_size: int = 36
    evolutionary_baseline_generations: int = 16
    n_runs: int = 3
    plot_preset: str = "paper"
    appendix_plots: bool = False
    kemm: KEMMConfig = field(default_factory=KEMMConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)


def build_default_config() -> ProblemConfig:
    """返回默认问题配置。"""

    return ProblemConfig()


def build_default_demo_config() -> DemoConfig:
    """返回默认 demo/report 配置。"""

    return DemoConfig()
