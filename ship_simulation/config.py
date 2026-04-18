"""ship_simulation 配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from kemm.core.types import KEMMConfig as RuntimeKEMMConfig


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
class ScenarioTuningConfig:
    """单个场景的几何与环境生成调参。"""

    family_name: str = "default"
    scenario_seed: int | None = None
    difficulty_scale: float = 1.0
    geometry_jitter_m: float = 0.0
    traffic_heading_jitter_deg: float = 0.0
    current_direction_jitter_deg: float = 0.0
    start_offset: Tuple[float, float] = (0.0, 0.0)
    goal_offset: Tuple[float, float] = (0.0, 0.0)
    own_speed_scale: float = 1.0
    target_speed_scale: float = 1.0
    scalar_amplitude_scale: float = 1.0
    vector_speed_scale: float = 1.0
    circular_radius_scale: float = 1.0
    polygon_scale: float = 1.0
    target_limit: int | None = None


@dataclass
class HarborClutterTuningConfig(ScenarioTuningConfig):
    """高密障碍港区的专项生成参数。"""

    family_name: str = "harbor_clutter"
    start_offset: Tuple[float, float] = (0.0, 220.0)
    goal_offset: Tuple[float, float] = (0.0, -220.0)
    own_speed_scale: float = 1.04
    target_speed_scale: float = 0.9
    scalar_amplitude_scale: float = 0.82
    vector_speed_scale: float = 0.86
    circular_radius_scale: float = 0.68
    polygon_scale: float = 0.82
    target_limit: int | None = 2
    circular_obstacle_limit: int | None = 6
    polygon_obstacle_limit: int | None = 3
    channel_width_scale: float = 1.15


@dataclass
class ScenarioGenerationConfig:
    """ship 场景生成的可调参数集合。"""

    head_on: ScenarioTuningConfig = field(default_factory=ScenarioTuningConfig)
    crossing: ScenarioTuningConfig = field(default_factory=ScenarioTuningConfig)
    overtaking: ScenarioTuningConfig = field(default_factory=ScenarioTuningConfig)
    harbor_clutter: HarborClutterTuningConfig = field(default_factory=HarborClutterTuningConfig)


@dataclass(frozen=True)
class ScenarioChangeStepConfig:
    """在某个重规划 step 生效的动态实验变化。"""

    step_index: int
    label: str
    notes: str = ""
    scalar_amplitude_scale: float = 1.0
    vector_speed_scale: float = 1.0
    current_speed_scale: float = 1.0
    target_speed_scale: float = 1.0
    target_heading_delta_deg: float = 0.0
    channel_width_scale: float = 1.0
    inject_channel_closure: bool = False
    closure_center_x: float | None = None
    closure_width: float = 260.0
    closure_gap_center_ratio: float = 0.5
    closure_gap_span_ratio: float = 0.24


@dataclass
class ScenarioExperimentConfig:
    """ship 物理实验的动态变化配置。"""

    enabled: bool = False
    profile_name: str = "baseline"
    difficulty_label: str = "baseline"
    recurrence_pattern: tuple[str, ...] = ()
    change_schedule: tuple[ScenarioChangeStepConfig, ...] = ()


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
    local_terminal_penalty_scale: float = 0.25
    soft_clearance_penalty_per_meter: float = 2.0
    hard_clearance_penalty_per_meter: float = 16.0
    fuel_safety_penalty_weight: float = 2.5
    time_safety_penalty_weight: float = 0.25
    risk_safety_penalty_weight: float = 0.025
    intrusion_risk_penalty_per_second: float = 0.02
    domain_risk_weight: float = 0.55
    dcpa_risk_weight: float = 0.2
    obstacle_risk_weight: float = 0.15
    environment_risk_weight: float = 0.1
    tcpa_decay_seconds: float = 480.0
    safety_clearance: float = 180.0
    population_evaluation_cache: bool = True
    population_cache_decimals: int = 8
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ship: ShipConfig = field(default_factory=ShipConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    scenario_generation: ScenarioGenerationConfig = field(default_factory=ScenarioGenerationConfig)
    experiment: ScenarioExperimentConfig = field(default_factory=ScenarioExperimentConfig)


@dataclass
class KEMMConfig:
    """ship 主线使用的 KEMM 运行参数。"""

    pop_size: int = 30
    generations: int = 14
    refresh_interval: int = 8
    seed: int = 42
    inject_initial_guess: bool = True
    initial_guess_copies: int = 4
    initial_guess_jitter_ratio: float = 0.04
    inject_heuristic_detours: bool = True
    heuristic_detour_limit: int = 6
    heuristic_detour_offset_scale: float = 1.0
    use_change_response: bool = True
    reuse_solver_state_across_replans: bool = False
    runtime: RuntimeKEMMConfig = field(default_factory=lambda: RuntimeKEMMConfig(benchmark_aware_prior=False))


@dataclass(frozen=True)
class ScenarioSolveProfile:
    random_search_samples: int
    evolutionary_baseline_pop_size: int
    evolutionary_baseline_generations: int
    kemm_pop_size: int
    kemm_generations: int
    kemm_initial_guess_copies: int
    kemm_heuristic_detour_limit: int
    kemm_reuse_solver_state: bool
    local_horizon: float
    execution_horizon: float
    max_replans: int
    objective_weights: Tuple[float, float, float]
    safety_clearance: float
    soft_clearance_penalty_per_meter: float
    hard_clearance_penalty_per_meter: float
    time_safety_penalty_weight: float
    risk_safety_penalty_weight: float
    domain_risk_weight: float


def _uniform_solve_profile(
    demo: "DemoConfig | None" = None,
    problem: ProblemConfig | None = None,
) -> ScenarioSolveProfile:
    demo_cfg = demo or DemoConfig.__new__(DemoConfig)
    if demo is None:
        demo_cfg.random_search_samples = 48
        demo_cfg.evolutionary_baseline_pop_size = 36
        demo_cfg.evolutionary_baseline_generations = 16
        demo_cfg.kemm = KEMMConfig()
        demo_cfg.episode = EpisodeConfig()
    problem_cfg = problem or ProblemConfig()
    return ScenarioSolveProfile(
        random_search_samples=int(demo_cfg.random_search_samples),
        evolutionary_baseline_pop_size=int(demo_cfg.evolutionary_baseline_pop_size),
        evolutionary_baseline_generations=int(demo_cfg.evolutionary_baseline_generations),
        kemm_pop_size=int(demo_cfg.kemm.pop_size),
        kemm_generations=int(demo_cfg.kemm.generations),
        kemm_initial_guess_copies=int(demo_cfg.kemm.initial_guess_copies),
        kemm_heuristic_detour_limit=int(demo_cfg.kemm.heuristic_detour_limit),
        kemm_reuse_solver_state=bool(demo_cfg.kemm.reuse_solver_state_across_replans),
        local_horizon=float(demo_cfg.episode.local_horizon),
        execution_horizon=float(demo_cfg.episode.execution_horizon),
        max_replans=int(demo_cfg.episode.max_replans),
        objective_weights=tuple(float(value) for value in problem_cfg.objective_weights),
        safety_clearance=float(problem_cfg.safety_clearance),
        soft_clearance_penalty_per_meter=float(problem_cfg.soft_clearance_penalty_per_meter),
        hard_clearance_penalty_per_meter=float(problem_cfg.hard_clearance_penalty_per_meter),
        time_safety_penalty_weight=float(problem_cfg.time_safety_penalty_weight),
        risk_safety_penalty_weight=float(problem_cfg.risk_safety_penalty_weight),
        domain_risk_weight=float(problem_cfg.domain_risk_weight),
    )


def _build_full_tuned_profiles() -> dict[str, ScenarioSolveProfile]:
    base_problem = ProblemConfig()
    base_demo = DemoConfig.__new__(DemoConfig)
    base_demo.random_search_samples = 48
    base_demo.evolutionary_baseline_pop_size = 36
    base_demo.evolutionary_baseline_generations = 16
    base_demo.kemm = KEMMConfig()
    base_demo.episode = EpisodeConfig()
    legacy = _uniform_solve_profile(base_demo, base_problem)
    return {
        "head_on": ScenarioSolveProfile(
            random_search_samples=32,
            evolutionary_baseline_pop_size=28,
            evolutionary_baseline_generations=10,
            kemm_pop_size=24,
            kemm_generations=10,
            kemm_initial_guess_copies=3,
            kemm_heuristic_detour_limit=2,
            kemm_reuse_solver_state=True,
            local_horizon=300.0,
            execution_horizon=240.0,
            max_replans=6,
            objective_weights=(1.0, 1.0, 1.10),
            safety_clearance=legacy.safety_clearance,
            soft_clearance_penalty_per_meter=legacy.soft_clearance_penalty_per_meter,
            hard_clearance_penalty_per_meter=legacy.hard_clearance_penalty_per_meter,
            time_safety_penalty_weight=legacy.time_safety_penalty_weight,
            risk_safety_penalty_weight=legacy.risk_safety_penalty_weight,
            domain_risk_weight=legacy.domain_risk_weight,
        ),
        "crossing": ScenarioSolveProfile(
            random_search_samples=40,
            evolutionary_baseline_pop_size=32,
            evolutionary_baseline_generations=12,
            kemm_pop_size=30,
            kemm_generations=12,
            kemm_initial_guess_copies=4,
            kemm_heuristic_detour_limit=4,
            kemm_reuse_solver_state=False,
            local_horizon=420.0,
            execution_horizon=210.0,
            max_replans=6,
            objective_weights=(1.0, 0.95, 1.35),
            safety_clearance=210.0,
            soft_clearance_penalty_per_meter=2.6,
            hard_clearance_penalty_per_meter=18.0,
            time_safety_penalty_weight=legacy.time_safety_penalty_weight,
            risk_safety_penalty_weight=0.035,
            domain_risk_weight=0.60,
        ),
        "overtaking": ScenarioSolveProfile(
            random_search_samples=32,
            evolutionary_baseline_pop_size=28,
            evolutionary_baseline_generations=10,
            kemm_pop_size=26,
            kemm_generations=10,
            kemm_initial_guess_copies=3,
            kemm_heuristic_detour_limit=3,
            kemm_reuse_solver_state=True,
            local_horizon=320.0,
            execution_horizon=240.0,
            max_replans=6,
            objective_weights=(1.0, 0.95, 1.25),
            safety_clearance=200.0,
            soft_clearance_penalty_per_meter=2.4,
            hard_clearance_penalty_per_meter=18.0,
            time_safety_penalty_weight=0.30,
            risk_safety_penalty_weight=0.032,
            domain_risk_weight=0.60,
        ),
        "harbor_clutter": ScenarioSolveProfile(
            random_search_samples=40,
            evolutionary_baseline_pop_size=30,
            evolutionary_baseline_generations=12,
            kemm_pop_size=28,
            kemm_generations=12,
            kemm_initial_guess_copies=4,
            kemm_heuristic_detour_limit=4,
            kemm_reuse_solver_state=False,
            local_horizon=420.0,
            execution_horizon=240.0,
            max_replans=6,
            objective_weights=(1.0, 1.0, 1.45),
            safety_clearance=220.0,
            soft_clearance_penalty_per_meter=2.8,
            hard_clearance_penalty_per_meter=18.0,
            time_safety_penalty_weight=legacy.time_safety_penalty_weight,
            risk_safety_penalty_weight=0.040,
            domain_risk_weight=0.62,
        ),
    }


@dataclass
class ScenarioSolveProfiles:
    active_profile_name: str = "legacy_uniform"
    legacy_uniform: dict[str, ScenarioSolveProfile] = field(default_factory=dict)
    full_tuned: dict[str, ScenarioSolveProfile] = field(default_factory=_build_full_tuned_profiles)

    def __post_init__(self) -> None:
        if not self.legacy_uniform:
            uniform = _uniform_solve_profile()
            self.legacy_uniform = {
                "head_on": uniform,
                "crossing": uniform,
                "overtaking": uniform,
                "harbor_clutter": uniform,
            }

    def resolve(
        self,
        scenario_key: str,
        *,
        demo: "DemoConfig | None" = None,
        problem: ProblemConfig | None = None,
    ) -> ScenarioSolveProfile:
        key = str(scenario_key)
        if self.active_profile_name == "legacy_uniform" and demo is not None and problem is not None:
            return _uniform_solve_profile(demo, problem)
        groups = {
            "legacy_uniform": self.legacy_uniform,
            "full_tuned": self.full_tuned,
        }
        selected = groups.get(self.active_profile_name)
        if selected is None:
            raise ValueError(f"Unsupported scenario solve profile set: {self.active_profile_name}")
        if key not in selected:
            raise KeyError(f"Unknown scenario key for solve profile: {key}")
        return selected[key]


@dataclass
class DemoConfig:
    """ship 演示与批量报告配置。"""

    optimizer_name: str = "kemm"
    random_search_samples: int = 48
    random_search_seed: int = 42
    evolutionary_baseline_pop_size: int = 36
    evolutionary_baseline_generations: int = 16
    n_runs: int = 3
    report_algorithms: tuple[str, ...] = ("kemm", "nsga_style", "random")
    plot_preset: str = "paper"
    appendix_plots: bool = False
    kemm: KEMMConfig = field(default_factory=KEMMConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    scenario_profiles: ScenarioSolveProfiles = field(default_factory=ScenarioSolveProfiles)
    episode_cache_enabled: bool = True
    render_workers: int = 2


def build_default_config() -> ProblemConfig:
    """返回默认问题配置。"""

    return ProblemConfig()


def build_default_demo_config() -> DemoConfig:
    """返回默认 demo/report 配置。"""

    demo = DemoConfig()
    demo.scenario_profiles.active_profile_name = "full_tuned"
    return demo


def build_experiment_profile(profile_name: str) -> ScenarioExperimentConfig:
    """返回与 KEMM 机制对齐的 ship 动态实验 profile。"""

    key = profile_name.strip().lower()
    if key in {"", "baseline"}:
        return ScenarioExperimentConfig(enabled=False, profile_name="baseline", difficulty_label="baseline")
    if key == "drift":
        return ScenarioExperimentConfig(
            enabled=True,
            profile_name="drift",
            difficulty_label="medium",
            recurrence_pattern=("baseline", "current_build_up", "traffic_drift"),
            change_schedule=(
                ScenarioChangeStepConfig(
                    step_index=1,
                    label="Current build-up",
                    notes="Strengthen flow layers and mild traffic acceleration.",
                    scalar_amplitude_scale=1.08,
                    vector_speed_scale=1.16,
                    current_speed_scale=1.12,
                    target_speed_scale=1.04,
                ),
                ScenarioChangeStepConfig(
                    step_index=3,
                    label="Traffic intent drift",
                    notes="Traffic headings drift while environmental forcing remains elevated.",
                    scalar_amplitude_scale=1.15,
                    vector_speed_scale=1.22,
                    current_speed_scale=1.18,
                    target_speed_scale=1.08,
                    target_heading_delta_deg=11.0,
                ),
            ),
        )
    if key == "shock":
        return ScenarioExperimentConfig(
            enabled=True,
            profile_name="shock",
            difficulty_label="hard",
            recurrence_pattern=("baseline", "closure_shock"),
            change_schedule=(
                ScenarioChangeStepConfig(
                    step_index=2,
                    label="Sudden channel closure",
                    notes="Inject a temporary closure barrier and stronger traffic deviation.",
                    scalar_amplitude_scale=1.12,
                    vector_speed_scale=1.18,
                    current_speed_scale=1.14,
                    target_speed_scale=1.06,
                    target_heading_delta_deg=14.0,
                    channel_width_scale=0.92,
                    inject_channel_closure=True,
                    closure_width=320.0,
                    closure_gap_center_ratio=0.56,
                    closure_gap_span_ratio=0.18,
                ),
            ),
        )
    if key == "recurring_harbor":
        return ScenarioExperimentConfig(
            enabled=True,
            profile_name="recurring_harbor",
            difficulty_label="medium-hard",
            recurrence_pattern=("harbor_base", "ebb_shift", "familiar_recovery"),
            change_schedule=(
                ScenarioChangeStepConfig(
                    step_index=1,
                    label="Ebb shift",
                    notes="Moderate environmental drift that preserves the same harbor topology.",
                    scalar_amplitude_scale=1.05,
                    vector_speed_scale=1.10,
                    current_speed_scale=1.08,
                    target_speed_scale=1.03,
                    target_heading_delta_deg=6.0,
                ),
                ScenarioChangeStepConfig(
                    step_index=3,
                    label="Familiar recovery",
                    notes="Conditions partially return, matching the recurring-scene memory hypothesis.",
                    scalar_amplitude_scale=0.98,
                    vector_speed_scale=0.96,
                    current_speed_scale=0.96,
                    target_speed_scale=0.99,
                    target_heading_delta_deg=-4.0,
                ),
            ),
        )
    raise ValueError(f"Unsupported experiment profile: {profile_name}")


def apply_experiment_profile(config: ProblemConfig, profile_name: str) -> ProblemConfig:
    """原地配置 ship 动态实验 profile，并返回 config 以便链式调用。"""

    config.experiment = build_experiment_profile(profile_name)
    return config
