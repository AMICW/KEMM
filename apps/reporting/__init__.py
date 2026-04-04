"""Unified reporting exports for both benchmark and ship-simulation workflows."""

from reporting_config import BenchmarkPlotConfig, PublicationStyle, ShipPlotConfig

from .benchmark_visualization import (
    AblationStudyPlots,
    AlgorithmMechanismPlots,
    BenchmarkFigurePayload,
    PerformanceComparisonPlots,
    ProcessAnalysisPlots,
    StatisticalAnalysisPlots,
    generate_all_figures,
)
from kemm.reporting import build_report_paths, export_benchmark_report
from ship_simulation.visualization import (
    ExperimentSeries,
    TrajectoryAnimator,
    save_convergence_plot,
    save_normalized_objective_bars,
    save_pareto_scatter,
    save_risk_bars,
    save_risk_time_series,
    save_speed_profiles,
    save_summary_dashboard,
    save_trajectory_comparison,
)

__all__ = [
    "AblationStudyPlots",
    "AlgorithmMechanismPlots",
    "BenchmarkFigurePayload",
    "BenchmarkPlotConfig",
    "ExperimentSeries",
    "PerformanceComparisonPlots",
    "PublicationStyle",
    "ProcessAnalysisPlots",
    "ShipPlotConfig",
    "StatisticalAnalysisPlots",
    "TrajectoryAnimator",
    "build_report_paths",
    "export_benchmark_report",
    "generate_all_figures",
    "save_convergence_plot",
    "save_normalized_objective_bars",
    "save_pareto_scatter",
    "save_risk_bars",
    "save_risk_time_series",
    "save_speed_profiles",
    "save_summary_dashboard",
    "save_trajectory_comparison",
]
