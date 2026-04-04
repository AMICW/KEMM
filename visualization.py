"""Thin compatibility layer for the refactored benchmark-visualization stack.

Actual benchmark visualization helpers now live in
``apps.reporting.benchmark_visualization``.
This module remains only to preserve older import paths.
"""

from apps.reporting.benchmark_visualization import (
    AblationStudyPlots,
    AlgorithmMechanismPlots,
    PerformanceComparisonPlots,
    ProcessAnalysisPlots,
    StatisticalAnalysisPlots,
    generate_all_figures,
)

__all__ = [
    "AblationStudyPlots",
    "AlgorithmMechanismPlots",
    "PerformanceComparisonPlots",
    "ProcessAnalysisPlots",
    "StatisticalAnalysisPlots",
    "generate_all_figures",
]
