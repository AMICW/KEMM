"""Thin compatibility layer for the refactored benchmark stack.

Actual implementations now live under ``kemm.algorithms`` and ``kemm.benchmark``.
This module remains only to preserve older import paths.
"""

from kemm.algorithms import (
    BaseDMOEA,
    KF_DMOEA,
    KEMM_DMOEA_Improved,
    MMTL_DMOEA,
    PPS_DMOEA,
    RI_DMOEA,
    SVR_DMOEA,
    Tr_DMOEA,
)
from kemm.benchmark import DynamicTestProblems, PerformanceMetrics

__all__ = [
    "BaseDMOEA",
    "DynamicTestProblems",
    "KF_DMOEA",
    "KEMM_DMOEA_Improved",
    "MMTL_DMOEA",
    "PPS_DMOEA",
    "PerformanceMetrics",
    "RI_DMOEA",
    "SVR_DMOEA",
    "Tr_DMOEA",
]
