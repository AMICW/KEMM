"""KEMM project package.

This package reorganizes the repository into reusable modules while
keeping backward compatibility with the original root-level scripts.
"""

from kemm.algorithms.kemm import KEMM_DMOEA_Improved
from kemm.adapters import BenchmarkPriorAdapter
from kemm.benchmark.metrics import PerformanceMetrics
from kemm.benchmark.problems import DynamicTestProblems
from kemm.core.types import KEMMChangeDiagnostics, KEMMConfig

__all__ = [
    "BenchmarkPriorAdapter",
    "DynamicTestProblems",
    "KEMMChangeDiagnostics",
    "KEMMConfig",
    "KEMM_DMOEA_Improved",
    "PerformanceMetrics",
]
