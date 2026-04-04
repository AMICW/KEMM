"""Thin compatibility layer for Pareto front drift prediction.

真实实现现在位于 `kemm.core.drift`。
保留本文件只是为了兼容旧导入路径。
"""

from kemm.core.drift import LightweightGPR, ParetoFrontDriftPredictor

__all__ = ["LightweightGPR", "ParetoFrontDriftPredictor"]
