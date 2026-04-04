"""Thin compatibility layer for adaptive operator selection.

真实实现现在位于 `kemm.core.adaptive`。
保留本文件只是为了兼容旧导入路径。
"""

from kemm.core.adaptive import AdaptiveOperatorSelector, ParetoFrontDriftDetector, UCB1Bandit

__all__ = ["AdaptiveOperatorSelector", "ParetoFrontDriftDetector", "UCB1Bandit"]
