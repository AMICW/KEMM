"""Thin compatibility layer for geodesic-flow transfer.

真实实现现在位于 `kemm.core.transfer`。
保留本文件只是为了兼容旧导入路径。
"""

from kemm.core.transfer import GrassmannGeodesicFlow, ManifoldTransferLearning, MultiSourceTransfer

__all__ = ["GrassmannGeodesicFlow", "ManifoldTransferLearning", "MultiSourceTransfer"]
