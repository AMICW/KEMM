"""Thin compatibility layer for compressed memory.

真实实现现在位于 `kemm.core.memory`。
保留本文件只是为了兼容旧导入路径。
"""

from kemm.core.memory import LightweightVAE, VAECompressedMemory

__all__ = ["LightweightVAE", "VAECompressedMemory"]
