"""ship_simulation/scenario/encounter.py

本文件只负责定义场景数据结构，不负责生成场景。

这样拆分的好处是：
- `generator.py` 专注“怎么生成”
- `encounter.py` 专注“场景对象长什么样”
- 后续无论是代码生成场景、人工构造场景，还是从外部文件读取场景，
  都可以统一落到这里的数据结构上
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ship_simulation.core.ship_model import ShipState


@dataclass
class VesselAgent:
    """场景中的一艘船舶静态描述。"""

    name: str
    initial_state: ShipState

    # 本船通常有 goal，目标船在当前 MVP 中一般没有显式终点
    goal: np.ndarray | None = None
    color: str = "tab:blue"


@dataclass
class EncounterScenario:
    """多船会遇场景对象。

    它是优化模块、仿真模块和可视化模块共享的场景容器。
    """

    name: str
    own_ship: VesselAgent
    target_ships: List[VesselAgent] = field(default_factory=list)
    area: Tuple[float, float, float, float] = (-1000.0, 9000.0, -4000.0, 4000.0)
