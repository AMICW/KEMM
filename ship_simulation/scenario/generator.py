"""ship_simulation/scenario/generator.py

本文件负责用纯代码生成经典船舶会遇场景。

当前内置：
- 对遇（Head-on）
- 交叉相遇（Crossing）
- 追越（Overtaking）

后续可继续扩展：
- 多目标船复杂会遇
- 编队/拥挤水域场景
- 含障碍物或航道约束的场景
"""

from __future__ import annotations

from math import pi

import numpy as np

from ship_simulation.config import ProblemConfig
from ship_simulation.core.ship_model import ShipState
from ship_simulation.scenario.encounter import EncounterScenario, VesselAgent


class ScenarioGenerator:
    """生成典型 COLREG 风格会遇场景。"""

    def __init__(self, config: ProblemConfig):
        self.config = config

    def head_on(self) -> EncounterScenario:
        """生成对遇场景。

        本船与目标船大致相向而行，是最典型的迎面相遇工况。
        """

        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=0.0, y=0.0, heading=0.0, speed=6.5),
            goal=np.array([7000.0, 0.0], dtype=float),
            color="tab:blue",
        )
        target = VesselAgent(
            name="Target A",
            initial_state=ShipState(x=7200.0, y=300.0, heading=pi, speed=6.0),
            color="tab:red",
        )
        return EncounterScenario("Head-on", own_ship=own, target_ships=[target], area=self.config.simulation.area)

    def crossing(self) -> EncounterScenario:
        """生成交叉相遇场景。"""

        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=0.0, y=-400.0, heading=0.0, speed=6.3),
            goal=np.array([7200.0, -400.0], dtype=float),
            color="tab:blue",
        )
        target = VesselAgent(
            name="Target A",
            initial_state=ShipState(x=3200.0, y=-3200.0, heading=pi / 2.0, speed=5.7),
            color="tab:orange",
        )
        return EncounterScenario("Crossing", own_ship=own, target_ships=[target], area=self.config.simulation.area)

    def overtaking(self) -> EncounterScenario:
        """生成追越场景。"""

        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=0.0, y=0.0, heading=0.0, speed=7.2),
            goal=np.array([7600.0, 0.0], dtype=float),
            color="tab:blue",
        )
        target = VesselAgent(
            name="Target A",
            initial_state=ShipState(x=1800.0, y=120.0, heading=0.0, speed=4.5),
            color="tab:green",
        )
        return EncounterScenario("Overtaking", own_ship=own, target_ships=[target], area=self.config.simulation.area)

    def generate(self, scenario_type: str) -> EncounterScenario:
        """统一场景工厂入口。"""

        key = scenario_type.strip().lower()
        # 这里允许多种字符串写法，减少外部调用时的格式负担
        mapping = {
            "head_on": self.head_on,
            "head-on": self.head_on,
            "crossing": self.crossing,
            "overtaking": self.overtaking,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported scenario_type: {scenario_type}")
        return mapping[key]()
