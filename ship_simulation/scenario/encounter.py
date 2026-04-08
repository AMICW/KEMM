"""船舶会遇场景的数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from ship_simulation.core.environment import (
    GaussianScalarField,
    GridScalarField,
    GridVectorField,
    UniformVectorField,
    VortexVectorField,
)
from ship_simulation.core.ship_model import ShipState


@dataclass(frozen=True)
class VesselAgent:
    """场景中的船舶代理。"""

    name: str
    initial_state: ShipState
    goal: np.ndarray | None = None
    color: str = "tab:blue"
    role: str = "traffic"


@dataclass(frozen=True)
class CircularObstacle:
    """圆形静态障碍物。"""

    name: str
    center: np.ndarray
    radius: float
    kind: str = "island"
    color: str = "#7c3aed"


@dataclass(frozen=True)
class PolygonObstacle:
    """多边形静态障碍物。"""

    name: str
    vertices: np.ndarray
    kind: str = "keep_out"
    color: str = "#475569"


@dataclass(frozen=True)
class KeepOutZone(PolygonObstacle):
    """显式禁航区类型。"""

    kind: str = "keep_out"
    color: str = "#64748b"


@dataclass(frozen=True)
class ChannelBoundary(PolygonObstacle):
    """显式水道边界类型。"""

    kind: str = "channel_boundary"
    color: str = "#d1d5db"


@dataclass(frozen=True)
class ScenarioMetadata:
    """场景元信息。"""

    category: str = "nearshore_encounter"
    family: str = "default"
    difficulty: str = "baseline"
    layout_seed: int | None = None
    description: str = ""
    recommended_view: str = "overview"
    colreg_roles: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EncounterScenario:
    """完整的会遇场景容器。"""

    name: str
    own_ship: VesselAgent
    target_ships: list[VesselAgent] = field(default_factory=list)
    static_obstacles: list[CircularObstacle | PolygonObstacle | KeepOutZone | ChannelBoundary] = field(default_factory=list)
    scalar_fields: list[GaussianScalarField | GridScalarField] = field(default_factory=list)
    vector_fields: list[UniformVectorField | VortexVectorField | GridVectorField] = field(default_factory=list)
    metadata: ScenarioMetadata = field(default_factory=ScenarioMetadata)
    area: tuple[float, float, float, float] = (-1000.0, 9000.0, -4000.0, 4000.0)

    def with_updated_states(
        self,
        own_state: ShipState,
        target_states: Sequence[ShipState],
        *,
        name_suffix: str = "",
    ) -> "EncounterScenario":
        """基于当前状态构造局部重规划场景。"""

        updated_targets = [
            replace(target, initial_state=state)
            for target, state in zip(self.target_ships, target_states)
        ]
        updated_own = replace(self.own_ship, initial_state=own_state)
        scenario_name = self.name if not name_suffix else f"{self.name} {name_suffix}"
        return EncounterScenario(
            name=scenario_name,
            own_ship=updated_own,
            target_ships=updated_targets,
            static_obstacles=list(self.static_obstacles),
            scalar_fields=list(self.scalar_fields),
            vector_fields=list(self.vector_fields),
            metadata=self.metadata,
            area=self.area,
        )

    def traffic_agents(self) -> Iterable[VesselAgent]:
        """返回场景中的所有动态交通体。"""

        return tuple(self.target_ships)
