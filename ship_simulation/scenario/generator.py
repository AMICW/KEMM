"""经典与近海增强会遇场景生成器。"""

from __future__ import annotations

from math import pi

import numpy as np

from ship_simulation.config import ProblemConfig
from ship_simulation.core.environment import (
    GaussianScalarField,
    GridScalarField,
    GridVectorField,
    UniformVectorField,
    VortexVectorField,
)
from ship_simulation.core.ship_model import ShipState
from ship_simulation.scenario.encounter import (
    ChannelBoundary,
    CircularObstacle,
    EncounterScenario,
    KeepOutZone,
    ScenarioMetadata,
    VesselAgent,
)


class ScenarioGenerator:
    """生成论文实验风格的近海会遇场景。"""

    def __init__(self, config: ProblemConfig):
        self.config = config

    def _grid_axes(self, nx: int = 28, ny: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xmin, xmax, ymin, ymax = self.config.simulation.area
        xs = np.linspace(xmin, xmax, nx, dtype=float)
        ys = np.linspace(ymin, ymax, ny, dtype=float)
        xx, yy = np.meshgrid(xs, ys)
        return xs, ys, xx, yy

    def _grid_scalar(
        self,
        name: str,
        *,
        center: np.ndarray,
        sigma_x: float,
        sigma_y: float,
        amplitude: float,
        nx: int = 28,
        ny: int = 24,
    ) -> GridScalarField:
        xs, ys, xx, yy = self._grid_axes(nx=nx, ny=ny)
        exponent = -0.5 * (
            ((xx - float(center[0])) / max(sigma_x, 1e-6)) ** 2
            + ((yy - float(center[1])) / max(sigma_y, 1e-6)) ** 2
        )
        values = amplitude * np.exp(exponent)
        return GridScalarField(name=name, x_coords=xs, y_coords=ys, values=values.astype(float))

    def _grid_vector(
        self,
        name: str,
        *,
        center: np.ndarray,
        sigma_x: float,
        sigma_y: float,
        base_speed: float,
        direction_deg: float,
        turn_scale_deg: float = 24.0,
        nx: int = 28,
        ny: int = 24,
    ) -> GridVectorField:
        xs, ys, xx, yy = self._grid_axes(nx=nx, ny=ny)
        envelope = np.exp(
            -0.5
            * (
                ((xx - float(center[0])) / max(sigma_x, 1e-6)) ** 2
                + ((yy - float(center[1])) / max(sigma_y, 1e-6)) ** 2
            )
        )
        direction = np.deg2rad(direction_deg + turn_scale_deg * np.tanh((yy - float(center[1])) / max(sigma_y, 1.0)))
        magnitude = base_speed * (0.45 + 0.55 * envelope)
        u_values = magnitude * np.cos(direction)
        v_values = magnitude * np.sin(direction)
        return GridVectorField(
            name=name,
            x_coords=xs,
            y_coords=ys,
            u_values=u_values.astype(float),
            v_values=v_values.astype(float),
        )

    def head_on(self) -> EncounterScenario:
        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=0.0, y=-180.0, heading=0.0, speed=6.4),
            goal=np.array([7200.0, -180.0], dtype=float),
            color="tab:blue",
            role="own_ship",
        )
        targets = [
            VesselAgent(
                name="Target A",
                initial_state=ShipState(x=7100.0, y=160.0, heading=pi, speed=5.8),
                color="tab:red",
                role="head_on",
            )
        ]
        obstacles = [
            ChannelBoundary(
                name="North Bank",
                vertices=np.array([[1500.0, 1100.0], [7600.0, 1100.0], [7600.0, 1800.0], [1500.0, 1800.0]], dtype=float),
            ),
            ChannelBoundary(
                name="South Bank",
                vertices=np.array([[1500.0, -1800.0], [7600.0, -1800.0], [7600.0, -1100.0], [1500.0, -1100.0]], dtype=float),
            ),
        ]
        scalar_fields = [
            GaussianScalarField("Traffic Separation", np.array([3600.0, 0.0]), 1200.0, 520.0, 0.45),
        ]
        vector_fields = [
            UniformVectorField("Tidal Set", speed=0.14, direction_deg=18.0, weight=1.0),
        ]
        metadata = ScenarioMetadata(
            category="nearshore_head_on",
            description="对遇场景叠加狭水道边界与轻微潮流，强调迎面避碰与航道约束。",
            recommended_view="corridor",
            colreg_roles={"Target A": "head_on"},
            tags=("head_on", "channel", "nearshore"),
        )
        return EncounterScenario(
            name="Head-on",
            own_ship=own,
            target_ships=targets,
            static_obstacles=obstacles,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
            area=self.config.simulation.area,
        )

    def crossing(self) -> EncounterScenario:
        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=0.0, y=-480.0, heading=0.03, speed=6.2),
            goal=np.array([7300.0, 320.0], dtype=float),
            color="tab:blue",
            role="own_ship",
        )
        targets = [
            VesselAgent(
                name="Target A",
                initial_state=ShipState(x=3250.0, y=-3200.0, heading=pi / 2.0, speed=5.7),
                color="tab:orange",
                role="crossing_give_way",
            ),
            VesselAgent(
                name="Target B",
                initial_state=ShipState(x=5400.0, y=1700.0, heading=-pi / 2.0, speed=4.8),
                color="tab:green",
                role="crossing_stand_on",
            ),
        ]
        obstacles = [
            CircularObstacle("Islet", center=np.array([3600.0, -150.0], dtype=float), radius=420.0, kind="island", color="#7c3aed"),
            KeepOutZone(
                name="Restricted Zone",
                vertices=np.array([[4700.0, 600.0], [6100.0, 900.0], [5900.0, 1800.0], [4500.0, 1500.0]], dtype=float),
            ),
        ]
        scalar_fields = [
            GaussianScalarField("Harbor Approach Risk", np.array([5000.0, 200.0]), 1100.0, 700.0, 0.65),
            self._grid_scalar(
                "Shallow Patch",
                center=np.array([2500.0, -900.0], dtype=float),
                sigma_x=850.0,
                sigma_y=650.0,
                amplitude=0.38,
            ),
        ]
        vector_fields = [
            UniformVectorField("Littoral Current", speed=0.18, direction_deg=35.0, weight=1.0),
            self._grid_vector(
                "Coastal Shear",
                center=np.array([4300.0, 0.0], dtype=float),
                sigma_x=1700.0,
                sigma_y=1200.0,
                base_speed=0.11,
                direction_deg=26.0,
            ),
            VortexVectorField("Breakwater Eddy", center=np.array([4200.0, 0.0]), strength=0.08, radius=1100.0, clockwise=True),
        ]
        metadata = ScenarioMetadata(
            category="nearshore_crossing",
            description="交叉会遇叠加小岛、禁航区和局部风险场，是最有代表性的多目标折中场景。",
            recommended_view="overview",
            colreg_roles={"Target A": "crossing_give_way", "Target B": "crossing_stand_on"},
            tags=("crossing", "island", "risk_field", "multi_ship"),
        )
        return EncounterScenario(
            name="Crossing",
            own_ship=own,
            target_ships=targets,
            static_obstacles=obstacles,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
            area=self.config.simulation.area,
        )

    def overtaking(self) -> EncounterScenario:
        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=0.0, y=-120.0, heading=0.0, speed=7.1),
            goal=np.array([7700.0, 180.0], dtype=float),
            color="tab:blue",
            role="own_ship",
        )
        targets = [
            VesselAgent(
                name="Target A",
                initial_state=ShipState(x=1950.0, y=70.0, heading=0.0, speed=4.6),
                color="tab:green",
                role="overtaking",
            ),
            VesselAgent(
                name="Target B",
                initial_state=ShipState(x=6200.0, y=-900.0, heading=pi / 16.0, speed=5.0),
                color="tab:red",
                role="crossing_give_way",
            ),
        ]
        obstacles = [
            ChannelBoundary(
                name="Channel North Edge",
                vertices=np.array([[1000.0, 900.0], [7800.0, 900.0], [7800.0, 1700.0], [1000.0, 1700.0]], dtype=float),
            ),
            ChannelBoundary(
                name="Channel South Edge",
                vertices=np.array([[1000.0, -1700.0], [7800.0, -1700.0], [7800.0, -900.0], [1000.0, -900.0]], dtype=float),
            ),
        ]
        scalar_fields = [
            GaussianScalarField("Wake Interaction Zone", np.array([2200.0, 80.0]), 700.0, 320.0, 0.55),
        ]
        vector_fields = [
            UniformVectorField("Harbor Outflow", speed=0.11, direction_deg=8.0, weight=1.0),
        ]
        metadata = ScenarioMetadata(
            category="nearshore_overtaking",
            description="追越场景叠加狭水道边界和尾流风险区，用于展示效率与安全间的再平衡。",
            recommended_view="corridor",
            colreg_roles={"Target A": "overtaking", "Target B": "crossing_give_way"},
            tags=("overtaking", "channel", "multi_ship"),
        )
        return EncounterScenario(
            name="Overtaking",
            own_ship=own,
            target_ships=targets,
            static_obstacles=obstacles,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
            area=self.config.simulation.area,
        )

    def harbor_clutter(self) -> EncounterScenario:
        own = VesselAgent(
            name="Own Ship",
            initial_state=ShipState(x=150.0, y=-1450.0, heading=0.08, speed=5.9),
            goal=np.array([7650.0, 1260.0], dtype=float),
            color="tab:blue",
            role="own_ship",
        )
        targets = [
            VesselAgent(
                name="Target A",
                initial_state=ShipState(x=2180.0, y=2150.0, heading=-pi / 2.2, speed=4.8),
                color="tab:red",
                role="crossing_give_way",
            ),
            VesselAgent(
                name="Target B",
                initial_state=ShipState(x=4820.0, y=-2180.0, heading=pi / 2.05, speed=4.4),
                color="tab:green",
                role="crossing_stand_on",
            ),
            VesselAgent(
                name="Target C",
                initial_state=ShipState(x=6260.0, y=1180.0, heading=pi, speed=4.2),
                color="tab:orange",
                role="head_on",
            ),
        ]
        obstacles = [
            ChannelBoundary(
                name="Harbor North Boundary",
                vertices=np.array([[350.0, 1850.0], [7900.0, 1850.0], [7900.0, 2700.0], [350.0, 2700.0]], dtype=float),
            ),
            ChannelBoundary(
                name="Harbor South Boundary",
                vertices=np.array([[350.0, -2700.0], [7900.0, -2700.0], [7900.0, -1850.0], [350.0, -1850.0]], dtype=float),
            ),
            CircularObstacle("Beacon Field A", center=np.array([1120.0, -1020.0], dtype=float), radius=230.0, kind="buoy_cluster", color="#7c3aed"),
            CircularObstacle("Beacon Field B", center=np.array([1960.0, 170.0], dtype=float), radius=250.0, kind="island", color="#8b5cf6"),
            CircularObstacle("Mooring Area", center=np.array([2780.0, -610.0], dtype=float), radius=310.0, kind="keep_out", color="#7c3aed"),
            CircularObstacle("Islet East", center=np.array([3800.0, 860.0], dtype=float), radius=280.0, kind="island", color="#7c3aed"),
            CircularObstacle("Buoy Cluster C", center=np.array([4550.0, -1120.0], dtype=float), radius=250.0, kind="buoy_cluster", color="#8b5cf6"),
            CircularObstacle("Shoal South", center=np.array([5650.0, -240.0], dtype=float), radius=300.0, kind="shoal", color="#7c3aed"),
            CircularObstacle("Breakwater Tip", center=np.array([6460.0, 820.0], dtype=float), radius=250.0, kind="breakwater_tip", color="#8b5cf6"),
            KeepOutZone(
                name="West Pier Zone",
                vertices=np.array([[820.0, 760.0], [1530.0, 1040.0], [1390.0, 1710.0], [700.0, 1430.0]], dtype=float),
            ),
            KeepOutZone(
                name="Central Terminal",
                vertices=np.array([[3150.0, 1180.0], [4310.0, 1430.0], [4140.0, 2230.0], [2970.0, 1990.0]], dtype=float),
            ),
            KeepOutZone(
                name="South Cargo Apron",
                vertices=np.array([[5050.0, -1710.0], [6560.0, -1570.0], [6480.0, -980.0], [4970.0, -1110.0]], dtype=float),
            ),
            KeepOutZone(
                name="East Dock Basin",
                vertices=np.array([[6760.0, -180.0], [7590.0, 120.0], [7420.0, 960.0], [6640.0, 720.0]], dtype=float),
            ),
        ]
        scalar_fields = [
            GaussianScalarField("Channel Exposure", np.array([3800.0, 80.0]), 2300.0, 980.0, 0.34),
            GaussianScalarField("Harbor Conflict Zone", np.array([4950.0, -120.0]), 1120.0, 820.0, 0.62),
            self._grid_scalar(
                "Sediment Shallow Water",
                center=np.array([2140.0, -1380.0], dtype=float),
                sigma_x=1180.0,
                sigma_y=640.0,
                amplitude=0.42,
                nx=34,
                ny=28,
            ),
            self._grid_scalar(
                "Terminal Wake Disturbance",
                center=np.array([6240.0, 1180.0], dtype=float),
                sigma_x=940.0,
                sigma_y=760.0,
                amplitude=0.37,
                nx=34,
                ny=28,
            ),
        ]
        vector_fields = [
            UniformVectorField("Harbor Set", speed=0.16, direction_deg=21.0, weight=1.0),
            self._grid_vector(
                "Berth Shear Flow",
                center=np.array([4080.0, 0.0], dtype=float),
                sigma_x=2100.0,
                sigma_y=1360.0,
                base_speed=0.13,
                direction_deg=18.0,
                turn_scale_deg=31.0,
                nx=34,
                ny=28,
            ),
            VortexVectorField("Breakwater Eddy", center=np.array([6060.0, 960.0]), strength=0.06, radius=980.0, clockwise=False),
        ]
        metadata = ScenarioMetadata(
            category="restricted_harbor_transit",
            description="高密障碍受限海域穿越场景，包含窄通道、码头禁入区、岛礁样障碍和多目标船会遇。",
            recommended_view="dense_harbor",
            colreg_roles={
                "Target A": "crossing_give_way",
                "Target B": "crossing_stand_on",
                "Target C": "head_on",
            },
            tags=("harbor", "dense_obstacles", "restricted_waters", "multi_ship", "risk_field"),
        )
        return EncounterScenario(
            name="Harbor Clutter",
            own_ship=own,
            target_ships=targets,
            static_obstacles=obstacles,
            scalar_fields=scalar_fields,
            vector_fields=vector_fields,
            metadata=metadata,
            area=self.config.simulation.area,
        )

    def generate(self, scenario_type: str) -> EncounterScenario:
        key = scenario_type.strip().lower()
        mapping = {
            "head_on": self.head_on,
            "head-on": self.head_on,
            "crossing": self.crossing,
            "overtaking": self.overtaking,
            "harbor_clutter": self.harbor_clutter,
            "harbor-clutter": self.harbor_clutter,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported scenario_type: {scenario_type}")
        return mapping[key]()
