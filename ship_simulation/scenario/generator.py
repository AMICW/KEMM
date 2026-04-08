"""经典与近海增强会遇场景生成器。"""

from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np

from ship_simulation.config import HarborClutterTuningConfig, ProblemConfig, ScenarioTuningConfig
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

    @staticmethod
    def _stable_seed(key: str) -> int:
        return sum((index + 1) * ord(ch) for index, ch in enumerate(key)) % 1_000_003

    def _scenario_rng(self, scenario_key: str, tuning: ScenarioTuningConfig):
        if tuning.scenario_seed is None:
            return None
        return np.random.default_rng(int(tuning.scenario_seed) + self._stable_seed(scenario_key))

    @staticmethod
    def _jitter_xy(point: np.ndarray, rng, magnitude: float) -> np.ndarray:
        if rng is None or magnitude <= 0.0:
            return np.asarray(point, dtype=float)
        return np.asarray(point, dtype=float) + rng.normal(0.0, float(magnitude), size=2)

    @staticmethod
    def _offset_xy(point: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
        return np.asarray(point, dtype=float) + np.asarray(offset, dtype=float)

    @staticmethod
    def _scale_polygon_vertices(vertices: np.ndarray, scale: float, *, anchor: np.ndarray | None = None) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=float)
        center = np.asarray(anchor if anchor is not None else np.mean(vertices, axis=0), dtype=float)
        return center + scale * (vertices - center)

    @staticmethod
    def _limit_items(items: list, limit: int | None) -> list:
        if limit is None:
            return list(items)
        return list(items[: max(int(limit), 0)])

    def _tuned_agent(
        self,
        agent: VesselAgent,
        tuning: ScenarioTuningConfig,
        *,
        apply_goal_offset: bool,
        target: bool = False,
        rng=None,
    ) -> VesselAgent:
        difficulty = max(float(tuning.difficulty_scale), 0.25)
        speed_scale = (tuning.target_speed_scale * difficulty) if target else tuning.own_speed_scale
        state_offset = tuning.goal_offset if target else tuning.start_offset
        jittered_position = self._jitter_xy(
            np.array([agent.initial_state.x + state_offset[0], agent.initial_state.y + state_offset[1]], dtype=float),
            rng,
            tuning.geometry_jitter_m,
        )
        heading = float(agent.initial_state.heading)
        if target and rng is not None and tuning.traffic_heading_jitter_deg > 0.0:
            heading += float(rng.normal(0.0, np.deg2rad(tuning.traffic_heading_jitter_deg)))
        state = replace(
            agent.initial_state,
            x=float(jittered_position[0]),
            y=float(jittered_position[1]),
            heading=heading,
            speed=float(agent.initial_state.speed * speed_scale),
        )
        goal = agent.goal
        if apply_goal_offset and goal is not None:
            goal = self._offset_xy(goal, tuning.goal_offset)
            goal = self._jitter_xy(goal, rng, tuning.geometry_jitter_m * 0.6)
        return replace(agent, initial_state=state, goal=None if goal is None else np.asarray(goal, dtype=float))

    def _tuned_circular_obstacle(self, obstacle: CircularObstacle, tuning: ScenarioTuningConfig, *, rng=None) -> CircularObstacle:
        difficulty = max(float(tuning.difficulty_scale), 0.25)
        return replace(
            obstacle,
            center=self._jitter_xy(self._offset_xy(obstacle.center, tuning.goal_offset), rng, tuning.geometry_jitter_m),
            radius=float(obstacle.radius * tuning.circular_radius_scale * difficulty),
        )

    def _tuned_polygon_obstacle(
        self,
        obstacle,
        tuning: ScenarioTuningConfig,
        *,
        scale: float | None = None,
        y_scale: float = 1.0,
        rng=None,
    ):
        vertices = np.asarray(obstacle.vertices, dtype=float)
        shifted = vertices.copy()
        shifted[:, 0] += tuning.goal_offset[0]
        shifted[:, 1] += tuning.goal_offset[1]
        if rng is not None and tuning.geometry_jitter_m > 0.0:
            shifted += rng.normal(0.0, tuning.geometry_jitter_m * 0.45, size=shifted.shape)
        centroid = np.mean(shifted, axis=0)
        shifted[:, 1] = centroid[1] + y_scale * (shifted[:, 1] - centroid[1])
        tuned_scale = float(scale if scale is not None else tuning.polygon_scale) * max(float(tuning.difficulty_scale), 0.25)
        return replace(obstacle, vertices=self._scale_polygon_vertices(shifted, tuned_scale))

    def _scale_scalar_fields(self, fields: list, amplitude_scale: float, *, rng=None, tuning: ScenarioTuningConfig | None = None) -> list:
        tuned = []
        tuning = tuning or ScenarioTuningConfig()
        effective_scale = amplitude_scale * max(float(tuning.difficulty_scale), 0.25)
        for field in fields:
            if isinstance(field, GaussianScalarField):
                center = np.asarray(field.center, dtype=float)
                center = self._jitter_xy(center, rng, tuning.geometry_jitter_m * 0.5)
                tuned.append(replace(field, center=center, amplitude=float(field.amplitude * effective_scale)))
            elif isinstance(field, GridScalarField):
                tuned.append(replace(field, values=np.asarray(field.values, dtype=float) * effective_scale))
            else:
                tuned.append(field)
        return tuned

    def _scale_vector_fields(self, fields: list, speed_scale: float, *, rng=None, tuning: ScenarioTuningConfig | None = None) -> list:
        tuned = []
        tuning = tuning or ScenarioTuningConfig()
        effective_scale = speed_scale * max(float(tuning.difficulty_scale), 0.25)
        direction_delta = 0.0
        if rng is not None and tuning.current_direction_jitter_deg > 0.0:
            direction_delta = float(rng.normal(0.0, tuning.current_direction_jitter_deg))
        for field in fields:
            if isinstance(field, UniformVectorField):
                tuned.append(
                    replace(
                        field,
                        speed=float(field.speed * effective_scale),
                        direction_deg=float(field.direction_deg + direction_delta),
                    )
                )
            elif isinstance(field, GridVectorField):
                tuned.append(
                    replace(
                        field,
                        u_values=np.asarray(field.u_values, dtype=float) * effective_scale,
                        v_values=np.asarray(field.v_values, dtype=float) * effective_scale,
                    )
                )
            elif isinstance(field, VortexVectorField):
                center = self._jitter_xy(np.asarray(field.center, dtype=float), rng, tuning.geometry_jitter_m * 0.4)
                tuned.append(replace(field, center=center, strength=float(field.strength * effective_scale)))
            else:
                tuned.append(field)
        return tuned

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
        tuning = self.config.scenario_generation.head_on
        rng = self._scenario_rng("head_on", tuning)
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
            family=tuning.family_name if tuning.family_name != "default" else "head_on",
            difficulty=self.config.experiment.difficulty_label,
            layout_seed=None if tuning.scenario_seed is None else int(tuning.scenario_seed) + self._stable_seed("head_on"),
            description="对遇场景叠加狭水道边界与轻微潮流，强调迎面避碰与航道约束。",
            recommended_view="corridor",
            colreg_roles={"Target A": "head_on"},
            tags=("head_on", "channel", "nearshore"),
        )
        own = self._tuned_agent(own, tuning, apply_goal_offset=True, rng=rng)
        targets = [self._tuned_agent(target, tuning, apply_goal_offset=False, target=True, rng=rng) for target in self._limit_items(targets, tuning.target_limit)]
        obstacles = [self._tuned_polygon_obstacle(obstacle, tuning, rng=rng) for obstacle in obstacles]
        scalar_fields = self._scale_scalar_fields(scalar_fields, tuning.scalar_amplitude_scale, rng=rng, tuning=tuning)
        vector_fields = self._scale_vector_fields(vector_fields, tuning.vector_speed_scale, rng=rng, tuning=tuning)
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
        tuning = self.config.scenario_generation.crossing
        rng = self._scenario_rng("crossing", tuning)
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
            family=tuning.family_name if tuning.family_name != "default" else "crossing",
            difficulty=self.config.experiment.difficulty_label,
            layout_seed=None if tuning.scenario_seed is None else int(tuning.scenario_seed) + self._stable_seed("crossing"),
            description="交叉会遇叠加小岛、禁航区和局部风险场，是最有代表性的多目标折中场景。",
            recommended_view="overview",
            colreg_roles={"Target A": "crossing_give_way", "Target B": "crossing_stand_on"},
            tags=("crossing", "island", "risk_field", "multi_ship"),
        )
        own = self._tuned_agent(own, tuning, apply_goal_offset=True, rng=rng)
        targets = [self._tuned_agent(target, tuning, apply_goal_offset=False, target=True, rng=rng) for target in self._limit_items(targets, tuning.target_limit)]
        tuned_obstacles = []
        for obstacle in obstacles:
            if isinstance(obstacle, CircularObstacle):
                tuned_obstacles.append(self._tuned_circular_obstacle(obstacle, tuning, rng=rng))
            else:
                tuned_obstacles.append(self._tuned_polygon_obstacle(obstacle, tuning, rng=rng))
        obstacles = tuned_obstacles
        scalar_fields = self._scale_scalar_fields(scalar_fields, tuning.scalar_amplitude_scale, rng=rng, tuning=tuning)
        vector_fields = self._scale_vector_fields(vector_fields, tuning.vector_speed_scale, rng=rng, tuning=tuning)
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
        tuning = self.config.scenario_generation.overtaking
        rng = self._scenario_rng("overtaking", tuning)
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
            family=tuning.family_name if tuning.family_name != "default" else "overtaking",
            difficulty=self.config.experiment.difficulty_label,
            layout_seed=None if tuning.scenario_seed is None else int(tuning.scenario_seed) + self._stable_seed("overtaking"),
            description="追越场景叠加狭水道边界和尾流风险区，用于展示效率与安全间的再平衡。",
            recommended_view="corridor",
            colreg_roles={"Target A": "overtaking", "Target B": "crossing_give_way"},
            tags=("overtaking", "channel", "multi_ship"),
        )
        own = self._tuned_agent(own, tuning, apply_goal_offset=True, rng=rng)
        targets = [self._tuned_agent(target, tuning, apply_goal_offset=False, target=True, rng=rng) for target in self._limit_items(targets, tuning.target_limit)]
        obstacles = [self._tuned_polygon_obstacle(obstacle, tuning, rng=rng) for obstacle in obstacles]
        scalar_fields = self._scale_scalar_fields(scalar_fields, tuning.scalar_amplitude_scale, rng=rng, tuning=tuning)
        vector_fields = self._scale_vector_fields(vector_fields, tuning.vector_speed_scale, rng=rng, tuning=tuning)
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
        tuning: HarborClutterTuningConfig = self.config.scenario_generation.harbor_clutter
        rng = self._scenario_rng("harbor_clutter", tuning)
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
            family=tuning.family_name,
            difficulty=self.config.experiment.difficulty_label,
            layout_seed=None if tuning.scenario_seed is None else int(tuning.scenario_seed) + self._stable_seed("harbor_clutter"),
            description="高密障碍受限海域穿越场景，包含窄通道、码头禁入区、岛礁样障碍和多目标船会遇。",
            recommended_view="dense_harbor",
            colreg_roles={
                "Target A": "crossing_give_way",
                "Target B": "crossing_stand_on",
                "Target C": "head_on",
            },
            tags=("harbor", "dense_obstacles", "restricted_waters", "multi_ship", "risk_field"),
        )
        own = self._tuned_agent(own, tuning, apply_goal_offset=True, rng=rng)
        targets = [self._tuned_agent(target, tuning, apply_goal_offset=False, target=True, rng=rng) for target in self._limit_items(targets, tuning.target_limit)]
        boundary_obstacles = [
            self._tuned_polygon_obstacle(obstacle, tuning, scale=1.0, y_scale=tuning.channel_width_scale, rng=rng)
            for obstacle in obstacles
            if isinstance(obstacle, ChannelBoundary)
        ]
        circular_obstacles = [
            self._tuned_circular_obstacle(obstacle, tuning, rng=rng)
            for obstacle in obstacles
            if isinstance(obstacle, CircularObstacle)
        ]
        polygon_obstacles = [
            self._tuned_polygon_obstacle(obstacle, tuning, rng=rng)
            for obstacle in obstacles
            if isinstance(obstacle, KeepOutZone)
        ]
        obstacles = (
            boundary_obstacles
            + self._limit_items(circular_obstacles, tuning.circular_obstacle_limit)
            + self._limit_items(polygon_obstacles, tuning.polygon_obstacle_limit)
        )
        scalar_fields = self._scale_scalar_fields(scalar_fields, tuning.scalar_amplitude_scale, rng=rng, tuning=tuning)
        vector_fields = self._scale_vector_fields(vector_fields, tuning.vector_speed_scale, rng=rng, tuning=tuning)
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
