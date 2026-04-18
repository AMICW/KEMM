"""环境场建模。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import Tuple

import numpy as np

from ship_simulation.config import EnvironmentConfig


@dataclass(frozen=True)
class UniformVectorField:
    """均匀附加矢量场。"""

    name: str
    speed: float
    direction_deg: float
    weight: float = 1.0


@dataclass(frozen=True)
class VortexVectorField:
    """简单涡旋矢量场。"""

    name: str
    center: np.ndarray
    strength: float
    radius: float
    clockwise: bool = True


@dataclass(frozen=True)
class GaussianScalarField:
    """高斯标量风险场。"""

    name: str
    center: np.ndarray
    sigma_x: float
    sigma_y: float
    amplitude: float


@dataclass(frozen=True)
class GridScalarField:
    """规则网格上的标量场。"""

    name: str
    x_coords: np.ndarray
    y_coords: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class GridVectorField:
    """规则网格上的矢量场。"""

    name: str
    x_coords: np.ndarray
    y_coords: np.ndarray
    u_values: np.ndarray
    v_values: np.ndarray
    weight: float = 1.0


@dataclass
class EnvironmentField:
    """组合环境场。"""

    config: EnvironmentConfig
    scalar_layers: list[GaussianScalarField | GridScalarField] = field(default_factory=list)
    vector_layers: list[UniformVectorField | VortexVectorField | GridVectorField] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._current_vector = self._polar_to_xy(self.config.current_speed, self.config.current_direction_deg)
        self._wind_vector = self._polar_to_xy(self.config.wind_speed, self.config.wind_direction_deg)
        self._wind_drift_vector = self.config.wind_drift_coefficient * self._wind_vector

    @staticmethod
    def _polar_to_xy(speed: float, direction_deg: float) -> np.ndarray:
        direction_rad = radians(direction_deg)
        return np.array([speed * cos(direction_rad), speed * sin(direction_rad)], dtype=float)

    def current_at(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        _ = position, time_s
        return self._current_vector.copy()

    def wind_at(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        _ = position, time_s
        return self._wind_vector.copy()

    @staticmethod
    def _coerce_series_inputs(
        positions: np.ndarray | Tuple[float, float],
        times: np.ndarray | Tuple[float, ...] | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(positions, dtype=float)
        if pos.ndim == 1:
            if pos.size != 2:
                raise ValueError("Expected 2-D position input.")
            pos = pos.reshape(1, 2)
        elif pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("Expected positions with shape (n, 2).")

        t = np.asarray(times, dtype=float)
        if t.ndim == 0:
            t = np.full(len(pos), float(t), dtype=float)
        elif t.ndim == 1 and len(t) == 1 and len(pos) != 1:
            t = np.full(len(pos), float(t[0]), dtype=float)
        elif t.ndim != 1 or len(t) != len(pos):
            raise ValueError("Times must be scalar or have the same length as positions.")
        return pos, t

    @staticmethod
    def _grid_indices(coords: np.ndarray, value: float) -> tuple[int, int, float]:
        if len(coords) < 2:
            return 0, 0, 0.0
        if value <= coords[0]:
            return 0, 1, 0.0
        if value >= coords[-1]:
            return len(coords) - 2, len(coords) - 1, 1.0
        upper = int(np.searchsorted(coords, value, side="right"))
        lower = upper - 1
        span = float(coords[upper] - coords[lower])
        weight = 0.0 if span <= 1e-12 else float((value - coords[lower]) / span)
        return lower, upper, weight

    @staticmethod
    def _grid_indices_series(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(coords) < 2:
            zeros = np.zeros(len(values), dtype=int)
            return zeros, zeros, np.zeros(len(values), dtype=float)

        upper = np.searchsorted(coords, values, side="right")
        upper = np.clip(upper, 1, len(coords) - 1)
        lower = upper - 1
        lower_values = coords[lower]
        upper_values = coords[upper]
        span = upper_values - lower_values
        safe_span = np.where(np.abs(span) <= 1e-12, 1.0, span)
        weight = (values - lower_values) / safe_span
        weight = np.where(values <= coords[0], 0.0, weight)
        weight = np.where(values >= coords[-1], 1.0, weight)
        weight = np.where(np.abs(span) <= 1e-12, 0.0, weight)
        return lower.astype(int), upper.astype(int), weight.astype(float)

    @classmethod
    def _sample_grid(cls, x_coords: np.ndarray, y_coords: np.ndarray, grid: np.ndarray, position: np.ndarray) -> float:
        x0, x1, tx = cls._grid_indices(np.asarray(x_coords, dtype=float), float(position[0]))
        y0, y1, ty = cls._grid_indices(np.asarray(y_coords, dtype=float), float(position[1]))
        f00 = float(grid[y0, x0])
        f10 = float(grid[y0, x1])
        f01 = float(grid[y1, x0])
        f11 = float(grid[y1, x1])
        return float(
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        )

    @classmethod
    def _sample_grid_series(cls, x_coords: np.ndarray, y_coords: np.ndarray, grid: np.ndarray, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        if len(pos) == 0:
            return np.zeros(0, dtype=float)
        x0, x1, tx = cls._grid_indices_series(np.asarray(x_coords, dtype=float), pos[:, 0])
        y0, y1, ty = cls._grid_indices_series(np.asarray(y_coords, dtype=float), pos[:, 1])
        grid = np.asarray(grid, dtype=float)
        f00 = grid[y0, x0]
        f10 = grid[y0, x1]
        f01 = grid[y1, x0]
        f11 = grid[y1, x1]
        return (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        ).astype(float)

    def _layer_vector(self, layer: UniformVectorField | VortexVectorField | GridVectorField, position: np.ndarray) -> np.ndarray:
        if isinstance(layer, UniformVectorField):
            return layer.weight * self._polar_to_xy(layer.speed, layer.direction_deg)
        if isinstance(layer, GridVectorField):
            u = self._sample_grid(layer.x_coords, layer.y_coords, np.asarray(layer.u_values, dtype=float), position)
            v = self._sample_grid(layer.x_coords, layer.y_coords, np.asarray(layer.v_values, dtype=float), position)
            return layer.weight * np.array([u, v], dtype=float)

        offset = np.asarray(position, dtype=float) - np.asarray(layer.center, dtype=float)
        radius = max(np.linalg.norm(offset), 1e-6)
        if radius > layer.radius:
            decay = np.exp(-0.5 * ((radius - layer.radius) / max(layer.radius * 0.4, 1.0)) ** 2)
        else:
            decay = 1.0
        tangent = np.array([-offset[1], offset[0]], dtype=float) / radius
        if not layer.clockwise:
            tangent = -tangent
        magnitude = layer.strength * decay / max(radius / layer.radius, 0.3)
        return magnitude * tangent

    def _layer_vector_series(
        self,
        layer: UniformVectorField | VortexVectorField | GridVectorField,
        positions: np.ndarray,
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        if len(pos) == 0:
            return np.zeros((0, 2), dtype=float)
        if isinstance(layer, UniformVectorField):
            base = layer.weight * self._polar_to_xy(layer.speed, layer.direction_deg)
            return np.repeat(base.reshape(1, 2), len(pos), axis=0)
        if isinstance(layer, GridVectorField):
            u = self._sample_grid_series(layer.x_coords, layer.y_coords, np.asarray(layer.u_values, dtype=float), pos)
            v = self._sample_grid_series(layer.x_coords, layer.y_coords, np.asarray(layer.v_values, dtype=float), pos)
            return layer.weight * np.column_stack([u, v])

        offset = pos - np.asarray(layer.center, dtype=float)
        radius = np.linalg.norm(offset, axis=1)
        safe_radius = np.maximum(radius, 1e-6)
        decay = np.ones(len(pos), dtype=float)
        outside = radius > layer.radius
        decay[outside] = np.exp(-0.5 * ((radius[outside] - layer.radius) / max(layer.radius * 0.4, 1.0)) ** 2)
        tangent = np.column_stack([-offset[:, 1], offset[:, 0]]) / safe_radius[:, None]
        if not layer.clockwise:
            tangent = -tangent
        magnitude = layer.strength * decay / np.maximum(safe_radius / max(layer.radius, 1e-6), 0.3)
        return magnitude[:, None] * tangent

    def _layer_scalar(self, layer: GaussianScalarField | GridScalarField, position: np.ndarray) -> float:
        if isinstance(layer, GridScalarField):
            return self._sample_grid(layer.x_coords, layer.y_coords, np.asarray(layer.values, dtype=float), position)
        offset = np.asarray(position, dtype=float) - np.asarray(layer.center, dtype=float)
        exponent = -0.5 * (
            (offset[0] / max(layer.sigma_x, 1e-6)) ** 2
            + (offset[1] / max(layer.sigma_y, 1e-6)) ** 2
        )
        return float(layer.amplitude * np.exp(exponent))

    def _layer_scalar_series(self, layer: GaussianScalarField | GridScalarField, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        if len(pos) == 0:
            return np.zeros(0, dtype=float)
        if isinstance(layer, GridScalarField):
            return self._sample_grid_series(layer.x_coords, layer.y_coords, np.asarray(layer.values, dtype=float), pos)
        offset = pos - np.asarray(layer.center, dtype=float)
        exponent = -0.5 * (
            (offset[:, 0] / max(layer.sigma_x, 1e-6)) ** 2
            + (offset[:, 1] / max(layer.sigma_y, 1e-6)) ** 2
        )
        return (layer.amplitude * np.exp(exponent)).astype(float)

    def scalar_risk_series(
        self,
        positions: np.ndarray | Tuple[float, float],
        times: np.ndarray | Tuple[float, ...] | float,
    ) -> np.ndarray:
        pos, _ = self._coerce_series_inputs(positions, times)
        if not self.scalar_layers:
            return np.zeros(len(pos), dtype=float)
        return np.sum([self._layer_scalar_series(layer, pos) for layer in self.scalar_layers], axis=0)

    def vector_field_series(
        self,
        positions: np.ndarray | Tuple[float, float],
        times: np.ndarray | Tuple[float, ...] | float,
    ) -> np.ndarray:
        pos, _ = self._coerce_series_inputs(positions, times)
        if not self.vector_layers:
            return np.zeros((len(pos), 2), dtype=float)
        return np.sum([self._layer_vector_series(layer, pos) for layer in self.vector_layers], axis=0)

    def drift_series(
        self,
        positions: np.ndarray | Tuple[float, float],
        times: np.ndarray | Tuple[float, ...] | float,
    ) -> np.ndarray:
        pos, t = self._coerce_series_inputs(positions, times)
        vector_component = self.vector_field_series(pos, t)
        base = self._current_vector + self._wind_drift_vector
        return vector_component + base.reshape(1, 2)

    def vector_field_at(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        _ = time_s
        pos = np.asarray(position, dtype=float)
        if not self.vector_layers:
            return np.zeros(2, dtype=float)
        return np.sum([self._layer_vector(layer, pos) for layer in self.vector_layers], axis=0)

    def scalar_risk_at(self, position: np.ndarray | Tuple[float, float], time_s: float) -> float:
        _ = time_s
        pos = np.asarray(position, dtype=float)
        if not self.scalar_layers:
            return 0.0
        return float(np.sum([self._layer_scalar(layer, pos) for layer in self.scalar_layers]))

    def drift_velocity(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        _ = time_s
        vector_component = self.vector_field_at(position, time_s)
        return self._current_vector + self._wind_drift_vector + vector_component

    def sample_scalar_grid(
        self,
        area: tuple[float, float, float, float],
        *,
        resolution: int = 60,
        time_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xmin, xmax, ymin, ymax = area
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(xs, ys)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        zz = self.scalar_risk_series(positions, float(time_s)).reshape(xx.shape)
        return xx, yy, zz

    def sample_vector_grid(
        self,
        area: tuple[float, float, float, float],
        *,
        resolution: int = 18,
        time_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xmin, xmax, ymin, ymax = area
        xs = np.linspace(xmin, xmax, resolution)
        ys = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(xs, ys)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        vectors = self.drift_series(positions, float(time_s)).reshape(xx.shape[0], xx.shape[1], 2)
        uu = vectors[:, :, 0]
        vv = vectors[:, :, 1]
        return xx, yy, uu, vv

    def describe_layers(self) -> dict[str, list[str]]:
        return {
            "scalar_layers": [layer.name for layer in self.scalar_layers],
            "vector_layers": [layer.name for layer in self.vector_layers],
        }


__all__ = [
    "EnvironmentField",
    "GaussianScalarField",
    "GridScalarField",
    "GridVectorField",
    "UniformVectorField",
    "VortexVectorField",
]
