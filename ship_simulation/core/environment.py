"""ship_simulation/core/environment.py

本文件负责环境场建模。

当前版本只实现恒定海流和恒定风：
- `current_at()` 返回海流速度
- `wind_at()` 返回风速向量
- `drift_velocity()` 返回对船舶地面航迹产生的合成漂移

后续如果要升级为：
- 网格化流场
- 随时间变化的风场
- 波浪扰动模型
可以优先扩展这个文件，而不需要改优化接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Tuple

import numpy as np

from ship_simulation.config import EnvironmentConfig


@dataclass
class EnvironmentField:
    """恒定环境场。

    之所以单独封装为类，而不是直接写成函数，是为了后续更容易替换成
    更复杂的环境对象，同时保持调用方式不变。
    """

    config: EnvironmentConfig

    @staticmethod
    def _polar_to_xy(speed: float, direction_deg: float) -> np.ndarray:
        """将极坐标形式的速度和方向转换为二维笛卡尔向量。"""

        direction_rad = radians(direction_deg)
        return np.array([speed * cos(direction_rad), speed * sin(direction_rad)], dtype=float)

    def current_at(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        """获取给定位置和时刻的海流速度。

        当前 MVP 中环境不随位置和时间变化，因此这里只是保留统一接口。
        """

        _ = position, time_s
        return self._polar_to_xy(self.config.current_speed, self.config.current_direction_deg)

    def wind_at(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        """获取给定位置和时刻的风速向量。"""

        _ = position, time_s
        return self._polar_to_xy(self.config.wind_speed, self.config.wind_direction_deg)

    def drift_velocity(self, position: np.ndarray | Tuple[float, float], time_s: float) -> np.ndarray:
        """返回环境导致的合成漂移速度。

        这里采用简化假设：
        地面速度 = 船体推进速度 + 海流速度 + 风致漂移
        """

        current = self.current_at(position, time_s)
        wind = self.wind_at(position, time_s)
        return current + self.config.wind_drift_coefficient * wind
