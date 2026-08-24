from __future__ import annotations

import math
from typing import Any

import numpy as np

from rendering.matplotlib_3d_renderer import project_body_3d

from .sensor_base import Sensor


class DepthSensor(Sensor):
    """Deterministic headless depth camera for the configured debug body."""

    def __init__(self, name: str, config: dict | None = None, body: Any = None):
        super().__init__(name, config)
        resolution = self.config.get("resolution", [64, 64])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resolution):
            raise ValueError("depth resolution must be a pair of positive integers")
        near = self.config.get("near", 0.1)
        far = self.config.get("far", 10.0)
        if not self._finite_positive(near) or not self._finite_positive(far) or float(near) >= float(far):
            raise ValueError("depth near and far must be finite positive numbers with near below far")
        view_scale = self.config.get("view_scale", 1.0)
        if not self._finite_positive(view_scale):
            raise ValueError("depth view_scale must be a finite positive number")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.view_scale = float(view_scale)
        self.near = float(near)
        self.far = float(far)
        self.body = body

    def observe(self, physics_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        depth = np.full((self.resolution[1], self.resolution[0]), self.far, dtype=np.float32)
        if self.body is not None:
            points = project_body_3d(self.body, physics_state)
            floor_width = self._floor_width(physics_state)
            x_half = max(floor_width / 2.0 if floor_width is not None else 1.0, 1.0) * self.view_scale
            view_half_height = max(1.0, x_half * self.resolution[1] / self.resolution[0])
            z_min = -0.15
            z_max = z_min + 2.0 * view_half_height
            projected = {
                name: (self._pixel((point[0], point[2]), x_half, z_min, z_max), self._distance(point[1]))
                for name, point in points.items()
            }
            for joint in self.body.joints.values():
                if joint.parent in projected and joint.child in projected:
                    self._draw_depth_line(depth, projected[joint.parent], projected[joint.child])
            for pixel, distance in projected.values():
                self._set_depth(depth, pixel, distance, radius=max(1, min(self.resolution) // 32))
        return {
            "depth": depth,
            "resolution": self.resolution,
            "near": self.near,
            "far": self.far,
            "frame_id": self.frame_id(physics_state),
            "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": self.resolution, "near": self.near, "far": self.far, "view_scale": self.view_scale},
            "available": self.body is not None,
            "source": "headless_body_projection" if self.body is not None else "unconfigured_body_projection",
            "valid_pixels": int(np.count_nonzero(depth < self.far)),
        }

    def _distance(self, y: float) -> float:
        return float(np.clip(2.0 - y, self.near, self.far))

    def _pixel(self, point: tuple[float, float], x_half: float, z_min: float, z_max: float) -> tuple[int, int]:
        width, height = self.resolution
        x, z = point
        pixel_x = round((x + x_half) / (2.0 * x_half) * (width - 1))
        pixel_y = round((z_max - z) / (z_max - z_min) * (height - 1))
        return max(0, min(width - 1, pixel_x)), max(0, min(height - 1, pixel_y))

    def _draw_depth_line(self, depth: np.ndarray, start: tuple[tuple[int, int], float], end: tuple[tuple[int, int], float]) -> None:
        (x0, y0), distance0 = start
        (x1, y1), distance1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for index in range(steps + 1):
            ratio = index / steps
            pixel = round(x0 + (x1 - x0) * ratio), round(y0 + (y1 - y0) * ratio)
            distance = distance0 + (distance1 - distance0) * ratio
            self._set_depth(depth, pixel, distance)

    @staticmethod
    def _set_depth(depth: np.ndarray, pixel: tuple[int, int], distance: float, radius: int = 0) -> None:
        x_center, y_center = pixel
        height, width = depth.shape
        for y in range(max(0, y_center - radius), min(height, y_center + radius + 1)):
            for x in range(max(0, x_center - radius), min(width, x_center + radius + 1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    depth[y, x] = min(depth[y, x], distance)

    @staticmethod
    def _floor_width(physics_state: dict[str, Any]) -> float | None:
        world = physics_state.get("world")
        if not isinstance(world, dict) or world.get("floor_enabled") is not True:
            return None
        for object_definition in world.get("objects", []):
            if isinstance(object_definition, dict) and object_definition.get("type") == "floor":
                size = object_definition.get("size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    return float(size[0])
        return None

    @staticmethod
    def _finite_positive(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0
