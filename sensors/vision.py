from __future__ import annotations

import math
from typing import Any

import numpy as np

from rendering.kinematics import project_body

from .sensor_base import Sensor


class VisionSensor(Sensor):
    """Deterministic headless RGB-like camera for the configured debug body.

    This is a compact sensor frame, not a photorealistic camera or a learned
    perception model. It rasterizes the simulator's 2D body projection so a
    brain can receive visual pixels without accessing simulator internals.
    """

    def __init__(self, name: str, config: dict | None = None, body: Any = None):
        super().__init__(name, config)
        resolution = self.config.get("resolution", [64, 64])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resolution):
            raise ValueError("vision resolution must be a pair of positive integers")
        fov = self.config.get("fov", 90.0)
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not math.isfinite(float(fov)) or not 0 < float(fov) <= 180:
            raise ValueError("vision fov must be a finite number in (0, 180]")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        view_scale = self.config.get("view_scale", 1.0)
        if isinstance(view_scale, bool) or not isinstance(view_scale, (int, float)) or not math.isfinite(float(view_scale)) or float(view_scale) <= 0:
            raise ValueError("vision view_scale must be a finite positive number")
        self.fov = float(fov)
        self.view_scale = float(view_scale)
        self.target = self.camera_target()
        self.body = body
        self.background = (18, 25, 35)
        self.link_color = (37, 99, 235)
        self.joint_color = (34, 211, 238)

    def observe(self, physics_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        frame = np.full((self.resolution[1], self.resolution[0], 3), self.background, dtype=np.uint8)
        if self.body is not None:
            points = project_body(self.body, physics_state)
            floor_width = self._floor_width(physics_state)
            x_half = max(floor_width / 2.0 if floor_width is not None else 1.0, 1.0) * self.view_scale
            view_half_height = max(1.0, x_half * self.resolution[1] / self.resolution[0])
            target_x, target_y = self.target if self.target is not None else (0.0, -0.15 + view_half_height)
            y_min = target_y - view_half_height
            y_max = target_y + view_half_height
            projected = {name: self._pixel(point, x_half, y_min, y_max, target_x) for name, point in points.items()}
            for joint in self.body.joints.values():
                if joint.parent in projected and joint.child in projected:
                    self._draw_line(frame, projected[joint.parent], projected[joint.child], self.link_color)
            for pixel in projected.values():
                self._draw_disk(frame, pixel, self.joint_color, radius=max(1, min(self.resolution) // 32))
            if floor_width is not None:
                left = self._pixel((-floor_width / 2.0, 0.0), x_half, y_min, y_max, target_x)[0]
                right = self._pixel((floor_width / 2.0, 0.0), x_half, y_min, y_max, target_x)[0]
                self._draw_line(frame, (left, self._pixel((0.0, 0.0), x_half, y_min, y_max, target_x)[1]), (right, self._pixel((0.0, 0.0), x_half, y_min, y_max, target_x)[1]), (148, 163, 184))
        non_background_pixels = int(np.count_nonzero(np.any(frame != self.background, axis=2)))
        return {
            "rgb": frame,
            "resolution": self.resolution,
            "fov": self.fov,
            "frame_id": self.frame_id(physics_state),
            "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": self.resolution, "fov": self.fov, "view_scale": self.view_scale, "target": self.target},
            "available": self.body is not None,
            "source": "headless_body_projection" if self.body is not None else "unconfigured_body_projection",
            "non_background_pixels": non_background_pixels,
        }

    def _pixel(self, point: tuple[float, float], x_half: float, y_min: float, y_max: float, target_x: float = 0.0) -> tuple[int, int]:
        width, height = self.resolution
        x, y = point
        pixel_x = round((x - target_x + x_half) / (2.0 * x_half) * (width - 1))
        pixel_y = round((y_max - y) / (y_max - y_min) * (height - 1))
        return max(0, min(width - 1, pixel_x)), max(0, min(height - 1, pixel_y))

    @staticmethod
    def _draw_line(frame: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for index in range(steps + 1):
            ratio = index / steps
            x = round(x0 + (x1 - x0) * ratio)
            y = round(y0 + (y1 - y0) * ratio)
            frame[y, x] = color

    @staticmethod
    def _draw_disk(frame: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
        x_center, y_center = center
        height, width = frame.shape[:2]
        for y in range(max(0, y_center - radius), min(height, y_center + radius + 1)):
            for x in range(max(0, x_center - radius), min(width, x_center + radius + 1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    frame[y, x] = color

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
