from __future__ import annotations

from typing import Any

import numpy as np

from rendering.kinematics import project_body

from .sensor_base import Sensor


class SegmentationSensor(Sensor):
    """Deterministic headless semantic labels for the configured body projection."""

    def __init__(self, name: str, config: dict | None = None, body: Any = None):
        super().__init__(name, config)
        resolution = self.config.get("resolution", [64, 64])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resolution):
            raise ValueError("segmentation resolution must be a pair of positive integers")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.body = body
        self.label_map = {"background": 0, "floor": 1}
        if body is not None:
            self.label_map.update({name: index for index, name in enumerate(body.links, start=2)})

    def observe(self, physics_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        labels = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.int32)
        if self.body is not None:
            points = project_body(self.body, physics_state)
            floor_width = self._floor_width(physics_state)
            x_half = max(floor_width / 2.0 if floor_width is not None else 1.0, 1.0)
            view_half_height = max(1.0, x_half * self.resolution[1] / self.resolution[0])
            y_min = -0.15
            y_max = y_min + 2.0 * view_half_height
            projected = {name: self._pixel(point, x_half, y_min, y_max) for name, point in points.items()}
            for joint in self.body.joints.values():
                if joint.parent in projected and joint.child in projected:
                    self._draw_line(labels, projected[joint.parent], projected[joint.child], self.label_map[joint.child])
            for name, pixel in projected.items():
                self._set_label(labels, pixel, self.label_map.get(name, 0), radius=max(1, min(self.resolution) // 32))
            if floor_width is not None:
                floor_y = self._pixel((0.0, 0.0), x_half, y_min, y_max)[1]
                left = self._pixel((-floor_width / 2.0, 0.0), x_half, y_min, y_max)[0]
                right = self._pixel((floor_width / 2.0, 0.0), x_half, y_min, y_max)[0]
                self._draw_line(labels, (left, floor_y), (right, floor_y), self.label_map["floor"])
        return {
            "segmentation": labels,
            "resolution": self.resolution,
            "frame_id": self.frame_id(physics_state),
            "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": self.resolution},
            "available": self.body is not None,
            "source": "headless_body_projection" if self.body is not None else "unconfigured_body_projection",
            "label_map": dict(self.label_map),
            "valid_pixels": int(np.count_nonzero(labels)),
        }

    def _pixel(self, point: tuple[float, float], x_half: float, y_min: float, y_max: float) -> tuple[int, int]:
        width, height = self.resolution
        x, y = point
        pixel_x = round((x + x_half) / (2.0 * x_half) * (width - 1))
        pixel_y = round((y_max - y) / (y_max - y_min) * (height - 1))
        return max(0, min(width - 1, pixel_x)), max(0, min(height - 1, pixel_y))

    @staticmethod
    def _draw_line(labels: np.ndarray, start: tuple[int, int], end: tuple[int, int], label: int) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for index in range(steps + 1):
            ratio = index / steps
            x = round(x0 + (x1 - x0) * ratio)
            y = round(y0 + (y1 - y0) * ratio)
            labels[y, x] = label

    @staticmethod
    def _set_label(labels: np.ndarray, center: tuple[int, int], label: int, radius: int) -> None:
        x_center, y_center = center
        height, width = labels.shape
        for y in range(max(0, y_center - radius), min(height, y_center + radius + 1)):
            for x in range(max(0, x_center - radius), min(width, x_center + radius + 1)):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                    labels[y, x] = label

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
