from __future__ import annotations

from typing import Any
import math

from rendering.matplotlib_3d_renderer import project_body_3d

from .sensor_base import Sensor


class PerceptionSensor(Sensor):
    """Headless structured perception summary; this is not a learned CNN."""

    def __init__(self, name: str, config: dict | None = None, body: Any = None):
        super().__init__(name, config)
        view_scale = self.config.get("view_scale", 1.0)
        if isinstance(view_scale, bool) or not isinstance(view_scale, (int, float)) or not math.isfinite(float(view_scale)) or float(view_scale) <= 0:
            raise ValueError("perception view_scale must be a finite positive number")
        self.view_scale = float(view_scale)
        self.body = body

    def observe(self, physics_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        if self.body is None:
            return {
                "available": False,
                "source": "unconfigured_body_projection",
                "frame_id": self.frame_id(physics_state),
                "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "view_scale": self.view_scale},
                "visible_links": [],
                "link_positions": {},
                "body_bounds": None,
                "world_objects": [],
            }
        points = project_body_3d(self.body, physics_state)
        link_positions = {
            name: {
                "position": list(point),
                "depth": float(2.0 - point[1]),
                "visible": True,
            }
            for name, point in points.items()
        }
        coordinates = list(points.values())
        bounds = {
            "min": [min(point[index] for point in coordinates) for index in range(3)],
            "max": [max(point[index] for point in coordinates) for index in range(3)],
        }
        world = physics_state.get("world", {})
        world_objects = []
        if isinstance(world, dict):
            for object_definition in world.get("objects", []):
                if isinstance(object_definition, dict) and isinstance(object_definition.get("type"), str):
                    world_objects.append({"type": object_definition["type"], "visible": True})
        return {
            "available": True,
            "source": "headless_body_projection",
            "frame_id": self.frame_id(physics_state),
            "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "view_scale": self.view_scale},
            "visible_links": list(points),
            "link_positions": link_positions,
            "body_bounds": bounds,
            "world_objects": world_objects,
        }
