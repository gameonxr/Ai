from __future__ import annotations

from typing import Any

from rendering.matplotlib_3d_renderer import project_body_3d

from .sensor_base import Sensor


class PerceptionSensor(Sensor):
    """Headless structured perception summary; this is not a learned CNN."""

    def __init__(self, name: str, config: dict | None = None, body: Any = None):
        super().__init__(name, config)
        self.body = body

    def observe(self, physics_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        if self.body is None:
            return {
                "available": False,
                "source": "unconfigured_body_projection",
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
            "visible_links": list(points),
            "link_positions": link_positions,
            "body_bounds": bounds,
            "world_objects": world_objects,
        }
