from __future__ import annotations

from typing import Any


def projection_bounds(
    physics_state: dict[str, Any],
    resolution: tuple[int, int],
    view_scale: float,
    target: tuple[float, float] | None,
) -> dict[str, Any]:
    """Return the shared orthographic world-to-image framing contract."""
    floor_width = _floor_width(physics_state)
    x_half = max(floor_width / 2.0 if floor_width is not None else 1.0, 1.0) * view_scale
    view_half_height = max(1.0, x_half * resolution[1] / resolution[0])
    target_x, target_y = target if target is not None else (0.0, -0.15 + view_half_height)
    return {
        "floor_width": floor_width,
        "x_half": x_half,
        "y_half": view_half_height,
        "target": (target_x, target_y),
        "x_min": target_x - x_half,
        "x_max": target_x + x_half,
        "y_min": target_y - view_half_height,
        "y_max": target_y + view_half_height,
    }


def camera_metadata(
    resolution: tuple[int, int],
    view_scale: float,
    target: tuple[float, float] | None,
    bounds: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    metadata = {
        "projection": "orthographic",
        "coordinate_frame": "body_debug",
        "resolution": resolution,
        "view_scale": view_scale,
        "target": bounds["target"],
        "world_bounds": {
            "x": (bounds["x_min"], bounds["x_max"]),
            "y": (bounds["y_min"], bounds["y_max"]),
        },
    }
    metadata.update(extra)
    return metadata


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
