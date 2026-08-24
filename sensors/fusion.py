from __future__ import annotations

import math
from typing import Any


_VISUAL_CHANNELS = ("vision", "depth", "segmentation", "perception")


def build_visual_fusion(readings: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Validate visual-channel alignment and return a compact fusion summary."""
    active = {name: readings[name] for name in _VISUAL_CHANNELS if name in readings}
    if not active:
        return None
    frame_ids = set()
    cameras = []
    for name, reading in active.items():
        if not isinstance(reading, dict):
            raise ValueError(f"visual sensor reading for {name} must be an object")
        frame_id = reading.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, float)) or not math.isfinite(float(frame_id)):
            raise ValueError(f"visual sensor frame_id for {name} must be finite")
        camera = reading.get("camera")
        if not isinstance(camera, dict):
            raise ValueError(f"visual sensor camera metadata for {name} must be an object")
        frame_ids.add(float(frame_id))
        cameras.append(camera)
    if len(frame_ids) != 1:
        raise ValueError("visual sensor frame_id values must match")
    base_camera = cameras[0]
    for camera in cameras[1:]:
        for key in ("projection", "coordinate_frame", "resolution"):
            if camera.get(key) != base_camera.get(key):
                raise ValueError(f"visual sensor camera metadata must match for {key}")
    channels = {
        name: {
            "available": reading.get("available") is True,
            "valid_pixels": reading.get("valid_pixels", reading.get("non_background_pixels", 0)),
        }
        for name, reading in active.items()
    }
    return {
        "aligned": True,
        "frame_id": next(iter(frame_ids)),
        "camera": dict(base_camera),
        "channels": channels,
        "available_channels": [name for name, channel in channels.items() if channel["available"]],
    }
