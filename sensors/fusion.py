from __future__ import annotations

import math
from typing import Any

import numpy as np


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
        _validate_channel(name, reading, camera)
        frame_ids.add(float(frame_id))
        cameras.append(camera)
    if len(frame_ids) != 1:
        raise ValueError("visual sensor frame_id values must match")
    base_camera = cameras[0]
    for camera in cameras[1:]:
        for key in ("projection", "coordinate_frame"):
            if camera.get(key) != base_camera.get(key):
                raise ValueError(f"visual sensor camera metadata must match for {key}")
        if camera.get("resolution") is not None and base_camera.get("resolution") is not None and camera.get("resolution") != base_camera.get("resolution"):
            raise ValueError("visual sensor camera metadata must match for resolution")
    channels = {
        name: {
            "available": reading.get("available") is True,
            "valid_pixels": int(reading.get("valid_pixels", reading.get("non_background_pixels", 0))),
            "shape": list(reading["rgb"].shape if name == "vision" else reading["depth"].shape if name == "depth" else reading["segmentation"].shape) if name in {"vision", "depth", "segmentation"} else None,
            "dtype": str(reading["rgb"].dtype if name == "vision" else reading["depth"].dtype if name == "depth" else reading["segmentation"].dtype) if name in {"vision", "depth", "segmentation"} else None,
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


def _validate_channel(name: str, reading: dict[str, Any], camera: dict[str, Any]) -> None:
    resolution = camera.get("resolution")
    expected_shape = None
    if resolution is not None:
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resolution):
            raise ValueError(f"visual sensor camera resolution for {name} must be a pair of positive integers")
        expected_shape = (int(resolution[1]), int(resolution[0]))
    if name == "vision":
        frame = reading.get("rgb")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.dtype(np.uint8) or (expected_shape is not None and frame.shape[:2] != expected_shape):
            raise ValueError("visual sensor RGB frame must be uint8 with shape height x width x 3")
    elif name == "depth":
        frame = reading.get("depth")
        if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype != np.dtype(np.float32) or (expected_shape is not None and frame.shape != expected_shape) or not np.all(np.isfinite(frame)):
            raise ValueError("visual sensor depth frame must be finite float32 with shape height x width")
    elif name == "segmentation":
        frame = reading.get("segmentation")
        if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype != np.dtype(np.int32) or (expected_shape is not None and frame.shape != expected_shape) or np.any(frame < 0):
            raise ValueError("visual sensor segmentation frame must be non-negative int32 with shape height x width")
