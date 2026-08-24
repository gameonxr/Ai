import math

from .sensor_base import Sensor


class VisionSensor(Sensor):
    """Phase 1 camera placeholder; no rendering is performed."""

    def __init__(self, name: str, config: dict | None = None):
        super().__init__(name, config)
        resolution = self.config.get("resolution", [64, 64])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resolution):
            raise ValueError("vision resolution must be a pair of positive integers")
        fov = self.config.get("fov", 90.0)
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not math.isfinite(float(fov)) or not 0 < float(fov) <= 180:
            raise ValueError("vision fov must be a finite number in (0, 180]")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.fov = float(fov)

    def observe(self, physics_state):
        return {"rgb": None, "resolution": self.resolution, "fov": self.fov, "available": False}
