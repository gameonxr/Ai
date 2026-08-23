from .sensor_base import Sensor


class VisionSensor(Sensor):
    """Phase 1 camera placeholder; no rendering is performed."""
    def observe(self, physics_state):
        resolution = tuple(self.config.get("resolution", [64, 64]))
        return {"rgb": None, "resolution": resolution, "fov": float(self.config.get("fov", 90.0)), "available": False}
