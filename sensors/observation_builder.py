from core.observation import Observation
from .fusion import build_visual_fusion
import math



class ObservationBuilder:
    def __init__(self, sensors: dict):
        self.sensors = sensors

    def build(self, physics_state: dict, timestamp: float) -> Observation:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be a finite number")
        readings = {name: sensor.observe(physics_state) for name, sensor in self.sensors.items() if sensor.enabled}
        visual_fusion = build_visual_fusion(readings)
        info = {"sensors_enabled": list(readings)}
        if visual_fusion is not None:
            info["visual_fusion"] = visual_fusion
        return Observation(timestamp=timestamp, proprioception=readings.get("proprioception"), vision=readings.get("vision"), depth=readings.get("depth"), segmentation=readings.get("segmentation"), perception=readings.get("perception"), imu=readings.get("imu"), touch=readings.get("touch"), info=info)
