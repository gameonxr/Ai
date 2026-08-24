from core.observation import Observation


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
        return Observation(timestamp=timestamp, proprioception=readings.get("proprioception"), vision=readings.get("vision"), depth=readings.get("depth"), segmentation=readings.get("segmentation"), imu=readings.get("imu"), touch=readings.get("touch"), info={"sensors_enabled": list(readings)})
