from .imu import IMUSensor
from .proprioception import ProprioceptionSensor
from .touch import TouchSensor
from .vision import VisionSensor

SENSOR_TYPES = {"proprioception": ProprioceptionSensor, "vision": VisionSensor, "imu": IMUSensor, "touch": TouchSensor}


def build_sensors(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("sensor configuration must be an object")
    sensors = {}
    for name, settings in config.items():
        if name not in SENSOR_TYPES:
            continue
        if settings is not None and not isinstance(settings, dict):
            raise ValueError(f"sensor settings for {name} must be an object")
        sensors[name] = SENSOR_TYPES[name](name, settings or {})
    return sensors
