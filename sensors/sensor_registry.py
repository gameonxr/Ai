from .imu import IMUSensor
from .proprioception import ProprioceptionSensor
from .touch import TouchSensor
from .vision import VisionSensor

SENSOR_TYPES = {"proprioception": ProprioceptionSensor, "vision": VisionSensor, "imu": IMUSensor, "touch": TouchSensor}


def build_sensors(config: dict) -> dict:
    return {name: SENSOR_TYPES[name](name, settings or {}) for name, settings in config.items() if name in SENSOR_TYPES}
