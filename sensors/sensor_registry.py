from .depth import DepthSensor
from .segmentation import SegmentationSensor
from .imu import IMUSensor
from .perception import PerceptionSensor
from .proprioception import ProprioceptionSensor
from .touch import TouchSensor
from .vision import VisionSensor

SENSOR_TYPES = {"proprioception": ProprioceptionSensor, "vision": VisionSensor, "depth": DepthSensor, "segmentation": SegmentationSensor, "perception": PerceptionSensor, "imu": IMUSensor, "touch": TouchSensor}


def build_sensors(config: dict, body=None) -> dict:
    if not isinstance(config, dict):
        raise ValueError("sensor configuration must be an object")
    sensors = {}
    for name, settings in config.items():
        if name not in SENSOR_TYPES:
            continue
        if settings is not None and not isinstance(settings, dict):
            raise ValueError(f"sensor settings for {name} must be an object")
        sensor_config = settings or {}
        sensors[name] = SENSOR_TYPES[name](name, sensor_config, body=body) if name in {"vision", "depth", "segmentation", "perception"} else SENSOR_TYPES[name](name, sensor_config)
    return sensors
