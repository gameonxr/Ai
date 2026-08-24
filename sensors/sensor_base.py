from abc import ABC, abstractmethod
import math



class Sensor(ABC):
    def __init__(self, name: str, config: dict | None = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("sensor name must be a non-empty string")
        if config is not None and not isinstance(config, dict):
            raise ValueError("sensor config must be an object")
        config = config or {}
        if "enabled" in config and not isinstance(config["enabled"], bool):
            raise ValueError("sensor enabled must be a boolean")
        self.name = name
        self.config = dict(config)
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def observe(self, physics_state: dict) -> dict: ...

    @staticmethod
    def frame_id(physics_state: dict) -> float:
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        value = physics_state.get("time", 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("physics_state time must be a finite number")
        return float(value)

    def camera_target(self) -> tuple[float, float] | None:
        value = self.config.get("target")
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("camera target must be a pair of finite numbers")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in value):
            raise ValueError("camera target must be a pair of finite numbers")
        return float(value[0]), float(value[1])

    def reset(self) -> None:
        return None
