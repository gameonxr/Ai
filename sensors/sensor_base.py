from abc import ABC, abstractmethod


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

    def reset(self) -> None:
        return None
