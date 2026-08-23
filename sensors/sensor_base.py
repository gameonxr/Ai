from abc import ABC, abstractmethod


class Sensor(ABC):
    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))

    @abstractmethod
    def observe(self, physics_state: dict) -> dict: ...

    def reset(self) -> None:
        return None
