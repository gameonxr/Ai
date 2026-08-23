from dataclasses import dataclass
import math


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: list[float]
    range_degrees: list[float]
    max_torque: float
    damping: float

    @property
    def range_radians(self) -> tuple[float, float]:
        return tuple(math.radians(x) for x in self.range_degrees)

    def clamp(self, value: float) -> float:
        low, high = self.range_radians
        return max(low, min(high, float(value)))

    @classmethod
    def from_config(cls, name: str, config: dict) -> "Joint":
        return cls(name, str(config.get("type", "revolute")), str(config["parent"]), str(config["child"]), list(config.get("axis", [0, 0, 1])), list(config.get("range", [-180, 180])), float(config.get("max_torque", 100.0)), float(config.get("damping", 0.01)))
