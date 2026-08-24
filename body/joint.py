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
        if not isinstance(config, dict):
            raise ValueError("Joint config must be a mapping")
        axis = config.get("axis", [0, 0, 1])
        joint_range = config.get("range", [-180, 180])
        if not cls._finite_vector(axis, 3):
            raise ValueError("Joint axis must be a finite 3-vector")
        if not cls._finite_vector(joint_range, 2) or float(joint_range[0]) >= float(joint_range[1]):
            raise ValueError("Joint range must be a finite 2-vector with lower bound below upper bound")
        max_torque = cls._numeric(config.get("max_torque", 100.0), "max_torque", positive=True)
        damping = cls._numeric(config.get("damping", 0.01), "damping")
        return cls(str(name), str(config.get("type", "revolute")), str(config["parent"]), str(config["child"]), list(map(float, axis)), list(map(float, joint_range)), max_torque, damping)

    @staticmethod
    def _finite_vector(value, size: int) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == size and all(not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)

    @staticmethod
    def _numeric(value, name: str, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Joint {name} must be a finite number")
        numeric = float(value)
        if positive and numeric <= 0:
            raise ValueError(f"Joint {name} must be positive")
        if not positive and numeric < 0:
            raise ValueError(f"Joint {name} must be non-negative")
        return numeric
