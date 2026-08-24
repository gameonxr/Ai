from dataclasses import dataclass, field
import math


@dataclass
class Link:
    name: str
    mass: float
    inertia: list[float]
    collision_shape: str
    properties: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, config: dict) -> "Link":
        if not isinstance(config, dict):
            raise ValueError("Link config must be a mapping")
        mass = config.get("mass", 1.0)
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(float(mass)) or float(mass) <= 0:
            raise ValueError("Link mass must be a finite positive number")
        inertia = config.get("inertia", [1.0, 1.0, 1.0])
        if not isinstance(inertia, (list, tuple)) or len(inertia) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in inertia):
            raise ValueError("Link inertia must be a finite positive 3-vector")
        collision_shape = config.get("collision_shape", "box")
        if not isinstance(collision_shape, str) or not collision_shape.strip():
            raise ValueError("Link collision_shape must be a non-empty string")
        return cls(str(name), float(mass), [float(value) for value in inertia], collision_shape, {k: v for k, v in config.items() if k not in {"mass", "inertia", "collision_shape"}})
