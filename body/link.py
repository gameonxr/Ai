from dataclasses import dataclass, field


@dataclass
class Link:
    name: str
    mass: float
    inertia: list[float]
    collision_shape: str
    properties: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, config: dict) -> "Link":
        return cls(name, float(config.get("mass", 1.0)), list(config.get("inertia", [1.0, 1.0, 1.0])), str(config.get("collision_shape", "box")), {k: v for k, v in config.items() if k not in {"mass", "inertia", "collision_shape"}})
