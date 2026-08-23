from dataclasses import dataclass


@dataclass
class World:
    gravity: list[float]
    floor_enabled: bool = True
    floor_friction: float = 0.5
    floor_size: tuple[float, float] = (10.0, 10.0)

    def definition(self) -> dict:
        return {"gravity": self.gravity, "floor_enabled": self.floor_enabled, "floor_friction": self.floor_friction, "floor_size": self.floor_size}
