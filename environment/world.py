from dataclasses import dataclass, field
import math

from .object_loader import load_floor
from .static_objects import Floor


@dataclass
class World:
    gravity: list[float]
    floor_enabled: bool = True
    floor_friction: float = 0.5
    floor_size: tuple[float, float] = (10.0, 10.0)
    floor: Floor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.gravity, (list, tuple)) or len(self.gravity) != 3:
            raise ValueError("gravity must be a finite 3-vector")
        try:
            gravity = [float(value) for value in self.gravity]
        except (TypeError, ValueError) as error:
            raise ValueError("gravity must be a finite 3-vector") from error
        if not all(math.isfinite(value) for value in gravity):
            raise ValueError("gravity must be a finite 3-vector")
        if not isinstance(self.floor_enabled, bool):
            raise ValueError("floor_enabled must be a boolean")
        if isinstance(self.floor_friction, bool) or not isinstance(self.floor_friction, (int, float)) or not math.isfinite(float(self.floor_friction)) or self.floor_friction < 0:
            raise ValueError("floor_friction must be a finite non-negative number")
        if not isinstance(self.floor_size, (list, tuple)) or len(self.floor_size) != 2:
            raise ValueError("floor_size must be a finite positive 2-vector")
        try:
            floor_size = tuple(float(value) for value in self.floor_size)
        except (TypeError, ValueError) as error:
            raise ValueError("floor_size must be a finite positive 2-vector") from error
        if not all(math.isfinite(value) and value > 0 for value in floor_size):
            raise ValueError("floor_size must be a finite positive 2-vector")
        self.gravity = gravity
        self.floor_friction = float(self.floor_friction)
        self.floor_size = floor_size
        self.floor = load_floor({"floor_size": self.floor_size, "floor_friction": self.floor_friction}) if self.floor_enabled else None

    def definition(self) -> dict:
        return {
            "gravity": self.gravity,
            "floor_enabled": self.floor_enabled,
            "floor_friction": self.floor_friction,
            "floor_size": self.floor_size,
            "objects": [] if self.floor is None else [{"type": "floor", "size": self.floor.size, "friction": self.floor.friction}],
        }
