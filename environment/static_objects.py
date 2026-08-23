from dataclasses import dataclass


@dataclass
class Floor:
    size: tuple[float, float] = (10.0, 10.0)
    friction: float = 0.5
