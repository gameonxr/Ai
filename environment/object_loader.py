import math
from collections.abc import Mapping

from .static_objects import Floor


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number")
    return numeric


def load_floor(config: Mapping[str, object] | None = None) -> Floor:
    """Build a validated static floor from an environment object mapping."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("floor config must be a mapping")

    raw_size = config.get("floor_size", [10.0, 10.0])
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raise ValueError("floor_size must be a finite positive 2-vector")
    try:
        size = tuple(_finite_number(value, "floor_size") for value in raw_size)
    except ValueError as error:
        raise ValueError("floor_size must be a finite positive 2-vector") from error
    if not all(value > 0 for value in size):
        raise ValueError("floor_size must be a finite positive 2-vector")

    friction = _finite_number(config.get("floor_friction", 0.5), "floor_friction")
    if friction < 0:
        raise ValueError("floor_friction must be a finite non-negative number")

    return Floor(size=size, friction=friction)
