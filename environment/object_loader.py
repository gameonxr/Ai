from .static_objects import Floor


def load_floor(config: dict) -> Floor:
    return Floor(tuple(config.get("floor_size", [10.0, 10.0])), float(config.get("floor_friction", 0.5)))
