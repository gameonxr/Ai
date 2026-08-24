import math

import pytest

from environment import World
from environment.object_loader import load_floor


def test_world_normalizes_valid_configuration():
    world = World([0, 0, -9.81], floor_friction=0.5, floor_size=[10, 12])
    assert world.gravity == [0.0, 0.0, -9.81]
    assert world.floor_friction == 0.5
    assert world.floor_size == (10.0, 12.0)


def test_world_exposes_static_floor_object():
    world = World([0, 0, -9.81], floor_friction=0.25, floor_size=[10, 12])
    assert world.floor is not None
    assert world.floor.size == (10.0, 12.0)
    assert world.floor.friction == 0.25
    assert world.definition()["objects"] == [{"type": "floor", "size": (10.0, 12.0), "friction": 0.25}]

    disabled = World([0, 0, -9.81], floor_enabled=False)
    assert disabled.floor is None
    assert disabled.definition()["objects"] == []


def test_load_floor_applies_defaults_and_returns_normalized_floor():
    floor = load_floor()
    assert floor.size == (10.0, 10.0)
    assert floor.friction == 0.5

    configured = load_floor({"floor_size": [4, 6], "floor_friction": 0.75})
    assert configured.size == (4.0, 6.0)
    assert configured.friction == 0.75


def test_load_floor_rejects_invalid_object_configuration():
    with pytest.raises(ValueError, match="floor config must be a mapping"):
        load_floor([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floor_size must be a finite positive 2-vector"):
        load_floor({"floor_size": [4, True]})
    with pytest.raises(ValueError, match="floor_size must be a finite positive 2-vector"):
        load_floor({"floor_size": [4, 0]})
    with pytest.raises(ValueError, match="floor_friction must be a finite number"):
        load_floor({"floor_friction": "0.5"})
    with pytest.raises(ValueError, match="floor_friction must be a finite non-negative number"):
        load_floor({"floor_friction": -0.1})


def test_world_rejects_invalid_gravity():
    for value in ([0.0, 0.0], [0.0, 0.0, math.nan], "gravity"):
        with pytest.raises(ValueError, match="gravity must be a finite 3-vector"):
            World(value)  # type: ignore[arg-type]


def test_world_rejects_invalid_floor_configuration():
    with pytest.raises(ValueError, match="floor_enabled must be a boolean"):
        World([0.0, 0.0, -9.81], floor_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floor_friction must be a finite non-negative number"):
        World([0.0, 0.0, -9.81], floor_friction=-0.1)
    with pytest.raises(ValueError, match="floor_friction must be a finite non-negative number"):
        World([0.0, 0.0, -9.81], floor_friction=math.inf)
    for value in ([10.0], [10.0, math.nan], [10.0, 0.0], "size"):
        with pytest.raises(ValueError, match="floor_size must be a finite positive 2-vector"):
            World([0.0, 0.0, -9.81], floor_size=value)  # type: ignore[arg-type]
