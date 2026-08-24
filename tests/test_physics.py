from body import BodyLoader
from core import Action
import math

import pytest

from physics import MuJoCoBackend
from physics.toy_backend import ToyPhysicsEngine


def test_toy_backend_rejects_invalid_reset_seed():
    body = BodyLoader.load("config/body_humanoid.yaml")
    engine = ToyPhysicsEngine({})
    engine.load_body(body)
    for value in (True, 1.5, "42"):
        with pytest.raises(ValueError, match="seed must be an integer or null"):
            engine.reset(value)  # type: ignore[arg-type]


def test_toy_backend_rejects_malformed_checkpoint_state():
    body = BodyLoader.load("config/body_humanoid.yaml")
    engine = ToyPhysicsEngine({})
    engine.load_body(body)
    state = engine.get_checkpoint_state()
    state["body_velocity"] = [0.0, 0.0, math.nan]
    with pytest.raises(ValueError, match="body_velocity must be a finite vector"):
        engine.restore_checkpoint_state(state)
    state = engine.get_checkpoint_state()
    state["positions"].pop("neck")
    with pytest.raises(ValueError, match="positions dof map does not match"):
        engine.restore_checkpoint_state(state)


def test_backend_is_deterministic():
    body = BodyLoader.load("config/body_humanoid.yaml")
    first, second = MuJoCoBackend({}), MuJoCoBackend({})
    for engine in (first, second):
        engine.load_body(body); engine.reset(seed=42); engine.apply_action(Action(joint_targets={"neck": 0.2})); engine.step()
    assert first.get_body_state() == second.get_body_state()


def test_backend_step_requires_finite_numeric_timestep():
    body = BodyLoader.load("config/body_humanoid.yaml")
    engine = MuJoCoBackend({})
    engine.load_body(body)
    for value in (True, "0.005", math.nan):
        with pytest.raises(ValueError, match="finite number"):
            engine.step(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt must be in"):
        engine.step(0.0)


def test_backend_gravity_requires_finite_three_vector():
    engine = MuJoCoBackend({})
    for value in ("gravity", [0.0, 0.0], [0.0, 0.0, math.nan]):
        with pytest.raises(ValueError, match="gravity must be a finite 3-vector"):
            engine.set_gravity(value)  # type: ignore[arg-type]
