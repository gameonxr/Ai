from body import BodyLoader
from core import Action
import math

import pytest

from physics import MuJoCoBackend
from physics.toy_backend import ToyPhysicsEngine


def test_toy_backend_validates_initial_gravity():
    for gravity in ([0.0, 0.0], [0.0, 0.0, math.nan], "gravity"):
        with pytest.raises(ValueError, match="gravity must be a finite 3-vector"):
            ToyPhysicsEngine({"gravity": gravity})  # type: ignore[arg-type]


def test_toy_backend_rejects_malformed_motor_commands():
    engine = ToyPhysicsEngine({})
    engine.load_body(BodyLoader.load("config/body_humanoid.yaml"))
    for command, message in (({"target": "bad"}, "motor command target must be a finite number"), ({"target": math.nan}, "motor command target must be a finite number"), ({"target": 1.0, "mode": "velocity"}, "motor command mode must be torque or position")):
        with pytest.raises(ValueError, match=message):
            engine.apply_action(Action(motor_commands={"neck": command}))


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


def test_toy_backend_rejects_invalid_checkpoint_command_mode():
    engine = ToyPhysicsEngine({})
    engine.load_body(BodyLoader.load("config/body_humanoid.yaml"))
    state = engine.get_checkpoint_state()
    state["commands"] = {"neck": {"target": 1.0, "mode": "velocity"}}
    with pytest.raises(ValueError, match="checkpoint command modes must be torque or position"):
        engine.restore_checkpoint_state(state)


def test_mujoco_xml_uses_configured_gravity():
    body = BodyLoader.load("config/body_humanoid.yaml")
    backend = MuJoCoBackend({"gravity": [1.0, 2.0, -3.5]})
    xml = backend._build_xml(body, backend.gravity)
    assert '<option gravity="1.0 2.0 -3.5"/>' in xml


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
