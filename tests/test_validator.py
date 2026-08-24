import math

import pytest


from actuators.actuator_registry import build_actuators
from body import BodyLoader
from core import Action
from core.validator import ActionValidator


def validator():
    body = BodyLoader.load("config/body_humanoid.yaml")
    return ActionValidator(build_actuators(body.joints, {"defaults": {}})), body


def test_actuator_registry_rejects_malformed_configuration():
    with pytest.raises(ValueError, match="actuator configuration must be an object"):
        build_actuators({}, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actuator defaults must be an object"):
        build_actuators({}, {"defaults": []})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control_mode must be"):
        build_actuators({}, {"control_mode": "velocity"})


def test_unknown_and_nan_commands_are_rejected():
    check, body = validator()
    valid, _, errors = check.validate(Action(joint_targets={"missing": 1.0, "neck": math.nan}))
    assert not valid and len(errors) == 2


def test_unsupported_motor_mode_is_rejected():
    check, _ = validator()
    valid, action, errors = check.validate(Action(motor_commands={"neck": {"target": 0.1, "mode": "velocity"}}))
    assert not valid
    assert action.metadata.get("noop") is True
    assert errors == ["Invalid control mode for neck: 'velocity'"]


def test_commands_are_clamped():
    check, body = validator()
    valid, action, errors = check.validate(Action(joint_targets={"neck": 1000.0}))
    assert valid and not errors
    assert action.joint_targets["neck"] <= body.joints["neck"].max_torque
