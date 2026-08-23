import math
from actuators.actuator_registry import build_actuators
from body import BodyLoader
from core import Action
from core.validator import ActionValidator


def validator():
    body = BodyLoader.load("config/body_humanoid.yaml")
    return ActionValidator(build_actuators(body.joints, {"defaults": {}})), body


def test_unknown_and_nan_commands_are_rejected():
    check, body = validator()
    valid, _, errors = check.validate(Action(joint_targets={"missing": 1.0, "neck": math.nan}))
    assert not valid and len(errors) == 2


def test_commands_are_clamped():
    check, body = validator()
    valid, action, errors = check.validate(Action(joint_targets={"neck": 1000.0}))
    assert valid and not errors
    assert action.joint_targets["neck"] <= body.joints["neck"].max_torque
