from body import BodyLoader
from core import Action
import math

import pytest

from physics import MuJoCoBackend


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
