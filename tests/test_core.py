import math
import pytest
from core import Action, Observation


def test_action_requires_command():
    with pytest.raises(ValueError): Action()
    assert Action.noop().metadata["noop"]


def test_observation_json_is_serializable():
    observation = Observation(0.5, proprioception={"joint_positions": {"a": 1.0}})
    assert '"timestamp": 0.5' in observation.to_json()


def test_invalid_timestamp_rejected():
    with pytest.raises(ValueError): Observation(math.nan)
