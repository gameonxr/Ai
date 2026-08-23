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


def test_timestamp_must_be_numeric():
    for value in (True, "0.5"):
        with pytest.raises(ValueError, match="finite"):
            Observation(value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite"):
            Action.noop(timestamp=value)  # type: ignore[arg-type]


def test_action_metadata_must_be_an_object():
    with pytest.raises(ValueError, match="Action metadata must be a JSON object"):
        Action(joint_targets={"neck": 0.1}, metadata=[])  # type: ignore[arg-type]


def test_observation_info_must_be_an_object():
    with pytest.raises(ValueError, match="Observation info must be a JSON object"):
        Observation(0.0, info=[])  # type: ignore[arg-type]
