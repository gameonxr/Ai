import pytest

from core import Action
from vector_env import VectorizedSimulator


def test_vectorized_reset_and_step():
    vector = VectorizedSimulator("config/simulator_config.yaml", num_envs=3)
    observations = vector.reset(seed=20)
    assert len(observations) == 3
    next_observations = vector.step([Action(joint_targets={"neck": 0.0}) for _ in observations])
    assert len(next_observations) == 3
    assert [item.timestamp for item in next_observations] == [0.005] * 3
    assert [state["step_count"] for state in vector.get_states()] == [2, 2, 2]
    vector.close()


def test_vectorized_num_envs_requires_positive_integer():
    for value in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="num_envs must be a positive integer"):
            VectorizedSimulator("config/simulator_config.yaml", num_envs=value)  # type: ignore[arg-type]


def test_vectorized_reset_seed_requires_integer_or_null():
    vector = VectorizedSimulator("config/simulator_config.yaml", num_envs=1)
    try:
        for value in (True, 1.5, "20"):
            with pytest.raises(ValueError, match="seed must be an integer or null"):
                vector.reset(seed=value)  # type: ignore[arg-type]
    finally:
        vector.close()


def test_vectorized_step_requires_action_container():
    vector = VectorizedSimulator("config/simulator_config.yaml", num_envs=1)
    vector.reset(seed=1)
    try:
        with pytest.raises(ValueError, match="actions must be a list or tuple"):
            vector.step("not-actions")  # type: ignore[arg-type]
    finally:
        vector.close()


def test_vectorized_action_count_is_validated():
    vector = VectorizedSimulator("config/simulator_config.yaml", num_envs=2)
    vector.reset(seed=1)
    with pytest.raises(ValueError):
        vector.step([Action(joint_targets={"neck": 0.0})])
    vector.close()
