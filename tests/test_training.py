import json
from pathlib import Path

import pytest

from core import Action, Observation
from simulator import Simulator
from training import RandomTorquePolicy, Rollout, Trainer
from training.rollout import Transition


def test_random_torque_policy_validates_constructor_inputs():
    invalid = [
        (([], 0.05, None), "joint_names must be a non-empty list of strings"),
        ((["neck", "neck"], 0.05, None), "joint_names must be unique"),
        ((["neck"], float("nan"), None), "scale must be a finite non-negative number"),
        ((["neck"], -0.1, None), "scale must be a finite non-negative number"),
        ((["neck"], 0.05, True), "seed must be an integer or null"),
    ]
    for (joint_names, scale, seed), message in invalid:
        with pytest.raises(ValueError, match=message):
            RandomTorquePolicy(joint_names, scale=scale, seed=seed)


def test_trainer_rejects_invalid_episode_inputs(tmp_path: Path):
    simulator = Simulator("config/simulator_config.yaml")
    trainer = Trainer(simulator, RandomTorquePolicy(["neck"], seed=1))
    try:
        invalid_calls = (
            ("run_episode", {"max_steps": 0}, "max_steps must be a positive integer"),
            ("run_episode", {"max_steps": True}, "max_steps must be a positive integer"),
            ("run_episode", {"seed": False}, "seed must be an integer or null"),
            ("train", {"episodes": 0}, "episodes must be a positive integer"),
            ("train", {"episodes": 1.5}, "episodes must be a positive integer"),
        )
        for method_name, kwargs, message in invalid_calls:
            with pytest.raises(ValueError, match=message):
                getattr(trainer, method_name)(**kwargs)
    finally:
        trainer.close()


def test_random_torque_policy_reset_rejects_invalid_seed():
    policy = RandomTorquePolicy(["neck"], seed=1)
    for value in (True, 1.5, "7"):
        with pytest.raises(ValueError, match="seed must be an integer or null"):
            policy.reset(value)  # type: ignore[arg-type]


def test_rollout_jsonl_persistence(tmp_path: Path):
    sim = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(list(sim.actuators), seed=1)
    trainer = Trainer(sim, policy, reward_fn=lambda observation, action: 2.0)
    rollout, metrics = trainer.run_episode(max_steps=4, seed=1)
    output = tmp_path / "rollout.jsonl"
    rollout.save_jsonl(output)
    assert metrics.steps == 4 and metrics.total_reward == 8.0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["reward"] == 2.0
    trainer.close()


def test_rollout_append_rejects_invalid_transition_inputs():
    rollout = Rollout()
    observation = Observation(0.0)
    action = Action.noop()
    cases = [
        ((None, action, 0.0, False), "observation must be an Observation"),
        ((observation, None, 0.0, False), "action must be an Action"),
        ((observation, action, float("nan"), False), "reward must be a finite number"),
        ((observation, action, 0.0, 1), "done must be a boolean"),
    ]
    for args, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            rollout.append(*args)


def test_rollout_append_rejects_non_mapping_info():
    rollout = Rollout()
    with pytest.raises(ValueError, match="info must be a mapping"):
        rollout.append(Observation(0.0), Action.noop(), 0.0, info=["not", "a", "mapping"])


def test_rollout_save_preserves_existing_file_on_serialization_failure(tmp_path: Path):
    path = tmp_path / "rollout.jsonl"
    path.write_text("existing\n", encoding="utf-8")
    rollout = Rollout()
    rollout.transitions.append(Transition({}, {}, 1.0, False, {"bad": object()}))
    with pytest.raises(TypeError):
        rollout.save_jsonl(path)
    assert path.read_text(encoding="utf-8") == "existing\n"


def test_trainer_seeded_policy_is_reproducible():
    def collect():
        sim = Simulator("config/simulator_config.yaml")
        trainer = Trainer(sim, RandomTorquePolicy(list(sim.actuators), seed=9), reward_fn=lambda observation, action: 1.0)
        rollout, _ = trainer.run_episode(max_steps=3, seed=9)
        trainer.close()
        return [item.action for item in rollout.transitions]

    assert collect() == collect()
