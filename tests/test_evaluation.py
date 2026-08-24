import pytest

from evaluation import EvaluationSummary, Evaluator
from simulator import Simulator
from training import RandomTorquePolicy


def test_evaluation_artifact_provenance_rejects_invalid_inputs():
    summary = EvaluationSummary(1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, [])
    with pytest.raises(ValueError, match="config_path must be a non-empty string"):
        summary.to_artifact_dict("", 42, 1.0)
    with pytest.raises(ValueError, match="seed must be an integer or null"):
        summary.to_artifact_dict("config.yaml", True, 1.0)
    with pytest.raises(ValueError, match="reward_per_step must be a finite number"):
        summary.to_artifact_dict("config.yaml", 42, float("nan"))


def test_evaluator_aggregates_episode_metrics():
    simulator = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(list(simulator.actuators), seed=7)
    summary = Evaluator(simulator, policy, reward_fn=lambda observation, action: 1.0).run(episodes=3, max_steps=4, seed=10)
    assert summary.episodes == 3
    assert summary.total_steps == 12
    assert summary.mean_reward == 4.0
    assert summary.reward_std == 0.0
    simulator.shutdown()


def test_evaluator_validates_limits():
    simulator = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(list(simulator.actuators), seed=7)
    evaluator = Evaluator(simulator, policy)
    with pytest.raises(ValueError):
        evaluator.run(episodes=0)
    simulator.shutdown()


def test_evaluator_rejects_invalid_numeric_inputs():
    simulator = Simulator()
    policy = RandomTorquePolicy(list(simulator.actuators), seed=1)
    evaluator = Evaluator(simulator, policy)
    try:
        invalid_inputs = [
            {"episodes": 0},
            {"episodes": 1.0},
            {"max_steps": 0},
            {"max_steps": True},
            {"seed": False},
            {"seed": 1.5},
        ]
        for overrides in invalid_inputs:
            with pytest.raises(ValueError):
                evaluator.run(**overrides)
    finally:
        simulator.shutdown()
