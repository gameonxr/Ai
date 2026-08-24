import pytest

from evaluation import EvaluationSummary, Evaluator
from simulator import Simulator
from training import RandomTorquePolicy


def test_evaluation_summary_artifact_rejects_malformed_summary():
    malformed = EvaluationSummary(1, 1, float("nan"), 0.0, 1.0, 1.0, 1.0, [])
    with pytest.raises(ValueError, match="mean_reward must be a finite number"):
        malformed.to_artifact_dict("config.yaml", 42, 1.0)

    malformed = EvaluationSummary(1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, [{"episode": 0, "steps": 1, "total_reward": 1.0, "terminated": "no"}])
    with pytest.raises(ValueError, match="episode metric terminated must be boolean"):
        malformed.to_artifact_dict("config.yaml", 42, 1.0)


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


def test_evaluator_honors_done_callback_and_reports_termination():
    simulator = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(["neck"], seed=7)
    evaluator = Evaluator(simulator, policy, reward_fn=lambda observation, action: 1.0, done_fn=lambda observation, step: step == 2)
    summary = evaluator.run(episodes=2, max_steps=5, seed=10)
    assert summary.total_steps == 4
    assert all(item["terminated"] is True and item["steps"] == 2 for item in summary.episode_metrics)
    simulator.shutdown()


def test_evaluator_rejects_invalid_callback_results():
    simulator = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(["neck"], seed=7)
    try:
        with pytest.raises(ValueError, match="reward_fn must return a finite number"):
            Evaluator(simulator, policy, reward_fn=lambda observation, action: "1.0").run(max_steps=1)
        with pytest.raises(ValueError, match="done_fn must return a boolean"):
            Evaluator(simulator, policy, done_fn=lambda observation, step: 1).run(max_steps=1)
    finally:
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
