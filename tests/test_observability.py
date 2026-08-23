import json
from pathlib import Path

from brain import DummyBrain
from observability import SimulationMetrics
from simulator import Simulator


def test_metrics_counts_steps_actions_and_episodes():
    metrics = SimulationMetrics()
    metrics.reset_episode()
    metrics.record_step(0.005, action_applied=True)
    metrics.record_step(0.005, invalid_action=True)
    snapshot = metrics.snapshot()
    assert snapshot["episodes"] == 1
    assert snapshot["steps"] == 2
    assert snapshot["actions"] == 1
    assert snapshot["invalid_actions"] == 1
    assert snapshot["simulation_seconds"] == 0.01


def test_simulator_exposes_metrics_and_writes_json_logs():
    log_path = Path("logs/simulation.log")
    if log_path.exists():
        log_path.unlink()
    simulator = Simulator("config/simulator_config.yaml")
    simulator.set_brain(DummyBrain())
    simulator.reset(seed=4)
    simulator.step(2)
    assert simulator.get_state()["metrics"]["steps"] == 2
    simulator.shutdown()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["event"] == "episode_reset" for line in lines)
