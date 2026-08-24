import json
from pathlib import Path

import pytest

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


def test_metrics_restore_snapshot_restores_counters_only():
    metrics = SimulationMetrics()
    metrics.reset_episode()
    metrics.record_step(0.005, action_applied=True, invalid_action=True)
    metrics.restore_snapshot({"steps": 8, "actions": 3, "invalid_actions": 2, "episodes": 4, "simulation_seconds": 0.04, "wall_seconds": 999.0})
    snapshot = metrics.snapshot()
    assert snapshot["steps"] == 8
    assert snapshot["actions"] == 3
    assert snapshot["invalid_actions"] == 2
    assert snapshot["episodes"] == 4
    assert snapshot["simulation_seconds"] == 0.04
    assert snapshot["wall_seconds"] < 999.0


def test_metrics_restore_snapshot_rejects_invalid_values():
    metrics = SimulationMetrics()
    invalid = [
        ([], "Metrics snapshot must be an object"),
        ({"steps": -1}, "Metrics steps must be a non-negative integer"),
        ({"actions": True}, "Metrics actions must be a non-negative integer"),
        ({"simulation_seconds": float("nan")}, "Metrics simulation_seconds must be a finite non-negative number"),
    ]
    for snapshot, message in invalid:
        with pytest.raises(ValueError, match=message):
            metrics.restore_snapshot(snapshot)


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
