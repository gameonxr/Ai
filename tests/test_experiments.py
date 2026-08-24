import json
from pathlib import Path

import pytest

from experiments import ExperimentRunner, RunManifest


def test_experiment_runner_writes_manifest_and_checkpoint(tmp_path: Path):
    runner = ExperimentRunner("config/simulator_config.yaml", "test-run", tmp_path)
    manifest = runner.run(episodes=2, max_steps=3, seed=20, checkpoint_every=1)
    assert manifest.status == "completed"
    assert manifest.episodes_completed == 2
    assert manifest.total_steps == 6
    assert (tmp_path / "test-run.json").exists()
    assert (tmp_path / "test-run.checkpoint.json").exists()
    payload = json.loads((tmp_path / "test-run.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "test-run"
    assert len(payload["metrics"]) == 2


def test_run_manifest_artifact_rejects_malformed_fields():
    base = RunManifest("run", "completed", "started", "finished", "config.yaml", 1, 1, 1)
    malformed = [
        ("status", "paused", "status must be running, completed, or failed"),
        ("episodes_completed", 2, "episodes_completed cannot exceed episodes_requested"),
        ("metrics", [{"episode": 0, "steps": 1, "total_reward": float("nan")}], "metric total_reward must be a finite number"),
    ]
    for field, value, message in malformed:
        candidate = RunManifest(**{**base.__dict__, field: value})
        with pytest.raises(ValueError, match=message):
            candidate.to_artifact_dict()


def test_experiment_runner_validates_limits(tmp_path: Path):
    runner = ExperimentRunner(manifest_dir=tmp_path)
    with pytest.raises(ValueError):
        runner.run(episodes=0)
    with pytest.raises(ValueError):
        runner.run(max_steps=0)


def test_experiment_runner_rejects_invalid_numeric_inputs(tmp_path: Path):
    runner = ExperimentRunner("config/simulator_config.yaml", "invalid-input", tmp_path)
    invalid_inputs = [
        {"episodes": 0},
        {"episodes": 1.0},
        {"max_steps": 0},
        {"max_steps": True},
        {"seed": False},
        {"seed": 1.5},
        {"checkpoint_every": -1},
        {"checkpoint_every": 1.0},
    ]
    for overrides in invalid_inputs:
        with pytest.raises(ValueError):
            runner.run(**overrides)


def test_experiment_runner_rejects_empty_constructor_inputs(tmp_path: Path):
    with pytest.raises(ValueError):
        ExperimentRunner("", "run", tmp_path)
    with pytest.raises(ValueError):
        ExperimentRunner("config/simulator_config.yaml", "", tmp_path)
    with pytest.raises(ValueError):
        ExperimentRunner(None, "run", tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExperimentRunner("config/simulator_config.yaml", None, tmp_path)  # type: ignore[arg-type]
