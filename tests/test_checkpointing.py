import json
from pathlib import Path

from brain import DummyBrain
from simulator import Simulator
from checkpointing import CheckpointManager


def test_checkpoint_round_trip_and_resume(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    simulator = Simulator("config/simulator_config.yaml")
    simulator.set_brain(DummyBrain())
    simulator.reset(seed=12)
    simulator.step(4)
    expected_state = simulator.physics.get_checkpoint_state()
    checkpoint = simulator.save_checkpoint(path, run_id="test", metadata={"seed": 12})
    simulator.step(3)
    restored = simulator.restore_checkpoint(path)
    assert checkpoint.run_id == "test"
    assert restored.metadata["seed"] == 12
    assert simulator.step_count == 4
    assert simulator.current_time == 0.02
    assert simulator.physics.get_checkpoint_state() == expected_state
    simulator.shutdown()


def test_checkpoint_version_is_validated(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": 999}', encoding="utf-8")
    try:
        CheckpointManager.load(path)
    except ValueError as error:
        assert "Unsupported checkpoint version" in str(error)
    else:
        raise AssertionError("Invalid checkpoint version was accepted")


def test_checkpoint_version_must_be_numeric(tmp_path: Path):
    path = tmp_path / "non-numeric-version.json"
    path.write_text('{"version": "future"}', encoding="utf-8")
    try:
        CheckpointManager.load(path)
    except ValueError as error:
        assert "Unsupported checkpoint version: future" in str(error)
    else:
        raise AssertionError("Non-numeric checkpoint version was accepted")


def test_checkpoint_persists_artifact_taxonomy(tmp_path: Path):
    simulator = Simulator("config/simulator_config.yaml")
    try:
        path = tmp_path / "checkpoint.json"
        CheckpointManager.save(simulator, path, "typed-checkpoint")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["artifact_type"] == "checkpoint"
        assert payload["schema_version"] == 1
        loaded = CheckpointManager.load(path)
        assert loaded.artifact_type == "checkpoint"
        assert loaded.schema_version == 1
    finally:
        simulator.shutdown()


def test_legacy_untyped_checkpoint_defaults_to_current_taxonomy(tmp_path: Path):
    simulator = Simulator("config/simulator_config.yaml")
    try:
        path = tmp_path / "legacy.json"
        CheckpointManager.save(simulator, path, "legacy-checkpoint")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("artifact_type")
        payload.pop("schema_version")
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = CheckpointManager.load(path)
        assert loaded.artifact_type == "checkpoint"
        assert loaded.schema_version == 1
    finally:
        simulator.shutdown()


def test_checkpoint_payload_must_be_an_object(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    try:
        CheckpointManager.load(path)
    except ValueError as error:
        assert "Checkpoint payload must be a JSON object" in str(error)
    else:
        raise AssertionError("Non-object checkpoint payload was accepted")


def test_checkpoint_artifact_type_is_validated(tmp_path: Path):
    path = tmp_path / "wrong-type.json"
    path.write_text(json.dumps({"artifact_type": "evaluation", "schema_version": 1, "version": 1}), encoding="utf-8")
    try:
        CheckpointManager.load(path)
    except ValueError as error:
        assert "Unsupported checkpoint artifact type" in str(error)
    else:
        raise AssertionError("Non-checkpoint artifact was accepted")


def test_checkpoint_schema_version_is_validated(tmp_path: Path):
    path = tmp_path / "new-schema.json"
    path.write_text(json.dumps({"artifact_type": "checkpoint", "schema_version": 2, "version": 1}), encoding="utf-8")
    try:
        CheckpointManager.load(path)
    except ValueError as error:
        assert "Unsupported checkpoint schema version" in str(error)
    else:
        raise AssertionError("Unsupported checkpoint schema was accepted")
