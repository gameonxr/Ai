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
