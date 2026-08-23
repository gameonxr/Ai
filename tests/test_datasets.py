from pathlib import Path

import pytest

from brain import DummyBrain
from datasets import TrajectoryDatasetWriter, load_dataset
from simulator import Simulator


def test_trajectory_dataset_round_trip(tmp_path: Path):
    path = tmp_path / "trajectory.jsonl"
    simulator = Simulator("config/simulator_config.yaml")
    simulator.set_brain(DummyBrain())
    simulator.reset(seed=4)
    with TrajectoryDatasetWriter(path, {"seed": 4}) as writer:
        assert not path.exists()
        observation = simulator.step()
        writer.append(observation, simulator.brain.get_action(), reward=1.5, terminated=True, info={"index": 1})
    dataset = load_dataset(path)
    assert dataset.metadata == {"seed": 4}
    assert dataset.size == 1
    assert dataset.transitions[0]["reward"] == 1.5
    simulator.shutdown()


def test_trajectory_writer_preserves_existing_file_on_failure(tmp_path: Path):
    path = tmp_path / "trajectory.jsonl"
    path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with TrajectoryDatasetWriter(path, {"seed": 9}):
            raise RuntimeError("simulated write failure")
    assert path.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".trajectory.jsonl.*.tmp"))


def test_trajectory_header_must_be_an_object(tmp_path: Path):
    path = tmp_path / "list-header.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Trajectory dataset header must be a JSON object"):
        load_dataset(path)


def test_trajectory_metadata_must_be_an_object(tmp_path: Path):
    path = tmp_path / "bad-metadata.jsonl"
    path.write_text('{"type":"metadata","schema_version":1,"metadata":[] }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Trajectory dataset metadata must be a JSON object"):
        load_dataset(path)


def test_trajectory_transition_must_be_an_object(tmp_path: Path):
    path = tmp_path / "list-transition.jsonl"
    path.write_text('{"type":"metadata","schema_version":1}\n[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Trajectory transition must be a JSON object"):
        load_dataset(path)


def test_trajectory_schema_is_validated(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type":"metadata","schema_version":99}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)
