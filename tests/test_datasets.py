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
        observation = simulator.step()
        writer.append(observation, simulator.brain.get_action(), reward=1.5, terminated=True, info={"index": 1})
    dataset = load_dataset(path)
    assert dataset.metadata == {"seed": 4}
    assert dataset.size == 1
    assert dataset.transitions[0]["reward"] == 1.5
    simulator.shutdown()


def test_trajectory_schema_is_validated(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type":"metadata","schema_version":99}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)
