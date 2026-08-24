from pathlib import Path

import pytest

from core import Action, Observation
from recording import EpisodeRecorder, ReplayBrain


@pytest.mark.parametrize("metadata", [[], "invalid"])
def test_recording_rejects_non_object_metadata(metadata):
    with pytest.raises(ValueError, match="Episode recording metadata must be a JSON object"):
        EpisodeRecorder(metadata)


def test_recording_rejects_non_mapping_info():
    recorder = EpisodeRecorder()
    with pytest.raises(ValueError, match="info must be a mapping"):
        recorder.record(None, None, info=["not", "a", "mapping"])


def test_recording_round_trip(tmp_path: Path):
    recorder = EpisodeRecorder({"seed": 3})
    observation = Observation(0.0, proprioception={"joint_positions": {"neck": 0.1}})
    action = Action(joint_targets={"neck": 0.1}, timestamp=0.0)
    recorder.record(observation, action, reward=1.5, done=True)
    output = tmp_path / "episode.jsonl"
    recorder.save_jsonl(output)
    loaded = EpisodeRecorder.load_jsonl(output)
    assert loaded.metadata == {"seed": 3}
    assert len(loaded) == 1
    assert loaded.transitions[0].reward == 1.5


def test_recording_invalid_json_is_contextual(tmp_path: Path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("{invalid}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid episode recording JSON at line 1"):
        EpisodeRecorder.load_jsonl(path)


def test_recording_metadata_and_transition_shapes_are_validated(tmp_path: Path):
    metadata_path = tmp_path / "bad-metadata.jsonl"
    metadata_path.write_text('{"type":"metadata","metadata":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Episode recording metadata must be a JSON object"):
        EpisodeRecorder.load_jsonl(metadata_path)

    transition_path = tmp_path / "bad-transition.jsonl"
    transition_path.write_text('{"type":"metadata"}\n{"type":"transition"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="transition missing required fields"):
        EpisodeRecorder.load_jsonl(transition_path)


def test_recording_non_object_line_is_rejected(tmp_path: Path):
    path = tmp_path / "list.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 must be a JSON object"):
        EpisodeRecorder.load_jsonl(path)


def test_replay_brain_reproduces_actions():
    recorder = EpisodeRecorder()
    recorder.record(Observation(0.0), Action(joint_targets={"neck": 0.2}))
    brain = ReplayBrain(recorder.transitions)
    brain.reset()
    brain.observe(Observation(0.0))
    brain.decide()
    assert brain.get_action().joint_targets == {"neck": 0.2}
    brain.decide()
    assert brain.get_action().metadata["noop"]


def test_replay_brain_rejects_non_object_action_payload():
    brain = ReplayBrain([type("Transition", (), {"action": []})()])
    brain.reset()
    with pytest.raises(ValueError, match="Replay action payload must be a JSON object"):
        brain.decide()
