from pathlib import Path

from core import Action, Observation
from recording import EpisodeRecorder, ReplayBrain


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
