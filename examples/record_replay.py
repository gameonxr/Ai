import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recording import EpisodeRecorder, ReplayBrain
from simulator import Simulator
from training import RandomTorquePolicy


if __name__ == "__main__":
    source = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(list(source.actuators), seed=42)
    source.set_brain(__import__("training", fromlist=["PolicyBrain"]).PolicyBrain(policy))
    source.reset(seed=42)
    recorder = EpisodeRecorder({"seed": 42, "steps": 5})
    for _ in range(5):
        observation = source.step()
        recorder.record(observation, source.brain.last_action)
    path = Path("/tmp/ai_body_episode.jsonl")
    recorder.save_jsonl(path)
    source.shutdown()

    loaded = EpisodeRecorder.load_jsonl(path)
    replay = Simulator("config/simulator_config.yaml")
    replay.set_brain(ReplayBrain(loaded.transitions))
    replay.reset(seed=42)
    for _ in loaded.transitions:
        replay.step()
    print(f"Recorded and replayed {len(loaded)} transitions from {path}")
    replay.shutdown()
