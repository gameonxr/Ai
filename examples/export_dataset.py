import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import DummyBrain
from datasets import TrajectoryDatasetWriter, load_dataset
from simulator import Simulator


if __name__ == "__main__":
    path = Path("/tmp/ai_body_trajectory.jsonl")
    simulator = Simulator("config/simulator_config.yaml")
    simulator.set_brain(DummyBrain())
    simulator.reset(seed=42)
    with TrajectoryDatasetWriter(path, {"source": "dummy-brain", "seed": 42}) as writer:
        for step in range(5):
            observation = simulator.step()
            writer.append(observation, simulator.brain.get_action(), reward=1.0, info={"step": step + 1})
    simulator.shutdown()
    print(f"Exported {load_dataset(path).size} transitions to {path}")
