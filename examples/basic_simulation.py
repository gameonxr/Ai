import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import Simulator


if __name__ == "__main__":
    sim = Simulator("config/simulator_config.yaml")
    sim.reset(seed=42)
    for _ in range(10):
        observation = sim.step()
        print(observation.timestamp, observation.proprioception["body_position"])
    sim.shutdown()
