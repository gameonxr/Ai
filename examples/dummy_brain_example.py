import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import DummyBrain
from simulator import Simulator


if __name__ == "__main__":
    sim = Simulator("config/simulator_config.yaml")
    sim.set_brain(DummyBrain())
    sim.reset(seed=42)
    for _ in range(100):
        sim.step()
    print(f"Completed {sim.step_count} deterministic steps")
    sim.shutdown()
