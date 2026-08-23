import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import DummyBrain
from simulator import Simulator


if __name__ == "__main__":
    checkpoint_path = Path("/tmp/ai_body_checkpoint.json")
    simulator = Simulator("config/simulator_config.yaml")
    simulator.set_brain(DummyBrain())
    simulator.reset(seed=42)
    simulator.step(5)
    simulator.save_checkpoint(checkpoint_path, run_id="demo", metadata={"purpose": "resume-demo"})
    saved_step = simulator.step_count
    simulator.step(2)
    simulator.restore_checkpoint(checkpoint_path)
    print(f"Restored step {simulator.step_count} (saved at {saved_step})")
    simulator.shutdown()
