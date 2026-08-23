import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import Action
from vector_env import VectorizedSimulator


if __name__ == "__main__":
    vector = VectorizedSimulator("config/simulator_config.yaml", num_envs=4)
    observations = vector.reset(seed=42)
    actions = [Action(joint_targets={"neck": 0.0}) for _ in observations]
    next_observations = vector.step(actions)
    print(f"Batch size: {len(next_observations)}, timestamps: {[item.timestamp for item in next_observations]}")
    vector.close()
