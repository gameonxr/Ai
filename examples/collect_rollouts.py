import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import Simulator
from training import RandomTorquePolicy, Trainer


if __name__ == "__main__":
    simulator = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(list(simulator.actuators), scale=0.05, seed=42)
    trainer = Trainer(simulator, policy, reward_fn=lambda observation, action: 1.0)
    rollout, metrics = trainer.run_episode(max_steps=25, seed=42)
    rollout.save_jsonl("/tmp/ai_body_rollout.jsonl")
    print(f"Collected {metrics.steps} steps with total reward {metrics.total_reward:.1f}")
    trainer.close()
