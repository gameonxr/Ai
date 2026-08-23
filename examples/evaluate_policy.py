import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from benchmarks import run_benchmark
from evaluation import Evaluator
from simulator import Simulator
from training import RandomTorquePolicy


if __name__ == "__main__":
    config = yaml.safe_load(Path("config/evaluation_config.yaml").read_text(encoding="utf-8"))["evaluation"]
    simulator = Simulator(config["simulator_config"])
    policy = RandomTorquePolicy(list(simulator.actuators), seed=config["seed"])
    summary = Evaluator(simulator, policy, reward_fn=lambda observation, action: 1.0).run(config["episodes"], config["max_steps"], config["seed"])
    print(f"Evaluated {summary.episodes} episodes: mean reward {summary.mean_reward:.2f}, mean steps {summary.mean_steps:.1f}")
    simulator.shutdown()
