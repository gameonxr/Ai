import json
from pathlib import Path

from simulator import Simulator
from training import RandomTorquePolicy, Rollout, Trainer


def test_rollout_jsonl_persistence(tmp_path: Path):
    sim = Simulator("config/simulator_config.yaml")
    policy = RandomTorquePolicy(list(sim.actuators), seed=1)
    trainer = Trainer(sim, policy, reward_fn=lambda observation, action: 2.0)
    rollout, metrics = trainer.run_episode(max_steps=4, seed=1)
    output = tmp_path / "rollout.jsonl"
    rollout.save_jsonl(output)
    assert metrics.steps == 4 and metrics.total_reward == 8.0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["reward"] == 2.0
    trainer.close()


def test_trainer_seeded_policy_is_reproducible():
    def collect():
        sim = Simulator("config/simulator_config.yaml")
        trainer = Trainer(sim, RandomTorquePolicy(list(sim.actuators), seed=9), reward_fn=lambda observation, action: 1.0)
        rollout, _ = trainer.run_episode(max_steps=3, seed=9)
        trainer.close()
        return [item.action for item in rollout.transitions]

    assert collect() == collect()
