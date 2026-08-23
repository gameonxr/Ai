from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Callable, Any

from simulator import Simulator
from training import Policy, PolicyBrain, EpisodeMetrics


@dataclass
class EvaluationSummary:
    episodes: int
    total_steps: int
    mean_reward: float
    reward_std: float
    min_reward: float
    max_reward: float
    mean_steps: float
    episode_metrics: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator:
    """Evaluate one policy over seeded episodes without changing policy state."""

    def __init__(self, simulator: Simulator, policy: Policy, reward_fn: Callable | None = None):
        self.simulator = simulator
        self.policy = policy
        self.reward_fn = reward_fn or (lambda observation, action: 0.0)

    def run(self, episodes: int = 1, max_steps: int = 100, seed: int | None = 42) -> EvaluationSummary:
        if episodes < 1 or max_steps < 1:
            raise ValueError("episodes and max_steps must be >= 1")
        brain = PolicyBrain(self.policy)
        self.simulator.set_brain(brain)
        results = []
        try:
            for index in range(episodes):
                self.simulator.reset(None if seed is None else seed + index)
                total_reward = 0.0
                steps = 0
                for _ in range(max_steps):
                    observation = self.simulator.step()
                    action = brain.last_action
                    total_reward += float(self.reward_fn(observation, action))
                    steps += 1
                results.append(EpisodeMetrics(episode=index, steps=steps, total_reward=total_reward, terminated=False))
        finally:
            self.simulator.set_brain(None)
        rewards = [item.total_reward for item in results]
        step_counts = [item.steps for item in results]
        return EvaluationSummary(episodes, sum(step_counts), mean(rewards), pstdev(rewards) if len(rewards) > 1 else 0.0, min(rewards), max(rewards), mean(step_counts), [asdict(item) for item in results])
