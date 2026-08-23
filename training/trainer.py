from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.action import Action
from core.observation import Observation
from simulator import Simulator
from .environment import SimulationEnvironment
from .policy import Policy
from .rollout import Rollout


RewardFn = Callable[[Observation, Action], float]
DoneFn = Callable[[Observation, int], bool]


@dataclass
class EpisodeMetrics:
    episode: int
    steps: int
    total_reward: float
    terminated: bool


class Trainer:
    """Collect policy rollouts without coupling training code to physics internals."""

    def __init__(self, simulator: Simulator, policy: Policy, reward_fn: RewardFn | None = None, done_fn: DoneFn | None = None):
        self.environment = SimulationEnvironment(simulator, policy)
        self.reward_fn = reward_fn or (lambda observation, action: 0.0)
        self.done_fn = done_fn or (lambda observation, step: False)

    def run_episode(self, max_steps: int = 1000, seed: int | None = None) -> tuple[Rollout, EpisodeMetrics]:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        rollout = Rollout()
        observation = self.environment.reset(seed)
        terminated = False
        for step in range(max_steps):
            observation, action = self.environment.step()
            reward = self.reward_fn(observation, action)
            terminated = bool(self.done_fn(observation, step + 1))
            rollout.append(observation, action, reward, terminated, {"step": step + 1})
            if terminated:
                break
        metrics = EpisodeMetrics(0, len(rollout.transitions), rollout.total_reward, terminated)
        return rollout, metrics

    def train(self, episodes: int = 1, max_steps: int = 1000, seed: int | None = None) -> list[EpisodeMetrics]:
        if episodes < 1:
            raise ValueError("episodes must be >= 1")
        metrics = []
        for episode in range(episodes):
            _, result = self.run_episode(max_steps, None if seed is None else seed + episode)
            metrics.append(EpisodeMetrics(episode, result.steps, result.total_reward, result.terminated))
        return metrics

    def close(self) -> None:
        self.environment.close()
