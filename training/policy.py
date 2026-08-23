from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from core.action import Action
from core.observation import Observation


class Policy(ABC):
    """Brain-agnostic policy contract for future RL or scripted controllers."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        return None


class RandomTorquePolicy(Policy):
    """Deterministic seeded baseline policy for rollout and pipeline tests."""

    def __init__(self, joint_names: list[str], scale: float = 0.05, seed: int | None = None):
        self.joint_names = list(joint_names)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: Observation) -> Action:
        values = {name: float(self.rng.uniform(-self.scale, self.scale)) for name in self.joint_names}
        return Action(motor_commands=values, timestamp=observation.timestamp)
