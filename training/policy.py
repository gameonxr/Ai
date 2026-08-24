from __future__ import annotations

from abc import ABC, abstractmethod
import math
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
        if not isinstance(joint_names, (list, tuple)) or not joint_names or any(not isinstance(name, str) or not name.strip() for name in joint_names):
            raise ValueError("joint_names must be a non-empty list of strings")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must be unique")
        if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer, np.floating)) or not math.isfinite(float(scale)) or float(scale) < 0:
            raise ValueError("scale must be a finite non-negative number")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        self.joint_names = list(joint_names)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: Observation) -> Action:
        values = {name: float(self.rng.uniform(-self.scale, self.scale)) for name in self.joint_names}
        return Action(motor_commands=values, timestamp=observation.timestamp)
