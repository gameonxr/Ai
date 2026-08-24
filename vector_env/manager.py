from __future__ import annotations

from brain import BrainInterface
from core import Action, Observation
from simulator import Simulator


class _ActionQueueBrain(BrainInterface):
    def __init__(self):
        self.pending = Action.noop()

    def reset(self, context=None):
        self.pending = Action.noop()

    def observe(self, observation):
        return None

    def decide(self):
        return None

    def get_action(self):
        action, self.pending = self.pending, Action.noop()
        return action


class VectorizedSimulator:
    """Manage independent simulator instances through batched reset and step calls."""

    def __init__(self, config_path: str = "config/simulator_config.yaml", num_envs: int = 1):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.envs: list[Simulator] = []
        try:
            self.envs = [Simulator(config_path) for _ in range(num_envs)]
            self._brains = [_ActionQueueBrain() for _ in self.envs]
            for simulator, brain in zip(self.envs, self._brains):
                simulator.set_brain(brain)
        except Exception:
            for simulator in self.envs:
                simulator.shutdown()
            raise
        self.num_envs = num_envs
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VectorizedSimulator is closed")

    def reset(self, seed: int | None = None) -> list[Observation]:
        self._ensure_open()
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        observations = []
        for index, simulator in enumerate(self.envs):
            simulator.reset(None if seed is None else seed + index)
            observations.append(simulator.step())
        return observations

    def step(self, actions: list[Action] | None = None) -> list[Observation]:
        self._ensure_open()
        if actions is not None:
            if not isinstance(actions, (list, tuple)):
                raise ValueError("actions must be a list or tuple")
            if len(actions) != self.num_envs:
                raise ValueError("actions length must match num_envs")
            for brain, action in zip(self._brains, actions):
                if not isinstance(action, Action):
                    raise TypeError("all vectorized actions must be Action objects")
                brain.pending = action
        return [simulator.step() for simulator in self.envs]

    def get_states(self) -> list[dict]:
        self._ensure_open()
        return [simulator.get_state() for simulator in self.envs]

    def close(self) -> None:
        if self._closed:
            return
        for simulator in self.envs:
            simulator.shutdown()
        self._closed = True
