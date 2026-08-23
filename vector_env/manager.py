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
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.envs = [Simulator(config_path) for _ in range(num_envs)]
        self._brains = [_ActionQueueBrain() for _ in self.envs]
        for simulator, brain in zip(self.envs, self._brains):
            simulator.set_brain(brain)
        self.num_envs = num_envs

    def reset(self, seed: int | None = None) -> list[Observation]:
        observations = []
        for index, simulator in enumerate(self.envs):
            simulator.reset(None if seed is None else seed + index)
            observations.append(simulator.step())
        return observations

    def step(self, actions: list[Action] | None = None) -> list[Observation]:
        if actions is not None:
            if len(actions) != self.num_envs:
                raise ValueError("actions length must match num_envs")
            for brain, action in zip(self._brains, actions):
                if not isinstance(action, Action):
                    raise TypeError("all vectorized actions must be Action objects")
                brain.pending = action
        return [simulator.step() for simulator in self.envs]

    def get_states(self) -> list[dict]:
        return [simulator.get_state() for simulator in self.envs]

    def close(self) -> None:
        for simulator in self.envs:
            simulator.shutdown()
