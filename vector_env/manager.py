from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, config_path: str = "config/simulator_config.yaml", num_envs: int = 1, parallel: bool = False):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(parallel, bool):
            raise ValueError("parallel must be a boolean")
        self.envs: list[Simulator] = []
        self._executor: ThreadPoolExecutor | None = None
        try:
            self.envs = [Simulator(config_path) for _ in range(num_envs)]
            self._brains = [_ActionQueueBrain() for _ in self.envs]
            for simulator, brain in zip(self.envs, self._brains):
                simulator.set_brain(brain)
            if parallel and num_envs > 1:
                self._executor = ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="ai-sim")
        except Exception:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            for simulator in self.envs:
                simulator.shutdown()
            raise
        self.num_envs = num_envs
        self.parallel = parallel
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VectorizedSimulator is closed")

    def reset(self, seed: int | None = None) -> list[Observation]:
        self._ensure_open()
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        reset_inputs = [(simulator, None if seed is None else seed + index) for index, simulator in enumerate(self.envs)]
        if self._executor is None:
            return [self._reset_one(item) for item in reset_inputs]
        return list(self._executor.map(self._reset_one, reset_inputs))

    @staticmethod
    def _reset_one(item: tuple[Simulator, int | None]) -> Observation:
        simulator, seed = item
        simulator.reset(seed)
        return simulator.step()

    @staticmethod
    def _step_one(simulator: Simulator) -> Observation:
        return simulator.step()

    def step(self, actions: list[Action] | None = None) -> list[Observation]:
        self._ensure_open()
        if actions is not None:
            if not isinstance(actions, (list, tuple)):
                raise ValueError("actions must be a list or tuple")
            if len(actions) != self.num_envs:
                raise ValueError("actions length must match num_envs")
            if any(not isinstance(action, Action) for action in actions):
                raise TypeError("all vectorized actions must be Action objects")
            for brain, action in zip(self._brains, actions):
                brain.pending = action
        if self._executor is None:
            return [self._step_one(simulator) for simulator in self.envs]
        return list(self._executor.map(self._step_one, self.envs))

    def get_states(self) -> list[dict]:
        self._ensure_open()
        return [simulator.get_state() for simulator in self.envs]

    def close(self) -> None:
        if self._closed:
            return
        for simulator in self.envs:
            simulator.shutdown()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True
