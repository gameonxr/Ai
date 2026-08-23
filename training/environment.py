from __future__ import annotations

from brain import BrainInterface
from core.action import Action
from core.observation import Observation
from simulator import Simulator
from .policy import Policy
from .policy_brain import PolicyBrain


class SimulationEnvironment:
    """Minimal training adapter that communicates only via brain/action APIs."""

    def __init__(self, simulator: Simulator, policy: Policy):
        self.simulator = simulator
        self.policy_brain = PolicyBrain(policy)
        self.simulator.set_brain(self.policy_brain)
        self.last_observation: Observation | None = None
        self.last_action: Action = Action.noop()

    def reset(self, seed: int | None = None) -> Observation:
        self.simulator.reset(seed)
        observation = self.simulator.step()
        self.last_observation = observation
        self.last_action = self.policy_brain.last_action
        return observation

    def step(self) -> tuple[Observation, Action]:
        observation = self.simulator.step()
        self.last_observation = observation
        self.last_action = self.policy_brain.last_action
        return observation, self.last_action

    def close(self) -> None:
        self.simulator.shutdown()
