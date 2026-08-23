from __future__ import annotations

from brain.brain_interface import BrainInterface
from core.action import Action
from core.observation import Observation
from .policy import Policy


class PolicyBrain(BrainInterface):
    """Adapts a policy to the simulator's stable brain contract."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.last_observation: Observation | None = None
        self.last_action = Action.noop()

    def reset(self, context: dict | None = None) -> None:
        self.last_observation = None
        self.last_action = Action.noop()
        self.policy.reset(None if context is None else context.get("seed"))

    def observe(self, observation: Observation) -> None:
        self.last_observation = observation

    def decide(self) -> None:
        if self.last_observation is None:
            self.last_action = Action.noop()
        else:
            self.last_action = self.policy.act(self.last_observation)

    def get_action(self) -> Action:
        return self.last_action
