from core.action import Action
from core.observation import Observation
from .brain_interface import BrainInterface


class DummyBrain(BrainInterface):
    """Minimal deterministic brain useful for examples and integration tests."""

    def __init__(self, hold_position: bool = True):
        self.hold_position = hold_position
        self.last_observation: Observation | None = None
        self._action = Action.noop()

    def reset(self, context: dict | None = None) -> None:
        self.last_observation = None
        self._action = Action.noop()

    def observe(self, observation: Observation) -> None:
        self.last_observation = observation

    def decide(self) -> None:
        if self.hold_position and self.last_observation and self.last_observation.proprioception:
            positions = self.last_observation.proprioception.get("joint_positions", {})
            self._action = Action(joint_targets=dict(positions), timestamp=self.last_observation.timestamp)
        else:
            self._action = Action.noop()

    def get_action(self) -> Action:
        return self._action
