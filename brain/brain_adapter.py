from .brain_interface import BrainInterface
from core.action import Action
from core.observation import Observation


class BrainAdapter:
    """Optional bridge for callback-based brains while preserving the public contract."""

    def __init__(self, brain: BrainInterface):
        self.brain = brain

    def reset(self, context: dict | None = None) -> None: self.brain.reset(context)
    def observe(self, observation: Observation) -> None: self.brain.observe(observation)
    def decide(self) -> None: self.brain.decide()
    def get_action(self) -> Action: return self.brain.get_action()
    def learn(self, reward: float, done: bool, info: dict | None = None) -> None: self.brain.learn(reward, done, info)
    def shutdown(self) -> None: self.brain.shutdown()
