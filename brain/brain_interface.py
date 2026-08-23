from abc import ABC, abstractmethod
from core.action import Action
from core.observation import Observation


class BrainInterface(ABC):
    """Stable brain contract. Implementations never access physics or body internals."""

    @abstractmethod
    def reset(self, context: dict | None = None) -> None: ...

    @abstractmethod
    def observe(self, observation: Observation) -> None: ...

    @abstractmethod
    def decide(self) -> None: ...

    @abstractmethod
    def get_action(self) -> Action: ...

    def learn(self, reward: float, done: bool, info: dict | None = None) -> None:
        return None

    def shutdown(self) -> None:
        return None
