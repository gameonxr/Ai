from __future__ import annotations

from abc import ABC, abstractmethod

from core.action import Action
from core.observation import Observation


class RobotAdapter(ABC):
    """Transport-neutral robot contract; live hardware is opt-in and not implemented here."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_observation(self) -> Observation: ...

    @abstractmethod
    def send_action(self, action: Action) -> None: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...
