from abc import ABC, abstractmethod
from core.action import Action


class PhysicsEngine(ABC):
    """Backend abstraction; only the simulator talks to this interface."""

    @abstractmethod
    def load_body(self, body_definition: dict) -> None: ...
    @abstractmethod
    def reset(self, seed: int | None = None) -> None: ...
    @abstractmethod
    def step(self, dt: float = 0.005) -> None: ...
    @abstractmethod
    def apply_action(self, action: Action) -> None: ...
    @abstractmethod
    def get_body_state(self) -> dict: ...
    @abstractmethod
    def get_contact_info(self) -> list: ...

    def get_checkpoint_state(self) -> dict:
        raise NotImplementedError("This backend does not support checkpointing")

    def restore_checkpoint_state(self, state: dict) -> None:
        raise NotImplementedError("This backend does not support checkpointing")
    @abstractmethod
    def set_gravity(self, gravity: list[float] | tuple[float, float, float]) -> None: ...
    @abstractmethod
    def shutdown(self) -> None: ...
