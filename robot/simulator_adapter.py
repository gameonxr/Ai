from __future__ import annotations

from brain import BrainInterface
from core.action import Action
from core.observation import Observation
from simulator import Simulator
from .adapter import RobotAdapter


class _QueuedActionBrain(BrainInterface):
    def __init__(self):
        self.pending = Action.noop()
        self.last_observation: Observation | None = None

    def reset(self, context=None):
        self.pending = Action.noop()
        self.last_observation = None

    def observe(self, observation):
        self.last_observation = observation

    def decide(self):
        return None

    def get_action(self):
        action, self.pending = self.pending, Action.noop()
        return action


class SimulatedRobotAdapter(RobotAdapter):
    """Run the robot adapter contract against Simulator without direct physics access."""

    def __init__(self, simulator: Simulator):
        self.simulator = simulator
        self.brain = _QueuedActionBrain()
        self._connected = False
        self._last_observation: Observation | None = None

    def connect(self) -> None:
        if not self._connected:
            self.simulator.set_brain(self.brain)
            self.simulator.reset()
            self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self.simulator.set_brain(None)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def read_observation(self) -> Observation:
        if not self.connected:
            raise RuntimeError("Robot adapter is not connected")
        self._last_observation = self.simulator.step()
        return self._last_observation

    def send_action(self, action: Action) -> None:
        if not self.connected:
            raise RuntimeError("Robot adapter is not connected")
        if not isinstance(action, Action):
            raise TypeError("send_action requires an Action")
        self.brain.pending = action

    def emergency_stop(self) -> None:
        self.brain.pending = Action.noop()
        self.simulator.pause()

    def resume_after_stop(self) -> None:
        if not self.connected:
            raise RuntimeError("Robot adapter is not connected")
        self.simulator.resume()
