from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.action import Action
from core.observation import Observation
from simulator import Simulator
from training.environment import SimulationEnvironment
from training.policy import Policy


@dataclass
class AgentStep:
    """One synchronized step of a named agent."""
    observation: Observation
    action: Action


class MultiAgentCoordinator:
    """Coordinate independent simulator instances without changing single-agent APIs."""

    def __init__(self, agents: dict[str, tuple[Simulator, Policy]]):
        if not agents:
            raise ValueError("At least one agent is required")
        if len(set(agents)) != len(agents):
            raise ValueError("Agent IDs must be unique")
        self.environments = {agent_id: SimulationEnvironment(simulator, policy) for agent_id, (simulator, policy) in agents.items()}
        self.last_steps: dict[str, AgentStep] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MultiAgentCoordinator is closed")

    def reset(self, seed: int | None = None) -> dict[str, Observation]:
        self._ensure_open()
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        observations = {}
        for offset, (agent_id, environment) in enumerate(self.environments.items()):
            observations[agent_id] = environment.reset(None if seed is None else seed + offset)
        self.last_steps.clear()
        return observations

    def step(self) -> dict[str, AgentStep]:
        self._ensure_open()
        steps = {}
        for agent_id, environment in self.environments.items():
            observation, action = environment.step()
            steps[agent_id] = AgentStep(observation, action)
        self.last_steps = steps
        return steps

    def get_state(self) -> dict[str, Any]:
        self._ensure_open()
        return {agent_id: environment.simulator.get_state() for agent_id, environment in self.environments.items()}

    def close(self) -> None:
        if self._closed:
            return
        for environment in self.environments.values():
            environment.close()
        self._closed = True
