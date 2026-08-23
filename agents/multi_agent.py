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

    def reset(self, seed: int | None = None) -> dict[str, Observation]:
        observations = {}
        for offset, (agent_id, environment) in enumerate(self.environments.items()):
            observations[agent_id] = environment.reset(None if seed is None else seed + offset)
        self.last_steps.clear()
        return observations

    def step(self) -> dict[str, AgentStep]:
        steps = {}
        for agent_id, environment in self.environments.items():
            observation, action = environment.step()
            steps[agent_id] = AgentStep(observation, action)
        self.last_steps = steps
        return steps

    def get_state(self) -> dict[str, Any]:
        return {agent_id: environment.simulator.get_state() for agent_id, environment in self.environments.items()}

    def close(self) -> None:
        for environment in self.environments.values():
            environment.close()
