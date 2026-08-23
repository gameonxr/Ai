from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class SimulationMetrics:
    """Counters for steps, actions, invalid actions, and wall-clock performance."""
    steps: int = 0
    actions: int = 0
    invalid_actions: int = 0
    episodes: int = 0
    simulation_seconds: float = 0.0
    started_at: float = field(default_factory=monotonic)

    def reset_episode(self) -> None:
        self.episodes += 1
        self.simulation_seconds = 0.0

    def record_step(self, timestep: float, action_applied: bool = False, invalid_action: bool = False) -> None:
        self.steps += 1
        self.simulation_seconds += float(timestep)
        if action_applied:
            self.actions += 1
        if invalid_action:
            self.invalid_actions += 1

    @property
    def wall_seconds(self) -> float:
        return monotonic() - self.started_at

    @property
    def realtime_factor(self) -> float:
        wall = self.wall_seconds
        return self.simulation_seconds / wall if wall > 0 else 0.0

    def snapshot(self) -> dict:
        return {"steps": self.steps, "actions": self.actions, "invalid_actions": self.invalid_actions, "episodes": self.episodes, "simulation_seconds": self.simulation_seconds, "wall_seconds": self.wall_seconds, "realtime_factor": self.realtime_factor}
