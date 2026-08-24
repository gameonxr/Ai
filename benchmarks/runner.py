from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Any

from simulator import Simulator


@dataclass
class BenchmarkResult:
    steps: int
    timestep: float
    simulation_seconds: float
    wall_seconds: float
    realtime_factor: float
    final_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_artifact_dict(self, config_path: str, seed: int | None) -> dict[str, Any]:
        """Return a persisted benchmark payload with run provenance."""
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("config_path must be a non-empty string")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        for field in ("timestep", "simulation_seconds", "wall_seconds", "realtime_factor"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{field} must be a finite non-negative number")
        if self.timestep <= 0:
            raise ValueError("timestep must be positive")
        if not isinstance(self.final_state, dict):
            raise ValueError("final_state must be an object")
        return {
            "artifact_type": "benchmark",
            "schema_version": 1,
            "config_path": config_path,
            "seed": seed,
            **self.to_dict(),
        }


def run_benchmark(config_path: str = "config/simulator_config.yaml", steps: int = 1000, seed: int | None = 42) -> BenchmarkResult:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or null")
    simulator = Simulator(config_path)
    try:
        simulator.reset(seed)
        started = perf_counter()
        simulator.step(steps)
        wall_seconds = perf_counter() - started
        simulation_seconds = steps * simulator.timestep
        return BenchmarkResult(steps, simulator.timestep, simulation_seconds, wall_seconds, simulation_seconds / wall_seconds if wall_seconds else 0.0, simulator.get_state())
    finally:
        simulator.shutdown()
