from __future__ import annotations

from dataclasses import asdict, dataclass
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
        return {
            "artifact_type": "benchmark",
            "config_path": config_path,
            "seed": seed,
            **self.to_dict(),
        }


def run_benchmark(config_path: str = "config/simulator_config.yaml", steps: int = 1000, seed: int | None = 42) -> BenchmarkResult:
    if steps < 1:
        raise ValueError("steps must be >= 1")
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
