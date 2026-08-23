from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from artifact_io import write_json_atomic


@dataclass
class Checkpoint:
    """Versioned, JSON-serializable simulator checkpoint."""
    version: int
    run_id: str
    episode: int
    step: int
    current_time: float
    simulator_state: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_type: str = "checkpoint"
    schema_version: int = 1


class CheckpointManager:
    CURRENT_VERSION = 1

    @classmethod
    def save(cls, simulator, path: str | Path, run_id: str = "default", metadata: dict | None = None) -> Checkpoint:
        state = {"physics": simulator.physics.get_checkpoint_state(), "paused": simulator.paused, "actuators": {name: actuator.current_command for name, actuator in simulator.actuators.items()}}
        checkpoint = Checkpoint(cls.CURRENT_VERSION, run_id, simulator.metrics.episodes, simulator.step_count, simulator.current_time, state, simulator.metrics.snapshot(), metadata or {})
        write_json_atomic(asdict(checkpoint), path)
        return checkpoint

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload must be a JSON object")
        artifact_type = payload.get("artifact_type", "checkpoint")
        if artifact_type != "checkpoint":
            raise ValueError(f"Unsupported checkpoint artifact type: {artifact_type}")
        schema_version = payload.get("schema_version", 1)
        try:
            schema_version = int(schema_version)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported checkpoint schema version: {schema_version}") from error
        if schema_version != 1:
            raise ValueError(f"Unsupported checkpoint schema version: {schema_version}")
        checkpoint_version = payload.get("version", -1)
        try:
            checkpoint_version = int(checkpoint_version)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported checkpoint version: {checkpoint_version}") from error
        if checkpoint_version != cls.CURRENT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {checkpoint_version}")
        required_fields = ("run_id", "episode", "step", "current_time", "simulator_state", "metrics")
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            raise ValueError(f"Checkpoint missing required fields: {', '.join(missing_fields)}")
        if not isinstance(payload["simulator_state"], dict) or not isinstance(payload["metrics"], dict):
            raise ValueError("Checkpoint simulator_state and metrics must be JSON objects")
        return Checkpoint(checkpoint_version, payload["run_id"], int(payload["episode"]), int(payload["step"]), float(payload["current_time"]), payload["simulator_state"], payload["metrics"], payload.get("metadata", {}), artifact_type, schema_version)

    @classmethod
    def restore(cls, simulator, path: str | Path) -> Checkpoint:
        checkpoint = cls.load(path)
        simulator.physics.restore_checkpoint_state(checkpoint.simulator_state["physics"])
        simulator.current_time = checkpoint.current_time
        simulator.step_count = checkpoint.step
        simulator.paused = bool(checkpoint.simulator_state.get("paused", False))
        for name, value in checkpoint.simulator_state.get("actuators", {}).items():
            if name in simulator.actuators:
                simulator.actuators[name].current_command = float(value)
        return checkpoint
