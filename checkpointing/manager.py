from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
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

    def to_artifact_dict(self) -> dict[str, Any]:
        """Return a validated checkpoint payload for persistence."""
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version != 1:
            raise ValueError("Checkpoint version must be 1")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Checkpoint run_id must be a non-empty string")
        for field in ("episode", "step"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Checkpoint {field} must be a non-negative integer")
        if isinstance(self.current_time, bool) or not isinstance(self.current_time, (int, float)) or not math.isfinite(float(self.current_time)) or self.current_time < 0:
            raise ValueError("Checkpoint current_time must be a finite non-negative number")
        if not isinstance(self.simulator_state, dict) or not isinstance(self.metrics, dict):
            raise ValueError("Checkpoint simulator_state and metrics must be JSON objects")
        if "physics" not in self.simulator_state or not isinstance(self.simulator_state["physics"], dict):
            raise ValueError("Checkpoint physics state must be a JSON object")
        if "actuators" in self.simulator_state and not isinstance(self.simulator_state["actuators"], dict):
            raise ValueError("Checkpoint actuators must be a JSON object")
        if not isinstance(self.metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON object")
        if self.artifact_type != "checkpoint":
            raise ValueError("Checkpoint artifact_type must be checkpoint")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != 1:
            raise ValueError("Checkpoint schema_version must be 1")
        return asdict(self)


class CheckpointManager:
    CURRENT_VERSION = 1

    @classmethod
    def save(cls, simulator, path: str | Path, run_id: str = "default", metadata: dict | None = None) -> Checkpoint:
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON object")
        state = {"physics": simulator.physics.get_checkpoint_state(), "paused": simulator.paused, "actuators": {name: actuator.current_command for name, actuator in simulator.actuators.items()}}
        checkpoint = Checkpoint(cls.CURRENT_VERSION, run_id, simulator.metrics.episodes, simulator.step_count, simulator.current_time, state, simulator.metrics.snapshot(), metadata or {})
        write_json_atomic(checkpoint.to_artifact_dict(), path)
        return checkpoint

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        checkpoint_path = Path(path)
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"Unable to read checkpoint {checkpoint_path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid checkpoint JSON in {checkpoint_path}: {error.msg}") from error
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
        simulator_state = payload["simulator_state"]
        if "physics" not in simulator_state:
            raise ValueError("Checkpoint simulator_state missing physics")
        if not isinstance(simulator_state["physics"], dict):
            raise ValueError("Checkpoint physics state must be a JSON object")
        if "actuators" in simulator_state and not isinstance(simulator_state["actuators"], dict):
            raise ValueError("Checkpoint actuators must be a JSON object")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON object")
        episode = cls._nonnegative_int(payload["episode"], "episode")
        step = cls._nonnegative_int(payload["step"], "step")
        current_time = cls._nonnegative_float(payload["current_time"], "current_time")
        return Checkpoint(checkpoint_version, payload["run_id"], episode, step, current_time, simulator_state, payload["metrics"], metadata, artifact_type, schema_version)

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Checkpoint {name} must be a non-negative integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Checkpoint {name} must be a non-negative integer") from error
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ValueError(f"Checkpoint {name} must be a non-negative integer")
        return int(numeric)

    @staticmethod
    def _nonnegative_float(value: Any, name: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Checkpoint {name} must be a finite non-negative number") from error
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Checkpoint {name} must be a finite non-negative number")
        return numeric

    @classmethod
    def restore(cls, simulator, path: str | Path) -> Checkpoint:
        checkpoint = cls.load(path)
        simulator.physics.restore_checkpoint_state(checkpoint.simulator_state["physics"])
        simulator.current_time = checkpoint.current_time
        simulator.step_count = checkpoint.step
        simulator.metrics.restore_snapshot(checkpoint.metrics)
        simulator.paused = bool(checkpoint.simulator_state.get("paused", False))
        for name, value in checkpoint.simulator_state.get("actuators", {}).items():
            if name in simulator.actuators:
                simulator.actuators[name].current_command = float(value)
        return checkpoint
