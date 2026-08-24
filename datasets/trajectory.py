from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable


@dataclass
class TrajectoryDataset:
    metadata: dict[str, Any]
    transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.transitions)


class TrajectoryDatasetWriter:
    """Write versioned observation/action/reward trajectories as JSONL."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, metadata: dict[str, Any] | None = None):
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Trajectory dataset metadata must be a JSON object")
        self.path = Path(path)
        self.metadata = metadata or {}
        self._handle = None
        self._temporary_path: Path | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False)
        self._handle = temporary
        self._temporary_path = Path(temporary.name)
        self._write({"type": "metadata", "schema_version": self.SCHEMA_VERSION, "metadata": self.metadata})
        return self

    def append(self, observation, action, reward: float = 0.0, terminated: bool = False, info: dict[str, Any] | None = None) -> None:
        if self._handle is None:
            raise RuntimeError("TrajectoryDatasetWriter must be used as a context manager")
        if info is not None and not isinstance(info, dict):
            raise ValueError("info must be a mapping when provided")
        self._write({"type": "transition", "observation": observation.to_dict() if hasattr(observation, "to_dict") else observation, "action": action.to_dict() if hasattr(action, "to_dict") else action, "reward": float(reward), "terminated": bool(terminated), "info": info or {}})

    def _write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        handle = self._handle
        temporary_path = self._temporary_path
        self._handle = None
        self._temporary_path = None
        if handle is None or temporary_path is None:
            return
        handle.close()
        try:
            if exc_type is None:
                os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)


def load_dataset(path: str | Path) -> TrajectoryDataset:
    dataset_path = Path(path)
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read trajectory dataset {dataset_path}: {error}") from error
    if not lines:
        raise ValueError("Trajectory dataset is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid trajectory dataset JSON at line 1: {error.msg}") from error
    if not isinstance(header, dict):
        raise ValueError("Trajectory dataset header must be a JSON object")
    if header.get("type") != "metadata":
        raise ValueError("Unsupported trajectory dataset schema")
    schema_version = header.get("schema_version", -1)
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported trajectory dataset schema version: {schema_version}") from error
    if schema_version != TrajectoryDatasetWriter.SCHEMA_VERSION:
        raise ValueError(f"Unsupported trajectory dataset schema version: {schema_version}")
    metadata = header.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Trajectory dataset metadata must be a JSON object")
    transitions = []
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid trajectory dataset JSON at line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError("Trajectory transition must be a JSON object")
        if record.get("type") != "transition":
            raise ValueError("Unexpected trajectory record type")
        required_fields = ("observation", "action", "reward", "terminated")
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            raise ValueError(f"Trajectory transition missing required fields: {', '.join(missing_fields)}")
        if not isinstance(record["observation"], dict):
            raise ValueError(f"Trajectory transition observation at line {line_number} must be a JSON object")
        if not isinstance(record["action"], dict):
            raise ValueError(f"Trajectory transition action at line {line_number} must be a JSON object")
        try:
            reward = float(record["reward"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Trajectory transition reward at line {line_number} must be numeric") from error
        if not math.isfinite(reward):
            raise ValueError(f"Trajectory transition reward at line {line_number} must be finite")
        if not isinstance(record["terminated"], bool):
            raise ValueError(f"Trajectory transition terminated at line {line_number} must be a boolean")
        info = record.get("info", {})
        if not isinstance(info, dict):
            raise ValueError(f"Trajectory transition info at line {line_number} must be a JSON object")
        transitions.append(record)
    return TrajectoryDataset(metadata, transitions)
