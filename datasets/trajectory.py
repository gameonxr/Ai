from __future__ import annotations

from dataclasses import dataclass, field
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
    lines = Path(path).read_text(encoding="utf-8").splitlines()
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
        transitions.append(record)
    return TrajectoryDataset(metadata, transitions)
