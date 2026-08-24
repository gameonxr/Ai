from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from artifact_io import write_text_atomic
from core.action import Action
from core.observation import Observation


@dataclass
class RecordedTransition:
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class EpisodeRecorder:
    """Capture standardized observations/actions without touching simulator internals."""

    def __init__(self, metadata: dict[str, Any] | None = None):
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Episode recording metadata must be a JSON object")
        self.metadata = metadata or {}
        self.transitions: list[RecordedTransition] = []

    def record(self, observation: Observation, action: Action, reward: float = 0.0, done: bool = False, info: dict | None = None) -> None:
        if info is not None and not isinstance(info, dict):
            raise ValueError("info must be a mapping when provided")
        self.transitions.append(RecordedTransition(observation.to_dict(), action.to_dict(), float(reward), bool(done), info or {}))

    def save_jsonl(self, path: str | Path) -> None:
        lines = [json.dumps({"type": "metadata", "metadata": self.metadata}, sort_keys=True)]
        lines.extend(json.dumps({"type": "transition", **asdict(transition)}, sort_keys=True) for transition in self.transitions)
        write_text_atomic("\n".join(lines) + "\n", path)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "EpisodeRecorder":
        recording_path = Path(path)
        recorder = cls()
        try:
            handle = recording_path.open(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"Unable to read episode recording {recording_path}: {error}") from error
        with handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid episode recording JSON at line {line_number}: {error.msg}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Episode recording line {line_number} must be a JSON object")
                if record.get("type") == "metadata":
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        raise ValueError("Episode recording metadata must be a JSON object")
                    recorder.metadata = metadata
                elif record.get("type") == "transition":
                    required_fields = ("observation", "action", "reward", "done")
                    missing_fields = [field for field in required_fields if field not in record]
                    if missing_fields:
                        raise ValueError(f"Episode recording transition missing required fields: {', '.join(missing_fields)}")
                    if not isinstance(record["observation"], dict):
                        raise ValueError(f"Episode recording transition observation at line {line_number} must be a JSON object")
                    if not isinstance(record["action"], dict):
                        raise ValueError(f"Episode recording transition action at line {line_number} must be a JSON object")
                    try:
                        reward = float(record["reward"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"Episode recording transition reward at line {line_number} must be numeric") from error
                    if not math.isfinite(reward):
                        raise ValueError(f"Episode recording transition reward at line {line_number} must be finite")
                    if not isinstance(record["done"], bool):
                        raise ValueError(f"Episode recording transition done at line {line_number} must be a boolean")
                    info = record.get("info", {})
                    if not isinstance(info, dict):
                        raise ValueError(f"Episode recording transition info at line {line_number} must be a JSON object")
                    recorder.transitions.append(RecordedTransition(record["observation"], record["action"], reward, record["done"], info))
                else:
                    raise ValueError(f"Unknown episode recording type at line {line_number}: {record.get('type')}")
        return recorder

    def __len__(self) -> int:
        return len(self.transitions)


class ReplayBrain:
    """Brain-compatible action source that replays a recorded action sequence."""

    def __init__(self, transitions: Iterable[RecordedTransition]):
        self.actions = [item.action for item in transitions]
        self.index = 0
        self.current_action = Action.noop()

    def reset(self, context: dict | None = None) -> None:
        self.index = 0
        self.current_action = Action.noop()

    def observe(self, observation: Observation) -> None:
        return None

    def decide(self) -> None:
        if self.index >= len(self.actions):
            self.current_action = Action.noop()
            return
        payload = self.actions[self.index]
        if not isinstance(payload, dict):
            raise ValueError("Replay action payload must be a JSON object")
        self.current_action = Action(joint_targets=payload.get("joint_targets"), motor_commands=payload.get("motor_commands"), gripper_commands=payload.get("gripper_commands"), forces=payload.get("forces"), torques=payload.get("torques"), timestamp=payload.get("timestamp"), metadata=payload.get("metadata", {}))
        self.index += 1

    def get_action(self) -> Action:
        return self.current_action

    def learn(self, reward: float, done: bool, info: dict | None = None) -> None:
        return None

    def shutdown(self) -> None:
        return None
