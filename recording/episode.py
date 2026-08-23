from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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
        self.metadata = metadata or {}
        self.transitions: list[RecordedTransition] = []

    def record(self, observation: Observation, action: Action, reward: float = 0.0, done: bool = False, info: dict | None = None) -> None:
        self.transitions.append(RecordedTransition(observation.to_dict(), action.to_dict(), float(reward), bool(done), info or {}))

    def save_jsonl(self, path: str | Path) -> None:
        lines = [json.dumps({"type": "metadata", "metadata": self.metadata}, sort_keys=True)]
        lines.extend(json.dumps({"type": "transition", **asdict(transition)}, sort_keys=True) for transition in self.transitions)
        write_text_atomic("\n".join(lines) + "\n", path)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "EpisodeRecorder":
        recorder = cls()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "metadata":
                    recorder.metadata = record.get("metadata", {})
                elif record.get("type") == "transition":
                    recorder.transitions.append(RecordedTransition(record["observation"], record["action"], float(record["reward"]), bool(record["done"]), record.get("info", {})))
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
        self.current_action = Action(joint_targets=payload.get("joint_targets"), motor_commands=payload.get("motor_commands"), gripper_commands=payload.get("gripper_commands"), forces=payload.get("forces"), torques=payload.get("torques"), timestamp=payload.get("timestamp"), metadata=payload.get("metadata", {}))
        self.index += 1

    def get_action(self) -> Action:
        return self.current_action

    def learn(self, reward: float, done: bool, info: dict | None = None) -> None:
        return None

    def shutdown(self) -> None:
        return None
