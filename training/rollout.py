from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from artifact_io import write_text_atomic
from core.action import Action
from core.observation import Observation


@dataclass
class Transition:
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class Rollout:
    """In-memory episode buffer with JSONL persistence for later training stages."""

    def __init__(self):
        self.transitions: list[Transition] = []

    def append(self, observation: Observation, action: Action, reward: float, done: bool = False, info: dict | None = None) -> None:
        self.transitions.append(Transition(observation.to_dict(), action.to_dict(), float(reward), bool(done), info or {}))

    @property
    def total_reward(self) -> float:
        return sum(item.reward for item in self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def to_dicts(self) -> list[dict]:
        return [asdict(item) for item in self.transitions]

    def save_jsonl(self, path: str | Path) -> None:
        lines = [json.dumps(item, sort_keys=True) for item in self.to_dicts()]
        content = "\n".join(lines)
        if content:
            content += "\n"
        write_text_atomic(content, path)
