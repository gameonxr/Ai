from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
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
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        if info is not None and not isinstance(info, dict):
            raise ValueError("info must be a mapping when provided")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError("reward must be a finite number")
        if not isinstance(done, bool):
            raise ValueError("done must be a boolean")
        self.transitions.append(Transition(observation.to_dict(), action.to_dict(), float(reward), done, info or {}))

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
