from dataclasses import dataclass, field
from typing import Any
import json
import numpy as np
from .observation import _json_safe


@dataclass
class Action:
    """Standardized action exchanged between a brain and the simulator."""
    joint_targets: dict[str, float] | None = None
    motor_commands: dict[str, float | dict] | None = None
    gripper_commands: dict[str, Any] | None = None
    forces: dict[str, list[float]] | None = None
    torques: dict[str, list[float]] | None = None
    timestamp: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValueError("Action metadata must be a JSON object")
        for field_name in ("joint_targets", "motor_commands", "gripper_commands", "forces", "torques"):
            if getattr(self, field_name) is not None and not isinstance(getattr(self, field_name), dict):
                raise ValueError(f"Action {field_name} must be a JSON object")
        if not self.has_commands and not self.metadata.get("noop", False):
            raise ValueError("Action must specify at least one command")
        if self.timestamp is not None:
            if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float, np.integer, np.floating)) or not np.isfinite(self.timestamp):
                raise ValueError("Action timestamp must be a finite number")

    @property
    def has_commands(self) -> bool:
        return any(bool(x) for x in (self.joint_targets, self.motor_commands, self.gripper_commands, self.forces, self.torques))

    @classmethod
    def noop(cls, timestamp: float | None = None) -> "Action":
        return cls(timestamp=timestamp, metadata={"noop": True})

    def to_dict(self) -> dict:
        return _json_safe({"joint_targets": self.joint_targets, "motor_commands": self.motor_commands, "gripper_commands": self.gripper_commands, "forces": self.forces, "torques": self.torques, "timestamp": self.timestamp, "metadata": self.metadata})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
