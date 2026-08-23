from dataclasses import dataclass, field
from typing import Any
import json
import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class Observation:
    """Standardized sensor output; it contains no unconfigured simulator state."""
    timestamp: float
    proprioception: dict | None = None
    vision: dict | None = None
    depth: dict | None = None
    imu: dict | None = None
    touch: dict | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None or not np.isfinite(self.timestamp):
            raise ValueError("Observation requires a finite timestamp")

    def to_dict(self) -> dict:
        return _json_safe({"timestamp": self.timestamp, "proprioception": self.proprioception, "vision": self.vision, "depth": self.depth, "imu": self.imu, "touch": self.touch, "info": self.info})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
