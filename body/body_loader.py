from pathlib import Path
import yaml
from .body import Body
from .joint import Joint
from .link import Link


class BodyLoader:
    @staticmethod
    def load(path: str | Path) -> Body:
        with Path(path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        body_cfg = config.get("body", {})
        links = {name: Link.from_config(name, data) for name, data in config.get("links", {}).items()}
        joints = {name: Joint.from_config(name, data) for name, data in config.get("joints", {}).items()}
        if len(joints) < 12:
            raise ValueError("Humanoid configuration must define at least 12 degrees of freedom")
        initial = config.get("initial_state", {}).get("joint_positions", {}) or {}
        return Body(str(body_cfg.get("type", "humanoid")), float(body_cfg.get("mass", 1.0)), float(body_cfg.get("height", 1.0)), float(body_cfg.get("default_damping", 0.01)), links, joints, {k: float(v) for k, v in initial.items()})
