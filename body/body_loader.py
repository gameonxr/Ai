from pathlib import Path
import yaml
from .body import Body
from .joint import Joint
from .link import Link


class BodyLoader:
    @staticmethod
    def load(path: str | Path) -> Body:
        body_path = Path(path)
        try:
            with body_path.open(encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
                config = {} if loaded is None else loaded
        except OSError as error:
            raise ValueError(f"Unable to read body configuration {body_path}: {error}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid body YAML in {body_path}: {error}") from error
        if not isinstance(config, dict):
            raise ValueError(f"Body configuration {body_path} must be a mapping")
        body_cfg = config.get("body", {})
        links_cfg = config.get("links", {})
        joints_cfg = config.get("joints", {})
        initial_cfg = config.get("initial_state", {})
        for name, value in (("body", body_cfg), ("links", links_cfg), ("joints", joints_cfg), ("initial_state", initial_cfg)):
            if not isinstance(value, dict):
                raise ValueError(f"Body configuration {body_path} section {name} must be a mapping")
        try:
            links = {name: Link.from_config(name, data) for name, data in links_cfg.items()}
            joints = {name: Joint.from_config(name, data) for name, data in joints_cfg.items()}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid body component in {body_path}: {error}") from error
        if len(joints) < 12:
            raise ValueError("Humanoid configuration must define at least 12 degrees of freedom")
        initial = initial_cfg.get("joint_positions", {}) or {}
        if not isinstance(initial, dict):
            raise ValueError(f"Body configuration {body_path} initial_state.joint_positions must be a mapping")
        try:
            return Body(str(body_cfg.get("type", "humanoid")), float(body_cfg.get("mass", 1.0)), float(body_cfg.get("height", 1.0)), float(body_cfg.get("default_damping", 0.01)), links, joints, {k: float(v) for k, v in initial.items()})
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid body metadata in {body_path}: {error}") from error
