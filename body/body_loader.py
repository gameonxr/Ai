from pathlib import Path
import math
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
            initial_positions = {
                name: BodyLoader._numeric_value(value, f"initial joint position {name}")
                for name, value in initial.items()
            }
            return Body(
                str(body_cfg.get("type", "humanoid")),
                BodyLoader._numeric_value(body_cfg.get("mass", 1.0), "mass", positive=True),
                BodyLoader._numeric_value(body_cfg.get("height", 1.0), "height", positive=True),
                BodyLoader._numeric_value(body_cfg.get("default_damping", 0.01), "default_damping"),
                links,
                joints,
                initial_positions,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid body metadata in {body_path}: {error}") from error

    @staticmethod
    def _numeric_value(value, name: str, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Body {name} must be a finite number")
        numeric = float(value)
        if positive and numeric <= 0:
            raise ValueError(f"Body {name} must be positive")
        if not positive and name == "default_damping" and numeric < 0:
            raise ValueError("Body default_damping must be non-negative")
        return numeric
