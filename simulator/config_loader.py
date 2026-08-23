from pathlib import Path
from typing import Any
import copy
import yaml


class ConfigLoader:
    """Load, merge, and validate simulator YAML configuration files."""

    REQUIRED_SECTIONS = ("simulator", "physics", "body", "sensors", "actuators")

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()

    def load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {self.config_path}")
        with self.config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        self._validate_root(config)
        base = self.config_path.parent.parent if self.config_path.parent.name == "config" else self.config_path.parent
        for section, key in (("physics", "config_path"), ("body", "config_path"), ("sensors", "config_path"), ("actuators", "config_path")):
            ref = config.get(section, {}).get(key)
            if ref:
                path = Path(ref)
                if not path.is_absolute():
                    path = base / path
                config[section]["loaded"] = self._read_yaml(path)
        return config

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Referenced configuration not found: {path}")
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _validate_root(self, config: dict[str, Any]) -> None:
        missing = [name for name in self.REQUIRED_SECTIONS if name not in config]
        if missing:
            raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
        timestep = float(config["simulator"].get("timestep", 0.005))
        max_timestep = float(config["simulator"].get("max_timestep", 0.1))
        if not 0 < timestep <= max_timestep:
            raise ValueError("simulator.timestep must be positive and <= max_timestep")

    @staticmethod
    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = ConfigLoader.merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
