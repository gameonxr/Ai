from pathlib import Path
from typing import Any
import copy
import math
import yaml


class ConfigLoader:
    """Load, merge, and validate simulator YAML configuration files."""

    REQUIRED_SECTIONS = ("simulator", "physics", "body", "sensors", "actuators")

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()

    def load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {self.config_path}")
        try:
            with self.config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        except OSError as error:
            raise FileNotFoundError(f"Unable to read configuration: {self.config_path}: {error}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration: {self.config_path}") from error
        if config is None:
            config = {}
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
        try:
            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        except OSError as error:
            raise FileNotFoundError(f"Unable to read referenced configuration: {path}: {error}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML referenced configuration: {path}") from error
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError(f"Referenced configuration must be a mapping: {path}")
        return config

    def _validate_root(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ValueError("Configuration root must be a mapping")
        missing = [name for name in self.REQUIRED_SECTIONS if name not in config]
        if missing:
            raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
        invalid_sections = [name for name in self.REQUIRED_SECTIONS if not isinstance(config[name], dict)]
        if invalid_sections:
            raise ValueError(f"Configuration sections must be mappings: {', '.join(invalid_sections)}")
        raw_timestep = config["simulator"].get("timestep", 0.005)
        raw_max_timestep = config["simulator"].get("max_timestep", 0.1)
        if (
            isinstance(raw_timestep, bool)
            or not isinstance(raw_timestep, (int, float))
            or isinstance(raw_max_timestep, bool)
            or not isinstance(raw_max_timestep, (int, float))
        ):
            raise ValueError("simulator timestep values must be numeric")
        timestep = float(raw_timestep)
        max_timestep = float(raw_max_timestep)
        if not math.isfinite(timestep) or not math.isfinite(max_timestep) or not 0 < timestep <= max_timestep:
            raise ValueError("simulator.timestep must be positive and <= max_timestep")

    @staticmethod
    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(base, dict):
            raise ValueError("Configuration merge base must be a mapping")
        if not isinstance(override, dict):
            raise ValueError("Configuration merge override must be a mapping")
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = ConfigLoader.merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
