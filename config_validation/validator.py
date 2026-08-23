from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ValidationReport:
    path: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "valid": self.valid, "errors": self.errors, "warnings": self.warnings}


class ConfigurationValidator:
    REQUIRED_SECTIONS = ("simulator", "physics", "body", "sensors", "actuators")
    REFERENCED_SECTIONS = ("physics", "body", "sensors", "actuators")

    def validate(self, path: str | Path) -> ValidationReport:
        config_path = Path(path).resolve()
        errors: list[str] = []
        warnings: list[str] = []
        if not config_path.exists():
            return ValidationReport(str(config_path), False, [f"Configuration not found: {config_path}"])
        try:
            with config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except yaml.YAMLError as error:
            return ValidationReport(str(config_path), False, [f"Invalid YAML: {error}"])
        if not isinstance(config, dict):
            return ValidationReport(str(config_path), False, ["Root configuration must be a mapping"])
        missing = [section for section in self.REQUIRED_SECTIONS if section not in config]
        errors.extend(f"Missing required section: {section}" for section in missing)
        simulator = config.get("simulator", {})
        if isinstance(simulator, dict):
            try:
                timestep = float(simulator.get("timestep", 0.005))
                max_timestep = float(simulator.get("max_timestep", 0.1))
                if not 0 < timestep <= max_timestep:
                    errors.append("simulator.timestep must be positive and <= max_timestep")
            except (TypeError, ValueError):
                errors.append("simulator.timestep and max_timestep must be numeric")
        for section in self.REFERENCED_SECTIONS:
            value = config.get(section)
            if not isinstance(value, dict):
                continue
            reference = value.get("config_path")
            if reference:
                referenced = Path(reference)
                if not referenced.is_absolute():
                    base = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
                    referenced = base / referenced
                if not referenced.exists():
                    errors.append(f"{section}.config_path does not exist: {referenced}")
            elif section != "physics":
                warnings.append(f"{section}.config_path is not set; inline defaults will be used")
        logging_cfg = config.get("logging")
        if logging_cfg is not None and not isinstance(logging_cfg, dict):
            errors.append("logging must be a mapping when provided")
        return ValidationReport(str(config_path), not errors, errors, warnings)

    def validate_or_raise(self, path: str | Path) -> ValidationReport:
        report = self.validate(path)
        if not report.valid:
            raise ValueError("Configuration validation failed: " + "; ".join(report.errors))
        return report
