from __future__ import annotations

import importlib
import platform
import sys
from typing import Any

from config_validation import ConfigurationValidator


REQUIRED_MODULES = ("numpy", "yaml", "matplotlib")
OPTIONAL_MODULES = ("mujoco",)


def _module_status(names: tuple[str, ...]) -> dict[str, bool]:
    status = {}
    for name in names:
        try:
            importlib.import_module(name)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


def to_artifact_dict(report: dict[str, Any], config_path: str) -> dict[str, Any]:
    """Return a validated health artifact payload for persistence."""
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("config_path must be a non-empty string")
    if not isinstance(report, dict):
        raise ValueError("health report must be an object")
    if not isinstance(report.get("healthy"), bool):
        raise ValueError("health healthy flag must be boolean")
    config = report.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("valid"), bool):
        raise ValueError("health config must be an object with a boolean valid flag")
    for field in ("errors", "warnings"):
        values = config.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"health config {field} must be a list of strings")
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("health dependencies must be an object")
    for group in ("required", "optional"):
        values = dependencies.get(group)
        if not isinstance(values, dict) or any(not isinstance(name, str) or not isinstance(available, bool) for name, available in values.items()):
            raise ValueError(f"health dependencies {group} must be a string-to-boolean object")
    for field in ("python", "platform"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            raise ValueError(f"health {field} must be a non-empty string")
    return {"artifact_type": "health", "schema_version": 1, "config_path": config_path, **report}


def collect_health(config_path: str = "config/simulator_config.yaml") -> dict[str, Any]:
    config_report = ConfigurationValidator().validate(config_path)
    required = _module_status(REQUIRED_MODULES)
    optional = _module_status(OPTIONAL_MODULES)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config_report.to_dict(),
        "dependencies": {"required": required, "optional": optional},
        "healthy": config_report.valid and all(required.values()),
    }
