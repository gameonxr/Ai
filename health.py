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
