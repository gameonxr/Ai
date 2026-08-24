from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

import yaml


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_positive_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) > 0


def _is_nonnegative_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) >= 0


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
                loaded = yaml.safe_load(handle)
                config = {} if loaded is None else loaded
        except OSError as error:
            return ValidationReport(str(config_path), False, [f"Unable to read configuration: {error}"])
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
                else:
                    try:
                        with referenced.open(encoding="utf-8") as handle:
                            loaded_value = yaml.safe_load(handle)
                            loaded = {} if loaded_value is None else loaded_value
                        if not isinstance(loaded, dict):
                            errors.append(f"{section}.config_path must contain a mapping: {referenced}")
                        elif section == "body":
                            self._validate_body(loaded, errors)
                        elif section == "actuators":
                            self._validate_actuators(loaded, errors, warnings)
                    except OSError as error:
                        errors.append(f"Unable to read {section} configuration: {referenced}: {error}")
                    except yaml.YAMLError as error:
                        errors.append(f"Invalid {section} YAML: {error}")
            elif section != "physics":
                warnings.append(f"{section}.config_path is not set; inline defaults will be used")
        environment = config.get("environment")
        if environment is not None:
            self._validate_environment(environment, errors)
        logging_cfg = config.get("logging")
        if logging_cfg is not None and not isinstance(logging_cfg, dict):
            errors.append("logging must be a mapping when provided")
        return ValidationReport(str(config_path), not errors, errors, warnings)

    @staticmethod
    def _validate_body(body: dict[str, Any], errors: list[str]) -> None:
        links = body.get("links")
        joints = body.get("joints")
        if not isinstance(links, dict):
            errors.append("body.links must be a mapping")
            links = {}
        if not isinstance(joints, dict):
            errors.append("body.joints must be a mapping")
            joints = {}
        if len(joints) < 12:
            errors.append("body.joints must define at least 12 joints")
        for name, joint in joints.items():
            if not isinstance(joint, dict):
                errors.append(f"body.joints.{name} must be a mapping")
                continue
            for endpoint in ("parent", "child"):
                link_name = joint.get(endpoint)
                if link_name not in links:
                    errors.append(f"body.joints.{name}.{endpoint} references unknown link: {link_name}")
            axis = joint.get("axis")
            if not isinstance(axis, (list, tuple)) or len(axis) != 3 or not all(_is_finite_number(value) for value in axis):
                errors.append(f"body.joints.{name}.axis must contain three finite numbers")
            joint_range = joint.get("range")
            if not isinstance(joint_range, (list, tuple)) or len(joint_range) != 2 or not all(_is_finite_number(value) for value in joint_range):
                errors.append(f"body.joints.{name}.range must contain two finite numbers")
            elif float(joint_range[0]) >= float(joint_range[1]):
                errors.append(f"body.joints.{name}.range lower bound must be less than upper bound")
            if not _is_positive_number(joint.get("max_torque")):
                errors.append(f"body.joints.{name}.max_torque must be positive")
        initial_state = body.get("initial_state", {})
        positions = initial_state.get("joint_positions", {}) if isinstance(initial_state, dict) else {}
        if not isinstance(positions, dict):
            errors.append("body.initial_state.joint_positions must be a mapping")
        else:
            for joint_name, position in positions.items():
                if joint_name not in joints:
                    errors.append(f"body.initial_state.joint_positions references unknown joint: {joint_name}")
                elif not _is_finite_number(position):
                    errors.append(f"body.initial_state.joint_positions.{joint_name} must be finite")

    @staticmethod
    def _validate_environment(environment: Any, errors: list[str]) -> None:
        if not isinstance(environment, dict):
            errors.append("environment must be a mapping when provided")
            return
        if "floor_enabled" in environment and not isinstance(environment["floor_enabled"], bool):
            errors.append("environment.floor_enabled must be a boolean")
        if "floor_friction" in environment and (
            isinstance(environment["floor_friction"], bool)
            or not _is_nonnegative_number(environment["floor_friction"])
        ):
            errors.append("environment.floor_friction must be a finite non-negative number")
        if "floor_size" in environment:
            floor_size = environment["floor_size"]
            if (
                not isinstance(floor_size, (list, tuple))
                or len(floor_size) != 2
                or not all(_is_finite_number(value) and float(value) > 0 for value in floor_size)
            ):
                errors.append("environment.floor_size must be a finite positive 2-vector")

    @staticmethod
    def _validate_actuators(actuator_config: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
        actuators = actuator_config.get("actuators", actuator_config)
        if not isinstance(actuators, dict):
            errors.append("actuators configuration must be a mapping")
            return
        control_mode = actuators.get("control_mode", "torque")
        if control_mode not in {"torque", "position"}:
            errors.append("actuators.control_mode must be torque or position")
        defaults = actuators.get("defaults", {})
        if not isinstance(defaults, dict):
            errors.append("actuators.defaults must be a mapping")
            return
        for field_name in ("max_torque", "max_velocity", "max_force"):
            if not _is_positive_number(defaults.get(field_name)):
                errors.append(f"actuators.defaults.{field_name} must be positive")
        for field_name in ("damping", "response_time"):
            if not _is_nonnegative_number(defaults.get(field_name)):
                errors.append(f"actuators.defaults.{field_name} must be non-negative")
        if "response_time" not in defaults:
            warnings.append("actuators.defaults.response_time is not set; actuator response will be immediate")

    def validate_or_raise(self, path: str | Path) -> ValidationReport:
        report = self.validate(path)
        if not report.valid:
            raise ValueError("Configuration validation failed: " + "; ".join(report.errors))
        return report
