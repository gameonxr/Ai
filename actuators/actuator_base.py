from __future__ import annotations

import math


class Actuator:
    def __init__(self, name: str, joint_name: str, config: dict | None = None):
        if config is not None and not isinstance(config, dict):
            raise ValueError("Actuator config must be a mapping when provided")
        self.name, self.joint_name, self.config = name, joint_name, config or {}
        self.max_torque = self._numeric_config("max_torque", 100.0, positive=True)
        self.max_velocity = self._numeric_config("max_velocity", 100.0, positive=True)
        self.max_force = self._numeric_config("max_force", 100.0, positive=True)
        self.damping = self._numeric_config("damping", 0.01)
        self.response_time = self._numeric_config("response_time", 0.0)
        self.current_command = 0.0

    def _numeric_config(self, name: str, default: float, positive: bool = False) -> float:
        value = self.config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Actuator {name} must be a finite number")
        value = float(value)
        if positive and value <= 0:
            raise ValueError(f"Actuator {name} must be positive")
        if not positive and value < 0:
            raise ValueError(f"Actuator {name} must be non-negative")
        return value

    def validate_command(self, value: float) -> tuple[bool, float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False, 0.0
        if not math.isfinite(value):
            return False, 0.0
        return True, self.clamp(value)

    def clamp(self, value: float) -> float:
        return max(-self.max_torque, min(self.max_torque, float(value)))

    def shape_command(self, target: float, dt: float) -> float:
        """Apply a first-order response model and an optional velocity limit."""
        valid, target = self.validate_command(target)
        if not valid or dt <= 0:
            return self.current_command
        if self.response_time > 0:
            alpha = min(1.0, float(dt) / self.response_time)
            target = self.current_command + alpha * (target - self.current_command)
        max_delta = self.max_velocity * float(dt)
        target = max(self.current_command - max_delta, min(self.current_command + max_delta, target))
        self.current_command = self.clamp(target)
        return self.current_command

    def reset(self) -> None:
        self.current_command = 0.0

    def execute(self, physics_engine, command: float):
        valid, value = self.validate_command(command)
        if not valid:
            raise ValueError(f"Invalid actuator command: {command}")
        return value
