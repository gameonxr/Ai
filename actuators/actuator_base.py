from __future__ import annotations

import math


class Actuator:
    def __init__(self, name: str, joint_name: str, config: dict | None = None):
        self.name, self.joint_name, self.config = name, joint_name, config or {}
        self.max_torque = float(self.config.get("max_torque", 100.0))
        self.max_velocity = float(self.config.get("max_velocity", 100.0))
        self.max_force = float(self.config.get("max_force", 100.0))
        self.damping = float(self.config.get("damping", 0.01))
        self.response_time = max(0.0, float(self.config.get("response_time", 0.0)))
        self.current_command = 0.0

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
