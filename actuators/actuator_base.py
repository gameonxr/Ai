import math


class Actuator:
    def __init__(self, name: str, joint_name: str, config: dict | None = None):
        self.name, self.joint_name, self.config = name, joint_name, config or {}
        self.max_torque = float(self.config.get("max_torque", 100.0))
        self.max_velocity = float(self.config.get("max_velocity", 100.0))
        self.max_force = float(self.config.get("max_force", 100.0))
        self.damping = float(self.config.get("damping", 0.01))

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

    def execute(self, physics_engine, command: float):
        valid, value = self.validate_command(command)
        if not valid:
            raise ValueError(f"Invalid actuator command: {command}")
        return value
