from .actuator_base import Actuator


class MotorActuator(Actuator):
    def __init__(self, name: str, joint_name: str, config: dict | None = None):
        super().__init__(name, joint_name, config)
        self.control_mode = self.config.get("control_mode", "torque")

    def execute(self, physics_engine, command):
        command = command if isinstance(command, dict) else {"target": command, "mode": self.control_mode}
        valid, value = self.validate_command(command.get("target", 0.0))
        if not valid:
            raise ValueError(f"Invalid motor command: {command}")
        physics_engine.apply_action(__import__("core.action", fromlist=["Action"]).Action(motor_commands={self.joint_name: {"target": value, "mode": command.get("mode", self.control_mode)}}))
        return value
