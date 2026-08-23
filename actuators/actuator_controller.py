from __future__ import annotations

from core.action import Action


class ActuatorController:
    def __init__(self, actuators: dict):
        self.actuators = actuators

    def prepare(self, action: Action, dt: float) -> Action:
        """Shape motor commands before they cross into the physics abstraction."""
        if not action.motor_commands:
            return action
        commands = {}
        for joint, command in action.motor_commands.items():
            if joint not in self.actuators:
                continue
            payload = command if isinstance(command, dict) else {"target": command, "mode": "torque"}
            mode = payload.get("mode", "torque")
            target = payload.get("target", 0.0)
            shaped = self.actuators[joint].shape_command(target, dt) if mode == "torque" else target
            commands[joint] = {"target": shaped, "mode": mode}
        return Action(joint_targets=action.joint_targets, motor_commands=commands or None, gripper_commands=action.gripper_commands, forces=action.forces, torques=action.torques, timestamp=action.timestamp, metadata=action.metadata)

    def reset(self) -> None:
        for actuator in self.actuators.values():
            actuator.reset()

    def apply(self, physics_engine, action: Action, dt: float | None = None) -> None:
        physics_engine.apply_action(self.prepare(action, dt) if dt is not None else action)
