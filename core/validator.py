from dataclasses import replace
import logging
import math
from .action import Action


class ActionValidator:
    """Safety boundary that rejects unknown joints and clamps finite commands."""

    def __init__(self, actuators: dict, config: dict | None = None):
        self.actuators = actuators
        self.config = config or {}
        self.log: list[dict] = []
        self.logger = logging.getLogger(__name__)

    def _value(self, joint: str, value: float) -> tuple[bool, float, str | None]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False, 0.0, f"Non-numeric value for {joint}: {value!r}"
        if not math.isfinite(numeric):
            return False, 0.0, f"Invalid value for {joint}: {value}"
        actuator = self.actuators[joint]
        return True, actuator.clamp(numeric), None

    def validate(self, action: Action) -> tuple[bool, Action, list[str]]:
        if not isinstance(action, Action):
            return False, Action.noop(), ["Brain must return an Action object"]
        errors: list[str] = []
        targets: dict[str, float] = {}
        for joint, value in (action.joint_targets or {}).items():
            if joint not in self.actuators:
                errors.append(f"Unknown joint: {joint}")
                continue
            valid, clamped, error = self._value(joint, value)
            if valid:
                targets[joint] = clamped
            else:
                errors.append(error or f"Invalid command for {joint}")
        commands: dict[str, object] = {}
        for joint, command in (action.motor_commands or {}).items():
            if joint not in self.actuators:
                errors.append(f"Unknown joint: {joint}")
                continue
            if isinstance(command, dict):
                value = command.get("target", 0.0)
                mode = command.get("mode", "torque")
            else:
                value, mode = command, "torque"
            valid, clamped, error = self._value(joint, value)
            if valid:
                commands[joint] = {"target": clamped, "mode": mode}
            else:
                errors.append(error or f"Invalid command for {joint}")
        for name, vectors in (("forces", action.forces), ("torques", action.torques)):
            if vectors:
                for body, vector in vectors.items():
                    try:
                        values = [float(x) for x in vector]
                        if len(values) != 3 or not all(math.isfinite(x) for x in values):
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(f"Invalid {name} for {body}")
        validated = Action(joint_targets=targets or None, motor_commands=commands or None, forces=action.forces, torques=action.torques, timestamp=action.timestamp, metadata=action.metadata) if (targets or commands or action.forces or action.torques) else Action.noop(action.timestamp)
        if errors:
            record = {"errors": errors, "action": action.to_dict()}
            self.log.append(record)
            self.logger.warning("Rejected action: %s", errors)
        return not errors, validated, errors
