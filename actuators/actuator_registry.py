from .motor import MotorActuator


def build_actuators(joints: dict, config: dict) -> dict:
    defaults = config.get("defaults", {})
    mode = config.get("control_mode", "torque")
    result = {}
    for name, joint in joints.items():
        settings = {**defaults, "max_torque": joint.max_torque, "damping": joint.damping, "control_mode": mode}
        result[name] = MotorActuator(name, name, settings)
    return result
