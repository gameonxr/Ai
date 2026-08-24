from .motor import MotorActuator


def build_actuators(joints: dict, config: dict) -> dict:
    if not isinstance(joints, dict):
        raise ValueError("joints must be an object")
    if not isinstance(config, dict):
        raise ValueError("actuator configuration must be an object")
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("actuator defaults must be an object")
    mode = config.get("control_mode", "torque")
    if mode not in {"torque", "position"}:
        raise ValueError("control_mode must be 'torque' or 'position'")
    result = {}
    for name, joint in joints.items():
        settings = {**defaults, "max_torque": joint.max_torque, "damping": joint.damping, "control_mode": mode}
        result[name] = MotorActuator(name, name, settings)
    return result
