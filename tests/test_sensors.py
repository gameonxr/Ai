from sensors import ObservationBuilder
from sensors.sensor_registry import build_sensors


def test_only_enabled_sensors_are_exposed():
    sensors = build_sensors({"proprioception": {"enabled": True}, "vision": {"enabled": False}, "imu": {"enabled": True}})
    observation = ObservationBuilder(sensors).build({"joint_positions": {}, "joint_velocities": {}, "joint_accelerations": {}, "body_position": [0,0,1], "body_velocity": [0,0,0], "body_rotation": [0,0,0,1], "body_angular_velocity": [0,0,0], "gravity": [0,0,-9.81]}, 0.0)
    assert observation.proprioception is not None and observation.imu is not None and observation.vision is None
