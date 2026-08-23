from .sensor_base import Sensor


class ProprioceptionSensor(Sensor):
    def observe(self, physics_state):
        return {key: physics_state[key] for key in ("joint_positions", "joint_velocities", "joint_accelerations", "body_position", "body_velocity", "body_rotation", "body_angular_velocity")}
