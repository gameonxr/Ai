from __future__ import annotations

from .sensor_base import Sensor
from .transforms import GaussianNoise, LowPassFilter


class ProprioceptionSensor(Sensor):
    def __init__(self, name: str, config: dict | None = None):
        super().__init__(name, config)
        try:
            self.noise = GaussianNoise(self.config.get("noise_std", 0.0), self.config.get("seed"))
            self.filter = LowPassFilter(self.config["filter_alpha"]) if "filter_alpha" in self.config else None
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid proprioception sensor configuration: {error}") from error

    def observe(self, physics_state):
        reading = {key: physics_state[key] for key in ("joint_positions", "joint_velocities", "joint_accelerations", "body_position", "body_velocity", "body_rotation", "body_angular_velocity")}
        reading = self.noise.apply(reading)
        return self.filter.apply(reading) if self.filter else reading

    def reset(self):
        self.noise.reset()
        if self.filter:
            self.filter.reset()
