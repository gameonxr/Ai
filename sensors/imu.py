from __future__ import annotations

from .sensor_base import Sensor
from .transforms import GaussianNoise, LowPassFilter


class IMUSensor(Sensor):
    def __init__(self, name: str, config: dict | None = None):
        super().__init__(name, config)
        try:
            self.noise = GaussianNoise(float(self.config.get("noise_std", 0.0)), self.config.get("seed"))
            self.filter = LowPassFilter(float(self.config.get("filter_alpha", 1.0))) if "filter_alpha" in self.config else None
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid IMU sensor configuration: {error}") from error

    def observe(self, physics_state):
        reading = {"acceleration": physics_state.get("gravity", [0.0, 0.0, -9.81]), "angular_velocity": physics_state["body_angular_velocity"], "orientation": physics_state["body_rotation"]}
        reading = self.noise.apply(reading)
        return self.filter.apply(reading) if self.filter else reading

    def reset(self):
        self.noise.reset()
        if self.filter:
            self.filter.reset()
