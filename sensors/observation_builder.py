from core.observation import Observation


class ObservationBuilder:
    def __init__(self, sensors: dict):
        self.sensors = sensors

    def build(self, physics_state: dict, timestamp: float) -> Observation:
        readings = {name: sensor.observe(physics_state) for name, sensor in self.sensors.items() if sensor.enabled}
        return Observation(timestamp=timestamp, proprioception=readings.get("proprioception"), vision=readings.get("vision"), depth=readings.get("depth"), imu=readings.get("imu"), touch=readings.get("touch"), info={"sensors_enabled": list(readings)})
