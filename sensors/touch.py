from .sensor_base import Sensor


class TouchSensor(Sensor):
    def observe(self, physics_state):
        return {"contacts": physics_state.get("contacts", [])}
