from .sensor_base import Sensor


class TouchSensor(Sensor):
    def observe(self, physics_state):
        if not isinstance(physics_state, dict):
            raise ValueError("physics_state must be an object")
        contacts = physics_state.get("contacts", [])
        if not isinstance(contacts, list):
            raise ValueError("contacts must be a list")
        return {"contacts": contacts}
