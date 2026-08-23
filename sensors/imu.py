from .sensor_base import Sensor


class IMUSensor(Sensor):
    def observe(self, physics_state):
        return {"acceleration": physics_state.get("gravity", [0.0, 0.0, -9.81]), "angular_velocity": physics_state["body_angular_velocity"], "orientation": physics_state["body_rotation"]}
