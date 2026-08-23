from core.action import Action


class ActuatorController:
    def __init__(self, actuators: dict):
        self.actuators = actuators

    def apply(self, physics_engine, action: Action) -> None:
        physics_engine.apply_action(action)
