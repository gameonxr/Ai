from pathlib import Path
import numpy as np
from body import BodyLoader
from brain import BrainInterface
from actuators.actuator_registry import build_actuators
from core.validator import ActionValidator
from environment.world import World
from physics import create_physics_engine
from sensors.observation_builder import ObservationBuilder
from sensors.sensor_registry import build_sensors
from rendering import create_renderer
from .config_loader import ConfigLoader


class Simulator:
    """Orchestrates physics -> sensors -> brain -> validation -> actuators."""

    def __init__(self, config_path: str = "config/simulator_config.yaml"):
        self.config_path = Path(config_path).resolve()
        self.config = ConfigLoader(self.config_path).load()
        self.timestep = float(self.config["simulator"].get("timestep", 0.005))
        body_path = self._path_from_config("body")
        self.body = BodyLoader.load(body_path)
        physics_cfg = self.config["physics"].get("loaded", {}).get("physics", self.config["physics"])
        engine_name = physics_cfg.get("engine", "mujoco")
        self.physics = create_physics_engine(engine_name, physics_cfg)
        self.physics.load_body(self.body)
        self.world = World(list(physics_cfg.get("gravity", [0, 0, -9.81])), bool(self.config.get("environment", {}).get("floor_enabled", True)), float(self.config.get("environment", {}).get("floor_friction", 0.5)), tuple(self.config.get("environment", {}).get("floor_size", [10, 10])))
        sensor_cfg = self.config["sensors"].get("loaded", {}).get("sensors", self.config["sensors"])
        actuator_cfg = self.config["actuators"].get("loaded", {}).get("actuators", self.config["actuators"])
        self.sensors = build_sensors(sensor_cfg)
        self.actuators = build_actuators(self.body.joints, actuator_cfg)
        self.validator = ActionValidator(self.actuators, self.config["simulator"])
        self.observation_builder = ObservationBuilder(self.sensors)
        rendering_cfg = self.config.get("rendering", {})
        self.renderer = create_renderer(rendering_cfg.get("renderer", "matplotlib"), rendering_cfg) if rendering_cfg.get("enabled", False) else None
        self.brain: BrainInterface | None = None
        self.current_time = 0.0
        self.step_count = 0
        self.paused = False
        self.running = False

    def _path_from_config(self, section: str) -> Path:
        path = Path(self.config[section]["config_path"])
        if not path.is_absolute():
            path = self.config_path.parent.parent / path
        return path

    def set_brain(self, brain: BrainInterface) -> None:
        self.brain = brain

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.physics.reset(seed)
        self.current_time = 0.0
        self.step_count = 0
        self.paused = False
        for sensor in self.sensors.values():
            sensor.reset()
        if self.brain:
            self.brain.reset({"seed": seed})

    def step(self, n_steps: int = 1):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        observation = None
        for _ in range(n_steps):
            if self.paused:
                return observation
            self.physics.step(self.timestep)
            state = self.physics.get_body_state()
            state["contacts"] = self.physics.get_contact_info()
            observation = self.observation_builder.build(state, self.current_time)
            if self.brain:
                self.brain.observe(observation)
                self.brain.decide()
                valid, action, errors = self.validator.validate(self.brain.get_action())
                self.physics.apply_action(action)
                if errors and self.config["simulator"].get("debug", False):
                    print(f"Invalid action: {errors}")
            self.current_time += self.timestep
            self.step_count += 1
        return observation

    def pause(self) -> None: self.paused = True
    def resume(self) -> None: self.paused = False

    def get_state(self) -> dict:
        return {"current_time": self.current_time, "step_count": self.step_count, "paused": self.paused, "physics_state": self.physics.get_body_state()}

    def render(self, output_path: str | Path | None = None):
        """Render the current state when rendering is enabled in YAML."""
        if self.renderer is None:
            raise RuntimeError("Rendering is disabled; set rendering.enabled to true")
        return self.renderer.render(self.body, self.physics.get_body_state(), output_path)

    def shutdown(self) -> None:
        if self.brain:
            self.brain.shutdown()
        if self.renderer:
            self.renderer.close()
        self.physics.shutdown()
