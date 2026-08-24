from pathlib import Path
import numpy as np
from body import BodyLoader
from brain import BrainInterface
from actuators.actuator_registry import build_actuators
from actuators.actuator_controller import ActuatorController
from core.validator import ActionValidator
from environment.world import World
from physics import create_physics_engine
from sensors.observation_builder import ObservationBuilder
from sensors.sensor_registry import build_sensors
from rendering import create_renderer
from observability import SimulationMetrics, configure_logging, log_event
from checkpointing import CheckpointManager
from .config_loader import ConfigLoader


class Simulator:
    """Orchestrates physics -> sensors -> brain -> validation -> actuators."""

    def __init__(self, config_path: str = "config/simulator_config.yaml"):
        self.config_path = Path(config_path).resolve()
        self.config = ConfigLoader(self.config_path).load()
        logging_cfg = self.config.get("logging", {})
        self.logger = configure_logging(logging_cfg.get("level", "INFO"), logging_cfg.get("file"))
        self.metrics = SimulationMetrics()
        self.timestep = float(self.config["simulator"].get("timestep", 0.005))
        body_path = self._path_from_config("body")
        self.body = BodyLoader.load(body_path)
        physics_cfg = self.config["physics"].get("loaded", {}).get("physics", self.config["physics"])
        engine_name = physics_cfg.get("engine", "mujoco")
        self.physics = create_physics_engine(engine_name, physics_cfg)
        self.physics.load_body(self.body)
        environment_cfg = self.config.get("environment", {})
        if not isinstance(environment_cfg, dict):
            raise ValueError("environment config must be a mapping")
        self.world = World(physics_cfg.get("gravity", [0, 0, -9.81]), environment_cfg.get("floor_enabled", True), environment_cfg.get("floor_friction", 0.5), environment_cfg.get("floor_size", [10, 10]))
        sensor_cfg = self.config["sensors"].get("loaded", {}).get("sensors", self.config["sensors"])
        actuator_cfg = self.config["actuators"].get("loaded", {}).get("actuators", self.config["actuators"])
        self.sensors = build_sensors(sensor_cfg)
        self.actuators = build_actuators(self.body.joints, actuator_cfg)
        self.actuator_controller = ActuatorController(self.actuators)
        self.validator = ActionValidator(self.actuators, self.config["simulator"])
        self.observation_builder = ObservationBuilder(self.sensors)
        rendering_cfg = self.config.get("rendering", {})
        self.renderer = create_renderer(rendering_cfg.get("renderer", "matplotlib"), rendering_cfg) if rendering_cfg.get("enabled", False) else None
        self.brain: BrainInterface | None = None
        self.current_time = 0.0
        self.step_count = 0
        self.paused = False
        self.running = False
        self._shutdown = False
        log_event(self.logger, 20, "simulator_initialized", {"engine": engine_name, "dof": self.body.dof, "timestep": self.timestep})

    def _path_from_config(self, section: str) -> Path:
        path = Path(self.config[section]["config_path"])
        if not path.is_absolute():
            path = self.config_path.parent.parent / path
        return path

    def _ensure_active(self) -> None:
        if self._shutdown:
            raise RuntimeError("Simulator is shut down")

    def set_brain(self, brain: BrainInterface | None) -> None:
        self._ensure_active()
        if brain is not None and not isinstance(brain, BrainInterface):
            raise TypeError("brain must implement BrainInterface or be None")
        self.brain = brain

    def reset(self, seed: int | None = None) -> None:
        self._ensure_active()
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        if seed is not None:
            np.random.seed(seed)
        self.physics.reset(seed)
        self.actuator_controller.reset()
        self.metrics.reset_episode()
        log_event(self.logger, 20, "episode_reset", {"seed": seed})
        self.current_time = 0.0
        self.step_count = 0
        self.paused = False
        for sensor in self.sensors.values():
            sensor.reset()
        if self.brain:
            self.brain.reset({"seed": seed})

    def step(self, n_steps: int = 1):
        self._ensure_active()
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        observation = None
        for _ in range(n_steps):
            if self.paused:
                return observation
            self.physics.step(self.timestep)
            state = self.physics.get_body_state()
            state["contacts"] = self.physics.get_contact_info()
            observation = self.observation_builder.build(state, self.current_time)
            action_applied = False
            invalid_action = False
            if self.brain:
                self.brain.observe(observation)
                self.brain.decide()
                valid, action, errors = self.validator.validate(self.brain.get_action())
                invalid_action = bool(errors)
                self.actuator_controller.apply(self.physics, action, self.timestep)
                action_applied = action.has_commands
                if errors:
                    log_event(self.logger, 30, "invalid_action", {"errors": errors})
                    if self.config["simulator"].get("debug", False):
                        print(f"Invalid action: {errors}")
            self.current_time += self.timestep
            self.step_count += 1
            self.metrics.record_step(self.timestep, action_applied, invalid_action)
        return observation

    def pause(self) -> None:
        self._ensure_active()
        self.paused = True

    def resume(self) -> None:
        self._ensure_active()
        self.paused = False

    def get_state(self) -> dict:
        self._ensure_active()
        return {"current_time": self.current_time, "step_count": self.step_count, "paused": self.paused, "physics_state": self.physics.get_body_state(), "metrics": self.metrics.snapshot()}

    def save_checkpoint(self, path: str | Path, run_id: str = "default", metadata: dict | None = None):
        """Persist simulator state for a later resumable run."""
        self._ensure_active()
        checkpoint = CheckpointManager.save(self, path, run_id, metadata)
        log_event(self.logger, 20, "checkpoint_saved", {"path": str(path), "step": self.step_count})
        return checkpoint

    def restore_checkpoint(self, path: str | Path):
        """Restore a checkpoint created by ``save_checkpoint``."""
        self._ensure_active()
        checkpoint = CheckpointManager.restore(self, path)
        log_event(self.logger, 20, "checkpoint_restored", {"path": str(path), "step": self.step_count})
        return checkpoint

    def render(self, output_path: str | Path | None = None):
        """Render the current state when rendering is enabled in YAML."""
        self._ensure_active()
        if self.renderer is None:
            raise RuntimeError("Rendering is disabled; set rendering.enabled to true")
        return self.renderer.render(self.body, self.physics.get_body_state(), output_path)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        log_event(self.logger, 20, "simulator_shutdown", self.metrics.snapshot())
        if self.brain:
            self.brain.shutdown()
        if self.renderer:
            self.renderer.close()
        self.physics.shutdown()
        self._shutdown = True
