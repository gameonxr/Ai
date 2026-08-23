from copy import deepcopy
import numpy as np
from .physics_engine import PhysicsEngine


class ToyPhysicsEngine(PhysicsEngine):
    """Small deterministic rigid-joint backend used as a portable fallback and test double."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.gravity = np.asarray(self.config.get("gravity", [0, 0, -9.81]), dtype=float)
        self.body = None
        self.positions: dict[str, float] = {}
        self.velocities: dict[str, float] = {}
        self.accelerations: dict[str, float] = {}
        self.commands: dict[str, dict] = {}
        self.body_position = np.array([0.0, 0.0, 0.875])
        self.body_velocity = np.zeros(3)
        self.body_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        self.body_angular_velocity = np.zeros(3)
        self.time = 0.0
        self.rng = np.random.default_rng(0)

    def load_body(self, body_definition) -> None:
        self.body = body_definition
        self.positions = {name: float(body_definition.initial_joint_positions.get(name, 0.0)) for name in body_definition.joints}
        self.velocities = {name: 0.0 for name in body_definition.joints}
        self.accelerations = {name: 0.0 for name in body_definition.joints}
        self.body_position[2] = body_definition.height / 2.0

    def reset(self, seed=None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for name in self.positions:
            self.positions[name] = float(self.body.initial_joint_positions.get(name, 0.0))
            self.velocities[name] = 0.0
            self.accelerations[name] = 0.0
        self.commands.clear()
        self.time = 0.0
        self.body_velocity[:] = 0.0
        self.body_angular_velocity[:] = 0.0
        self.body_rotation[:] = [0.0, 0.0, 0.0, 1.0]

    def apply_action(self, action) -> None:
        for joint, command in (action.joint_targets or {}).items():
            if joint in self.positions:
                self.commands[joint] = {"target": float(command), "mode": "position"}
        for joint, command in (action.motor_commands or {}).items():
            if joint in self.positions:
                self.commands[joint] = command if isinstance(command, dict) else {"target": float(command), "mode": "torque"}

    def step(self, dt=0.005) -> None:
        if isinstance(dt, bool) or not isinstance(dt, (int, float, np.integer, np.floating)):
            raise ValueError("dt must be a finite number")
        dt = float(dt)
        if not np.isfinite(dt):
            raise ValueError("dt must be a finite number")
        if not 0 < dt <= 0.1:
            raise ValueError("dt must be in (0, 0.1]")
        for name, joint in self.body.joints.items():
            command = self.commands.get(name, {"target": 0.0, "mode": "torque"})
            target = float(command.get("target", 0.0))
            if command.get("mode", "torque") == "position":
                acceleration = (target - self.positions[name]) * 30.0 - joint.damping * self.velocities[name]
            else:
                acceleration = target - joint.damping * self.velocities[name]
            self.accelerations[name] = acceleration
            self.velocities[name] += acceleration * dt
            self.positions[name] = joint.clamp(self.positions[name] + self.velocities[name] * dt)
            low, high = joint.range_radians
            if self.positions[name] in (low, high):
                self.velocities[name] = 0.0
        self.time += dt

    def get_body_state(self) -> dict:
        return {"joint_positions": deepcopy(self.positions), "joint_velocities": deepcopy(self.velocities), "joint_accelerations": deepcopy(self.accelerations), "body_position": self.body_position.tolist(), "body_velocity": self.body_velocity.tolist(), "body_rotation": self.body_rotation.tolist(), "body_angular_velocity": self.body_angular_velocity.tolist(), "gravity": self.gravity.tolist(), "time": self.time}

    def get_checkpoint_state(self) -> dict:
        return {"positions": deepcopy(self.positions), "velocities": deepcopy(self.velocities), "accelerations": deepcopy(self.accelerations), "commands": deepcopy(self.commands), "body_position": self.body_position.tolist(), "body_velocity": self.body_velocity.tolist(), "body_rotation": self.body_rotation.tolist(), "body_angular_velocity": self.body_angular_velocity.tolist(), "gravity": self.gravity.tolist(), "time": self.time}

    def restore_checkpoint_state(self, state: dict) -> None:
        if self.body is None:
            raise RuntimeError("Body must be loaded before restoring a checkpoint")
        expected = set(self.positions)
        if set(state.get("positions", {})) != expected:
            raise ValueError("Checkpoint joint set does not match the loaded body")
        self.positions = {name: float(value) for name, value in state["positions"].items()}
        self.velocities = {name: float(value) for name, value in state["velocities"].items()}
        self.accelerations = {name: float(value) for name, value in state["accelerations"].items()}
        self.commands = deepcopy(state.get("commands", {}))
        self.body_position = np.asarray(state["body_position"], dtype=float)
        self.body_velocity = np.asarray(state["body_velocity"], dtype=float)
        self.body_rotation = np.asarray(state["body_rotation"], dtype=float)
        self.body_angular_velocity = np.asarray(state["body_angular_velocity"], dtype=float)
        self.gravity = np.asarray(state["gravity"], dtype=float)
        self.time = float(state["time"])

    def get_contact_info(self) -> list:
        return [{"position": [float(self.body_position[0]), float(self.body_position[1]), 0.0], "force": [0.0, 0.0, float(self.body.mass * abs(self.gravity[2]))], "object_id": "floor"}] if self.body is not None else []

    def set_gravity(self, gravity) -> None:
        gravity = np.asarray(gravity, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be a finite 3-vector")
        self.gravity = gravity

    def shutdown(self) -> None:
        self.commands.clear()
