from copy import deepcopy
import math
import numpy as np
from .physics_engine import PhysicsEngine


class ToyPhysicsEngine(PhysicsEngine):
    """Small deterministic rigid-joint backend used as a portable fallback and test double."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.gravity = np.zeros(3, dtype=float)
        self.set_gravity(self.config.get("gravity", [0, 0, -9.81]))
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
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
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
                self.commands[joint] = self._normalize_command(command, "position")
        for joint, command in (action.motor_commands or {}).items():
            if joint in self.positions:
                self.commands[joint] = self._normalize_command(command, "torque")

    @staticmethod
    def _normalize_command(command, default_mode: str) -> dict:
        if isinstance(command, dict):
            target = command.get("target", 0.0)
            mode = command.get("mode", default_mode)
        else:
            target = command
            mode = default_mode
        if isinstance(target, bool) or not isinstance(target, (int, float, np.integer, np.floating)) or not math.isfinite(float(target)):
            raise ValueError("motor command target must be a finite number")
        if mode not in {"torque", "position"}:
            raise ValueError("motor command mode must be torque or position")
        return {"target": float(target), "mode": mode}

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
        if not isinstance(state, dict):
            raise ValueError("ToyPhysics checkpoint state must be an object")
        expected = set(self.positions)
        restored = {}
        for field_name in ("positions", "velocities", "accelerations"):
            values = state.get(field_name)
            if not isinstance(values, dict) or set(values) != expected:
                raise ValueError(f"ToyPhysics checkpoint {field_name} dof map does not match the loaded body")
            try:
                restored[field_name] = {name: float(value) for name, value in values.items()}
            except (TypeError, ValueError) as error:
                raise ValueError(f"ToyPhysics checkpoint {field_name} values must be numeric") from error
            if not all(np.isfinite(value) for value in restored[field_name].values()):
                raise ValueError(f"ToyPhysics checkpoint {field_name} values must be finite")
        commands = state.get("commands", {})
        if not isinstance(commands, dict):
            raise ValueError("ToyPhysics checkpoint commands must be an object")
        for name, command in commands.items():
            if not isinstance(command, dict):
                raise ValueError("ToyPhysics checkpoint command values must be objects")
            if command.get("mode", "torque") not in {"torque", "position"}:
                raise ValueError("ToyPhysics checkpoint command modes must be torque or position")
            if "target" in command:
                try:
                    target = float(command["target"])
                except (TypeError, ValueError) as error:
                    raise ValueError("ToyPhysics checkpoint command targets must be numeric") from error
                if not np.isfinite(target):
                    raise ValueError("ToyPhysics checkpoint command targets must be finite")
        vectors = {}
        for field_name, size in (("body_position", 3), ("body_velocity", 3), ("body_rotation", 4), ("body_angular_velocity", 3), ("gravity", 3)):
            try:
                vector = np.asarray(state[field_name], dtype=float)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"ToyPhysics checkpoint {field_name} must be a finite vector") from error
            if vector.shape != (size,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"ToyPhysics checkpoint {field_name} must be a finite vector of length {size}")
            vectors[field_name] = vector
        try:
            time = float(state["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("ToyPhysics checkpoint time must be a finite non-negative number") from error
        if not np.isfinite(time) or time < 0:
            raise ValueError("ToyPhysics checkpoint time must be a finite non-negative number")
        self.positions = restored["positions"]
        self.velocities = restored["velocities"]
        self.accelerations = restored["accelerations"]
        self.commands = deepcopy(commands)
        self.body_position = vectors["body_position"]
        self.body_velocity = vectors["body_velocity"]
        self.body_rotation = vectors["body_rotation"]
        self.body_angular_velocity = vectors["body_angular_velocity"]
        self.gravity = vectors["gravity"]
        self.time = time

    def get_contact_info(self) -> list:
        return [{"position": [float(self.body_position[0]), float(self.body_position[1]), 0.0], "force": [0.0, 0.0, float(self.body.mass * abs(self.gravity[2]))], "object_id": "floor"}] if self.body is not None else []

    def set_gravity(self, gravity) -> None:
        try:
            gravity = np.asarray(gravity, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("gravity must be a finite 3-vector") from error
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must be a finite 3-vector")
        self.gravity = gravity

    def shutdown(self) -> None:
        self.commands.clear()
