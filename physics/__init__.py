from .physics_engine import PhysicsEngine
from .toy_backend import ToyPhysicsEngine
from .mujoco_backend import MuJoCoBackend
from .bullet_backend import BulletBackend


def create_physics_engine(name: str, config: dict) -> PhysicsEngine:
    name = name.lower()
    if name == "mujoco":
        return MuJoCoBackend(config)
    if name == "bullet":
        return BulletBackend(config)
    if name == "toy":
        return ToyPhysicsEngine(config)
    raise ValueError(f"Unsupported physics engine: {name}")

__all__ = ["PhysicsEngine", "ToyPhysicsEngine", "MuJoCoBackend", "BulletBackend", "create_physics_engine"]
