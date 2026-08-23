from .toy_backend import ToyPhysicsEngine


class BulletBackend(ToyPhysicsEngine):
    """PyBullet-compatible fallback boundary with the same backend contract."""

    engine_name = "bullet"
