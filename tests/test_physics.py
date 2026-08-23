from body import BodyLoader
from core import Action
from physics import MuJoCoBackend


def test_backend_is_deterministic():
    body = BodyLoader.load("config/body_humanoid.yaml")
    first, second = MuJoCoBackend({}), MuJoCoBackend({})
    for engine in (first, second):
        engine.load_body(body); engine.reset(seed=42); engine.apply_action(Action(joint_targets={"neck": 0.2})); engine.step()
    assert first.get_body_state() == second.get_body_state()
