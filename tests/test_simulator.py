import pytest

from brain import DummyBrain
from simulator import Simulator


def test_simulator_rejects_non_mapping_environment_config(monkeypatch):
    from simulator.config_loader import ConfigLoader

    monkeypatch.setattr(ConfigLoader, "load", lambda self: {
        "simulator": {"timestep": 0.005},
        "physics": {"engine": "toy", "gravity": [0, 0, -9.81]},
        "body": {"config_path": "config/body_humanoid.yaml"},
        "sensors": {"sensors": {}},
        "actuators": {"defaults": {}},
        "environment": [],
    })
    with pytest.raises(ValueError, match="environment config must be a mapping"):
        Simulator("config/simulator_config.yaml")


def test_simulator_preserves_strict_environment_values(monkeypatch):
    from simulator.config_loader import ConfigLoader

    base = ConfigLoader("config/simulator_config.yaml").load()
    base["environment"] = {"floor_enabled": "false", "floor_friction": "0.5", "floor_size": [10, 10]}
    monkeypatch.setattr(ConfigLoader, "load", lambda self: base)
    with pytest.raises(ValueError, match="floor_enabled must be a boolean"):
        Simulator("config/simulator_config.yaml")


@pytest.mark.parametrize("rendering", [[], {"enabled": "false"}])
def test_simulator_rejects_invalid_rendering_configuration(monkeypatch, rendering):
    from simulator.config_loader import ConfigLoader

    base = ConfigLoader("config/simulator_config.yaml").load()
    base["rendering"] = rendering
    monkeypatch.setattr(ConfigLoader, "load", lambda self: base)
    message = "rendering config must be a mapping" if not isinstance(rendering, dict) else "rendering.enabled must be a boolean"
    with pytest.raises(ValueError, match=message):
        Simulator("config/simulator_config.yaml")


def test_get_state_exposes_world_definition():
    sim = Simulator("config/simulator_config.yaml")
    try:
        state = sim.get_state()
        assert state["world"]["floor_enabled"] is True
        assert state["world"]["objects"] == [{"type": "floor", "size": (10.0, 10.0), "friction": 0.5}]
    finally:
        sim.shutdown()


def test_get_state_exposes_disabled_floor_without_objects(monkeypatch):
    from simulator.config_loader import ConfigLoader

    base = ConfigLoader("config/simulator_config.yaml").load()
    base["environment"]["floor_enabled"] = False
    monkeypatch.setattr(ConfigLoader, "load", lambda self: base)
    sim = Simulator("config/simulator_config.yaml")
    try:
        state = sim.get_state()
        assert state["world"]["floor_enabled"] is False
        assert state["world"]["objects"] == []
    finally:
        sim.shutdown()


def test_floor_configuration_controls_contact_observations(monkeypatch):
    from simulator.config_loader import ConfigLoader

    base = ConfigLoader("config/simulator_config.yaml").load()
    base["environment"]["floor_enabled"] = False
    base["sensors"]["loaded"]["sensors"]["touch"]["enabled"] = True
    monkeypatch.setattr(ConfigLoader, "load", lambda self: base)
    simulator = Simulator("config/simulator_config.yaml")
    try:
        observation = simulator.step()
        assert observation.touch == {"contacts": []}
    finally:
        simulator.shutdown()


def test_simulator_set_brain_requires_interface_or_none():
    sim = Simulator("config/simulator_config.yaml")
    try:
        with pytest.raises(TypeError, match="brain must implement BrainInterface or be None"):
            sim.set_brain(object())  # type: ignore[arg-type]
        sim.set_brain(None)
    finally:
        sim.shutdown()


def test_simulator_reset_seed_requires_integer_or_null():
    sim = Simulator("config/simulator_config.yaml")
    try:
        for value in (True, 1.5, "7"):
            with pytest.raises(ValueError, match="seed must be an integer or null"):
                sim.reset(seed=value)  # type: ignore[arg-type]
    finally:
        sim.shutdown()


def test_simulator_step_requires_positive_integer():
    sim = Simulator("config/simulator_config.yaml")
    try:
        for value in (0, -1, True, 1.5, "2"):
            with pytest.raises(ValueError, match="n_steps must be a positive integer"):
                sim.step(value)  # type: ignore[arg-type]
    finally:
        sim.shutdown()


def test_shutdown_is_idempotent_and_guards_operations():
    sim = Simulator("config/simulator_config.yaml")
    sim.shutdown()
    sim.shutdown()
    for operation in (lambda: sim.reset(), lambda: sim.step(), lambda: sim.get_state(), lambda: sim.pause(), lambda: sim.resume()):
        with pytest.raises(RuntimeError, match="Simulator is shut down"):
            operation()


def test_integration_loop_and_pause():
    sim = Simulator("config/simulator_config.yaml")
    sim.set_brain(DummyBrain())
    sim.reset(seed=7)
    observation = sim.step(5)
    assert sim.step_count == 5 and observation.timestamp >= 0
    sim.pause(); assert sim.step() is None
    sim.resume(); sim.step(); assert sim.step_count == 6
    sim.shutdown()
