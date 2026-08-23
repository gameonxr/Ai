import pytest

from brain import DummyBrain
from simulator import Simulator


def test_simulator_step_requires_positive_integer():
    sim = Simulator("config/simulator_config.yaml")
    try:
        for value in (0, -1, True, 1.5, "2"):
            with pytest.raises(ValueError, match="n_steps must be a positive integer"):
                sim.step(value)  # type: ignore[arg-type]
    finally:
        sim.shutdown()


def test_integration_loop_and_pause():
    sim = Simulator("config/simulator_config.yaml")
    sim.set_brain(DummyBrain())
    sim.reset(seed=7)
    observation = sim.step(5)
    assert sim.step_count == 5 and observation.timestamp >= 0
    sim.pause(); assert sim.step() is None
    sim.resume(); sim.step(); assert sim.step_count == 6
    sim.shutdown()
