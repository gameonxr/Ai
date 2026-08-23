from brain import DummyBrain
from simulator import Simulator


def test_integration_loop_and_pause():
    sim = Simulator("config/simulator_config.yaml")
    sim.set_brain(DummyBrain())
    sim.reset(seed=7)
    observation = sim.step(5)
    assert sim.step_count == 5 and observation.timestamp >= 0
    sim.pause(); assert sim.step() is None
    sim.resume(); sim.step(); assert sim.step_count == 6
    sim.shutdown()
