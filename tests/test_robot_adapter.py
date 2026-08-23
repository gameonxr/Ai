import pytest

from core import Action
from robot import SimulatedRobotAdapter
from simulator import Simulator


def test_simulated_robot_adapter_contract():
    simulator = Simulator("config/simulator_config.yaml")
    adapter = SimulatedRobotAdapter(simulator)
    with pytest.raises(RuntimeError):
        adapter.read_observation()
    adapter.connect()
    adapter.send_action(Action(joint_targets={"neck": 0.1}))
    observation = adapter.read_observation()
    assert adapter.connected and observation.timestamp == 0.0
    adapter.emergency_stop()
    assert adapter.simulator.paused
    adapter.resume_after_stop()
    adapter.disconnect()
    assert not adapter.connected
    assert simulator.brain is None
    simulator.shutdown()
