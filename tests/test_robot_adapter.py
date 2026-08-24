import pytest

from core import Action
from robot import SimulatedRobotAdapter
from simulator import Simulator


@pytest.mark.parametrize("config", [[], {"emergency_stop_on_disconnect": "true"}, {"robot_adapter": []}])
def test_simulated_robot_adapter_rejects_invalid_config(config):
    simulator = Simulator("config/simulator_config.yaml")
    try:
        with pytest.raises(ValueError):
            SimulatedRobotAdapter(simulator, config)  # type: ignore[arg-type]
    finally:
        simulator.shutdown()


def test_disconnect_emergency_stops_by_default():
    simulator = Simulator("config/simulator_config.yaml")
    adapter = SimulatedRobotAdapter(simulator, {"robot_adapter": {"emergency_stop_on_disconnect": True}})
    adapter.connect()
    adapter.send_action(Action(joint_targets={"neck": 0.1}))
    adapter.disconnect()
    assert simulator.paused and adapter.brain.pending.metadata["noop"]
    simulator.shutdown()


def test_disconnect_can_leave_simulator_running_when_disabled():
    simulator = Simulator("config/simulator_config.yaml")
    adapter = SimulatedRobotAdapter(simulator, {"emergency_stop_on_disconnect": False})
    adapter.connect()
    adapter.disconnect()
    assert not simulator.paused
    simulator.shutdown()


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
