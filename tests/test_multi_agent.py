from agents import MultiAgentCoordinator
import pytest

from simulator import Simulator
from training import RandomTorquePolicy


def make_agents():
    result = {}
    for agent_id, seed in (("alpha", 1), ("beta", 2)):
        simulator = Simulator("config/simulator_config.yaml")
        result[agent_id] = (simulator, RandomTorquePolicy(list(simulator.actuators), seed=seed))
    return result


def test_multi_agent_rejects_invalid_seed_and_guards_closed_state():
    coordinator = MultiAgentCoordinator(make_agents())
    with pytest.raises(ValueError, match="seed must be an integer or null"):
        coordinator.reset(seed=True)
    coordinator.close()
    coordinator.close()
    for operation in (lambda: coordinator.reset(), lambda: coordinator.step(), lambda: coordinator.get_state()):
        with pytest.raises(RuntimeError, match="MultiAgentCoordinator is closed"):
            operation()


def test_multi_agent_steps_are_named_and_synchronized():
    coordinator = MultiAgentCoordinator(make_agents())
    observations = coordinator.reset(seed=10)
    steps = coordinator.step()
    assert set(observations) == {"alpha", "beta"}
    assert set(steps) == {"alpha", "beta"}
    assert steps["alpha"].observation.timestamp == steps["beta"].observation.timestamp
    assert coordinator.get_state()["alpha"]["step_count"] == 2
    coordinator.close()
