from agents import MultiAgentCoordinator
from simulator import Simulator
from training import RandomTorquePolicy


def make_agents():
    result = {}
    for agent_id, seed in (("alpha", 1), ("beta", 2)):
        simulator = Simulator("config/simulator_config.yaml")
        result[agent_id] = (simulator, RandomTorquePolicy(list(simulator.actuators), seed=seed))
    return result


def test_multi_agent_steps_are_named_and_synchronized():
    coordinator = MultiAgentCoordinator(make_agents())
    observations = coordinator.reset(seed=10)
    steps = coordinator.step()
    assert set(observations) == {"alpha", "beta"}
    assert set(steps) == {"alpha", "beta"}
    assert steps["alpha"].observation.timestamp == steps["beta"].observation.timestamp
    assert coordinator.get_state()["alpha"]["step_count"] == 2
    coordinator.close()
