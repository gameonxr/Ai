import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import MultiAgentCoordinator
from simulator import Simulator
from training import RandomTorquePolicy


if __name__ == "__main__":
    agents = {}
    for agent_id, seed in (("alpha", 42), ("beta", 43)):
        simulator = Simulator("config/simulator_config.yaml")
        policy = RandomTorquePolicy(list(simulator.actuators), seed=seed)
        agents[agent_id] = (simulator, policy)
    coordinator = MultiAgentCoordinator(agents)
    coordinator.reset(seed=42)
    for _ in range(5):
        coordinator.step()
    print({agent_id: step.observation.timestamp for agent_id, step in coordinator.last_steps.items()})
    coordinator.close()
