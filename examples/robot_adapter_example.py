import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import Action
from robot import SimulatedRobotAdapter
from simulator import Simulator


if __name__ == "__main__":
    adapter = SimulatedRobotAdapter(Simulator("config/simulator_config.yaml"))
    adapter.connect()
    adapter.send_action(Action(joint_targets={"neck": 0.1}))
    observation = adapter.read_observation()
    print(f"Adapter observation timestamp: {observation.timestamp}")
    adapter.emergency_stop()
    adapter.resume_after_stop()
    adapter.disconnect()
    adapter.simulator.shutdown()
