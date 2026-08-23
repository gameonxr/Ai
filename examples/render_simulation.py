import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body import BodyLoader
from physics import MuJoCoBackend
from rendering import MatplotlibRenderer


if __name__ == "__main__":
    body = BodyLoader.load("config/body_humanoid.yaml")
    physics = MuJoCoBackend({})
    physics.load_body(body)
    physics.reset(seed=42)
    renderer = MatplotlibRenderer({"label_links": True})
    output = Path("/tmp/ai_body_simulator.png")
    renderer.render(body, physics.get_body_state(), output)
    renderer.close()
    physics.shutdown()
    print(f"Saved headless visualization to {output}")
