from pathlib import Path

from body import BodyLoader
from physics import MuJoCoBackend
from rendering import MatplotlibRenderer


def test_headless_renderer_writes_png(tmp_path: Path):
    body = BodyLoader.load("config/body_humanoid.yaml")
    engine = MuJoCoBackend({})
    engine.load_body(body)
    output = tmp_path / "body.png"
    renderer = MatplotlibRenderer({"width": 320, "height": 320, "label_links": True})
    figure = renderer.render(body, engine.get_body_state(), output)
    assert figure is not None
    assert output.exists() and output.stat().st_size > 0
    renderer.close()
