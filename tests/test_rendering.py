from pathlib import Path

import pytest

from body import BodyLoader
from physics import MuJoCoBackend
from rendering import MatplotlibRenderer


def test_renderer_auto_scales_to_body_points():
    body = BodyLoader.load("config/body_humanoid.yaml")
    engine = MuJoCoBackend({})
    engine.load_body(body)
    renderer = MatplotlibRenderer({"padding": 0.1})
    figure = renderer.render(body, engine.get_body_state())
    x_limits = figure.axes[0].get_xlim()
    y_limits = figure.axes[0].get_ylim()
    assert x_limits[0] < -0.65 and x_limits[1] > 0.65
    assert y_limits[0] < -0.1 and y_limits[1] < 2.0
    renderer.close()
    engine.shutdown()


def test_renderer_rejects_invalid_padding():
    with pytest.raises(ValueError, match="padding must be finite and non-negative"):
        MatplotlibRenderer({"padding": -0.1})


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
