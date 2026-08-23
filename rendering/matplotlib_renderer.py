from __future__ import annotations

from pathlib import Path
from typing import Any

from .kinematics import project_body
from .renderer import Renderer


class MatplotlibRenderer(Renderer):
    """Headless 2D humanoid renderer for debugging and examples."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.width = int(self.config.get("width", 640))
        self.height = int(self.config.get("height", 640))
        self.dpi = int(self.config.get("dpi", 100))
        self.show_axes = bool(self.config.get("show_axes", True))
        self.last_figure = None

    def render(self, body, state: dict[str, Any], output_path: str | Path | None = None):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        points = project_body(body, state)
        figure, axis = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.4, 1.4)
        axis.set_ylim(-0.8, 2.0)
        axis.set_title("AI Body Simulator")
        if not self.show_axes:
            axis.axis("off")

        for joint in body.joints.values():
            if joint.parent in points and joint.child in points:
                x_values = [points[joint.parent][0], points[joint.child][0]]
                y_values = [points[joint.parent][1], points[joint.child][1]]
                axis.plot(x_values, y_values, color="#2563eb", linewidth=4, solid_capstyle="round")
        for name, (x, y) in points.items():
            axis.scatter([x], [y], color="#0f172a", s=30, zorder=3)
            if self.config.get("label_links", False):
                axis.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        figure.tight_layout()
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, format=output.suffix.lstrip(".") or "png")
        self.last_figure = figure
        return figure

    def close(self) -> None:
        if self.last_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.last_figure)
            self.last_figure = None
