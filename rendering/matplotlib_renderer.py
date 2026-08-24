from __future__ import annotations

from pathlib import Path
from typing import Any

import math

from .kinematics import project_body
from .renderer import Renderer


class MatplotlibRenderer(Renderer):
    """Headless 2D humanoid renderer for debugging and examples."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.width = self._positive_int(self.config.get("width", 640), "width")
        self.height = self._positive_int(self.config.get("height", 640), "height")
        self.dpi = self._positive_int(self.config.get("dpi", 100), "dpi")
        self.show_axes = bool(self.config.get("show_axes", True))
        self.auto_scale = bool(self.config.get("auto_scale", True))
        self.padding = float(self.config.get("padding", 0.15))
        self.show_ground = bool(self.config.get("show_ground", True))
        self.show_grid = bool(self.config.get("show_grid", False))
        self.ground_y = float(self.config.get("ground_y", 0.0))
        if not math.isfinite(self.padding) or self.padding < 0:
            raise ValueError("Renderer padding must be finite and non-negative")
        if not math.isfinite(self.ground_y):
            raise ValueError("Renderer ground_y must be finite")
        self.last_figure = None

    def render(self, body, state: dict[str, Any], output_path: str | Path | None = None):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        points = project_body(body, state)
        figure, axis = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        axis.set_aspect("equal", adjustable="box")
        if self.auto_scale and points:
            x_values = [point[0] for point in points.values()]
            y_values = [point[1] for point in points.values()]
            self._set_auto_limits(axis, x_values, y_values)
        else:
            axis.set_xlim(-1.4, 1.4)
            axis.set_ylim(-0.8, 2.0)
        if self.show_ground:
            axis.axhline(self.ground_y, color="#94a3b8", linewidth=1.2, linestyle="--", label="ground", zorder=0)
        if self.show_grid:
            axis.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.6)
            axis.set_axisbelow(True)
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

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Renderer {name} must be a positive integer")
        return value

    def _set_auto_limits(self, axis, x_values: list[float], y_values: list[float]) -> None:
        def limits(values: list[float]) -> tuple[float, float]:
            minimum = min(values)
            maximum = max(values)
            span = max(maximum - minimum, 0.5)
            margin = max(span * self.padding, 0.1)
            return minimum - margin, maximum + margin

        axis.set_xlim(*limits(x_values))
        axis.set_ylim(*limits(y_values))

    def close(self) -> None:
        if self.last_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.last_figure)
            self.last_figure = None
