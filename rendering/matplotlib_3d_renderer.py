from __future__ import annotations

from pathlib import Path
from typing import Any

import math

from .matplotlib_renderer import MatplotlibRenderer


_SEGMENTS_3D = {
    "torso": (0.0, 0.0, 1.0),
    "head": (0.0, 0.0, 0.38),
    "upper_arm_r": (-0.28, 0.0, 0.20),
    "lower_arm_r": (-0.30, 0.0, -0.28),
    "hand_r": (-0.14, 0.0, -0.18),
    "upper_arm_l": (0.28, 0.0, 0.20),
    "lower_arm_l": (0.30, 0.0, -0.28),
    "hand_l": (0.14, 0.0, -0.18),
    "upper_leg_r": (-0.16, 0.0, -0.45),
    "lower_leg_r": (-0.02, 0.0, -0.45),
    "foot_r": (0.10, 0.12, -0.12),
    "upper_leg_l": (0.16, 0.0, -0.45),
    "lower_leg_l": (0.02, 0.0, -0.45),
    "foot_l": (-0.10, 0.12, -0.12),
}


def _mat_vec(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _mat_mul(left: tuple[tuple[float, float, float], ...], right: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _axis_rotation(axis: list[float], angle: float) -> tuple[tuple[float, float, float], ...]:
    length = math.sqrt(sum(value * value for value in axis))
    if length == 0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z = (value / length for value in axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return (
        (cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine),
        (y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine),
        (z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine),
    )


def project_body_3d(body, state: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    """Project configured links into deterministic 3D debug points."""
    if not isinstance(state, dict):
        raise ValueError("Body state must be a mapping")
    positions = state.get("joint_positions", {})
    if not isinstance(positions, dict):
        raise ValueError("Body state joint_positions must be a mapping")
    normalized_positions: dict[str, float] = {}
    for name, value in positions.items():
        if isinstance(value, bool):
            raise ValueError(f"Body state joint position must be finite: {name}")
        try:
            angle = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Body state joint position must be numeric: {name}") from error
        if not math.isfinite(angle):
            raise ValueError(f"Body state joint position must be finite: {name}")
        normalized_positions[name] = angle

    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    points: dict[str, tuple[float, float, float]] = {"torso": (0.0, 0.0, 1.0)}
    rotations: dict[str, tuple[tuple[float, float, float], ...]] = {"torso": identity}
    parent_for = {joint.child: joint.parent for joint in body.joints.values()}
    joint_for_child = {joint.child: joint for joint in body.joints.values()}

    def point_for(link: str, visiting: set[str] | None = None) -> tuple[float, float, float]:
        if link in points:
            return points[link]
        visiting = visiting or set()
        if link in visiting or link not in _SEGMENTS_3D:
            return points.get(link, (0.0, 0.0, 0.0))
        visiting.add(link)
        parent = parent_for.get(link, "torso")
        parent_point = point_for(parent, visiting)
        parent_rotation = rotations.get(parent, identity)
        joint = joint_for_child.get(link)
        joint_rotation = _axis_rotation(joint.axis, normalized_positions.get(joint.name, 0.0)) if joint else identity
        world_rotation = _mat_mul(parent_rotation, joint_rotation)
        offset = _mat_vec(world_rotation, _SEGMENTS_3D[link])
        points[link] = tuple(parent_point[index] + offset[index] for index in range(3))
        rotations[link] = world_rotation
        return points[link]

    for link in body.links:
        point_for(link)
    return points


class Matplotlib3DRenderer(MatplotlibRenderer):
    """Optional headless 3D debug renderer; it is not a photorealistic body viewer."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.ground_z = self._finite_float(self.config.get("ground_z", self.config.get("ground_y", 0.0)), "ground_z")

    def render(self, body, state: dict[str, Any], output_path: str | Path | None = None):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        points = project_body_3d(body, state)
        figure = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        axis = figure.add_subplot(111, projection="3d")
        if self.auto_scale and points:
            self._set_auto_limits(axis, points, state)
        else:
            axis.set_xlim(-1.4, 1.4)
            axis.set_ylim(-1.4, 1.4)
            axis.set_zlim(-0.8, 2.0)
        floor_size = self._floor_size(state)
        if self.show_ground and floor_size is not None:
            half_x, half_y = floor_size[0] / 2.0, floor_size[1] / 2.0
            axis.plot_surface(
                np.array([[-half_x, half_x], [-half_x, half_x]]),
                np.array([[-half_y, -half_y], [half_y, half_y]]),
                np.array([[self.ground_z, self.ground_z], [self.ground_z, self.ground_z]]),
                color="#94a3b8",
                alpha=0.25,
                linewidth=0,
            )
        if self.show_grid:
            axis.grid(True, color="#cbd5e1", linewidth=0.7, alpha=0.6)
        else:
            axis.grid(False)
        axis.set_title("AI Body Simulator — 3D Debug View")
        if not self.show_axes:
            axis.set_axis_off()
        for joint in body.joints.values():
            if joint.parent in points and joint.child in points:
                parent = points[joint.parent]
                child = points[joint.child]
                link = body.links.get(joint.child)
                axis.plot(
                    [parent[0], child[0]],
                    [parent[1], child[1]],
                    [parent[2], child[2]],
                    color="#2563eb",
                    linewidth=self._link_linewidth(link),
                )
        for name, (x, y, z) in points.items():
            axis.scatter([x], [y], [z], color="#0f172a", s=self._joint_marker_size(body.links.get(name)), depthshade=True)
            if self.label_links:
                axis.text(x, y, z, name, fontsize=7)
        figure.tight_layout()
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, format=output.suffix.lstrip(".") or "png")
        self.last_figure = figure
        return figure

    def _set_auto_limits(self, axis, points: dict[str, tuple[float, float, float]], state: dict[str, Any]) -> None:
        values = [[point[index] for point in points.values()] for index in range(3)]
        floor_size = self._floor_size(state)
        if self.show_ground and floor_size is not None:
            values[0].extend([-floor_size[0] / 2.0, floor_size[0] / 2.0])
            values[1].extend([-floor_size[1] / 2.0, floor_size[1] / 2.0])
            values[2].append(self.ground_z)
        limits = []
        for axis_values in values:
            minimum, maximum = min(axis_values), max(axis_values)
            span = max(maximum - minimum, 0.5)
            margin = max(span * self.padding, 0.1)
            limits.append((minimum - margin, maximum + margin))
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_zlim(*limits[2])
        axis.set_box_aspect((1.0, 1.0, 1.2))

    @staticmethod
    def _link_linewidth(link) -> float:
        if link is None:
            return 4.0
        if link.collision_shape == "capsule":
            radius = Matplotlib3DRenderer._positive_property(link.properties.get("radius"), 0.05)
            return min(9.0, max(2.5, 100.0 * radius))
        if link.collision_shape == "sphere":
            radius = Matplotlib3DRenderer._positive_property(link.properties.get("radius"), 0.1)
            return min(9.0, max(3.0, 24.0 * radius))
        dimensions = link.properties.get("dimensions")
        if isinstance(dimensions, (list, tuple)) and dimensions:
            largest = max(Matplotlib3DRenderer._positive_property(value, 0.1) for value in dimensions)
            return min(9.0, max(3.0, 12.0 * largest))
        return 4.0

    @staticmethod
    def _joint_marker_size(link) -> float:
        if link is None:
            return 30.0
        if link.collision_shape == "sphere":
            radius = Matplotlib3DRenderer._positive_property(link.properties.get("radius"), 0.1)
            return min(180.0, max(30.0, 900.0 * radius))
        dimensions = link.properties.get("dimensions")
        if isinstance(dimensions, (list, tuple)) and dimensions:
            largest = max(Matplotlib3DRenderer._positive_property(value, 0.1) for value in dimensions)
            return min(180.0, max(30.0, 220.0 * largest))
        radius = Matplotlib3DRenderer._positive_property(link.properties.get("radius"), 0.05)
        return min(180.0, max(30.0, 700.0 * radius))

    @staticmethod
    def _positive_property(value: Any, fallback: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            return fallback
        return float(value)

    @staticmethod
    def _floor_size(state: dict[str, Any]) -> tuple[float, float] | None:
        world = state.get("world")
        if not isinstance(world, dict) or world.get("floor_enabled") is not True:
            return None
        for object_definition in world.get("objects", []):
            if isinstance(object_definition, dict) and object_definition.get("type") == "floor":
                size = object_definition.get("size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    return float(size[0]), float(size[1])
        return None

    @staticmethod
    def _finite_float(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Renderer {name} must be finite")
        return float(value)
