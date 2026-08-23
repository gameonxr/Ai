from __future__ import annotations

import math
from typing import Any


# A stable, readable 2D projection for Phase 2 headless visualization.
_SEGMENTS = {
    "torso": (0.0, 1.0, 0.0),
    "head": (0.0, 0.38, 0.0),
    "upper_arm_r": (-0.28, 0.20, -0.25),
    "lower_arm_r": (-0.30, -0.28, 0.0),
    "hand_r": (-0.14, -0.18, 0.0),
    "upper_arm_l": (0.28, 0.20, 0.25),
    "lower_arm_l": (0.30, -0.28, 0.0),
    "hand_l": (0.14, -0.18, 0.0),
    "upper_leg_r": (-0.16, -0.45, 0.0),
    "lower_leg_r": (-0.02, -0.45, 0.0),
    "foot_r": (0.10, -0.12, 0.0),
    "upper_leg_l": (0.16, -0.45, 0.0),
    "lower_leg_l": (0.02, -0.45, 0.0),
    "foot_l": (-0.10, -0.12, 0.0),
}


def project_body(body, state: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Project configured links into deterministic 2D points for visualization."""
    positions = state.get("joint_positions", {})
    points: dict[str, tuple[float, float]] = {"torso": (0.0, 1.0)}
    parent_for = {joint.child: joint.parent for joint in body.joints.values()}
    joint_for_child = {joint.child: joint.name for joint in body.joints.values()}

    def point_for(link: str, visiting: set[str] | None = None) -> tuple[float, float]:
        if link in points:
            return points[link]
        visiting = visiting or set()
        if link in visiting or link not in _SEGMENTS:
            return points.get(link, (0.0, 0.0))
        visiting.add(link)
        parent = parent_for.get(link, "torso")
        px, py = point_for(parent, visiting)
        dx, dy, bias = _SEGMENTS[link]
        angle = float(positions.get(joint_for_child.get(link, ""), 0.0)) + bias
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x = px + dx * cos_a - dy * sin_a
        y = py + dx * sin_a + dy * cos_a
        points[link] = (x, y)
        return points[link]

    for link in body.links:
        point_for(link)
    return points
