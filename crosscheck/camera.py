"""One camera pose for both simulators, so their frames can sit side by side.

Rendering is for the media in the README only. The camera has no collider and
no mass. In OmniSim it lives only in the ``_render`` worlds; in MuJoCo it is a
free viewer camera set from the same pose, so ``scenes/lift.xml`` is unchanged
and no result in ``results/`` depends on it.
"""

from __future__ import annotations

import numpy as np

POSITION = np.array([0.42, -0.42, 0.68])
TARGET = np.array([0.0, 0.0, 0.53])
WIDTH, HEIGHT = 480, 360
HFOV = 0.85  # horizontal field of view, radians


def _basis():
    """Forward, left and up unit vectors of a level camera looking at TARGET."""
    forward = TARGET - POSITION
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    return forward, left, up


def omnisim_rotation() -> tuple:
    """Axis-angle for an OmniSim (Webots FLU) camera: x forward, y left, z up."""
    forward, left, up = _basis()
    r = np.column_stack([forward, left, up])
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    axis /= np.linalg.norm(axis)
    return (*axis, angle)


def mujoco_free_camera() -> dict:
    """Lookat, distance, azimuth and elevation for a MuJoCo free camera."""
    d = TARGET - POSITION
    dist = float(np.linalg.norm(d))
    return {"lookat": TARGET.tolist(), "distance": dist,
            "azimuth": float(np.degrees(np.arctan2(d[1], d[0]))),
            "elevation": float(np.degrees(np.arcsin(d[2] / dist)))}


def mujoco_fovy_degrees() -> float:
    return float(np.degrees(2.0 * np.arctan(np.tan(HFOV / 2.0) * HEIGHT / WIDTH)))
