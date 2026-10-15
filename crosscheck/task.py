"""The lift task, defined once and shared by both simulators.

Everything here is plain numpy so the same file runs on the host, in CI and
inside the OmniSim controller, which only has the interpreter the engine ships.

Geometry
--------
World z is up and the table top is at ``TABLE_Z``. A parallel-jaw hand rides a
three-axis gantry: three prismatic joints under position control carry the palm,
and two prismatic fingers close on the box. The gantry stands in for the mocap
weld of the original repository, because a position-servoed prismatic joint is
something both simulators model natively, so the two scenes describe the same
mechanism instead of two approximations of it.

Observation (20 dimensions)
---------------------------
=====  ====  ===============================================
Index  Size  Quantity
=====  ====  ===============================================
0:3     3    grip point position, world frame
3:6     3    grip point linear velocity
6       1    gripper opening (distance between the pad faces)
7       1    gripper opening rate
8:11    3    object position
11:14   3    object position relative to the grip point
14:17   3    object linear velocity
17:20   3    hold point position
=====  ====  ===============================================

Object orientation is left out on purpose. The expert never reads it, and the
original repository showed that a behaviour-cloned policy will lean on an input
its teacher ignores, which would add a confound to a cross-simulator test.

Action (4 dimensions, all in [-1, 1])
-------------------------------------
0:3 move the gantry setpoint by ``POS_STEP`` metres per unit; 3 is the gripper
command, -1 fully open and +1 fully closed.
"""

from __future__ import annotations

import numpy as np

# Scene geometry, in metres and kilograms.
TABLE_Z = 0.40
BOX_HALF = 0.022
BOX_MASS = 0.08
BOX_FRICTION = 1.0
PAD_FRICTION = 0.05          # below any box value, see scenes/lift.xml
FINGER_TRAVEL = 0.040        # each finger closes this far from fully open
OPEN_GAP = 0.078             # pad-face gap with both fingers fully open
GRIP_BELOW_PALM = 0.046      # grip point below the gantry carriage origin

# Control.
CONTROL_DT = 0.04            # seconds per policy step
MAX_STEPS = 100              # 4 s episodes
POS_STEP = 0.02              # setpoint change per unit action
HOLD_HEIGHT = 0.15           # hold point above the table
WORKSPACE_LOW = np.array([-0.25, -0.30, TABLE_Z + 0.02])
WORKSPACE_HIGH = np.array([0.25, 0.30, TABLE_Z + 0.35])

# Reset distribution.
SPAWN_XY = 0.08              # box centre uniform in +/- this (y is 1.2x wider)
SPAWN_YAW = np.pi / 4.0
HAND_OFFSET_XY = 0.06
HAND_Z_RANGE = (0.58, 0.66)

# Success: read at the final step only, so dropping the box late scores zero.
GOAL_TOLERANCE = 0.05
HELD_TOLERANCE = 0.03

OBS_DIM = 20
ACT_DIM = 4

GRIP_POS = slice(0, 3)
GRIP_VEL = slice(3, 6)
OPENING = 6
OPENING_RATE = 7
OBJ_POS = slice(8, 11)
OBJ_REL = slice(11, 14)
OBJ_VEL = slice(14, 17)
GOAL_POS = slice(17, 20)


def sample_reset(rng: np.random.Generator) -> dict:
    """Draw one episode's initial state. Both simulators consume the same draw."""
    ox = rng.uniform(-SPAWN_XY, SPAWN_XY)
    oy = rng.uniform(-SPAWN_XY * 1.2, SPAWN_XY * 1.2)
    yaw = rng.uniform(-SPAWN_YAW, SPAWN_YAW)
    hand_xy = np.array([ox, oy]) + rng.uniform(-HAND_OFFSET_XY, HAND_OFFSET_XY, size=2)
    hand_z = rng.uniform(*HAND_Z_RANGE)
    hand = np.clip(np.array([hand_xy[0], hand_xy[1], hand_z]), WORKSPACE_LOW, WORKSPACE_HIGH)
    return {
        "box_xy": np.array([ox, oy]),
        "box_yaw": float(yaw),
        "grip": hand,
    }


def goal_for(box_xy: np.ndarray) -> np.ndarray:
    return np.array([box_xy[0], box_xy[1], TABLE_Z + HOLD_HEIGHT])


def build_obs(grip, grip_vel, opening, opening_rate, obj, obj_vel, goal) -> np.ndarray:
    grip = np.asarray(grip, dtype=np.float64)
    obj = np.asarray(obj, dtype=np.float64)
    return np.concatenate([
        grip, np.asarray(grip_vel, dtype=np.float64),
        [float(opening), float(opening_rate)],
        obj, obj - grip, np.asarray(obj_vel, dtype=np.float64),
        np.asarray(goal, dtype=np.float64),
    ])


def is_success(obj, grip, goal) -> bool:
    obj = np.asarray(obj)
    return bool(np.linalg.norm(obj - np.asarray(goal)) < GOAL_TOLERANCE
                and np.linalg.norm(obj - np.asarray(grip)) < HELD_TOLERANCE)


def next_setpoint(setpoint: np.ndarray, action: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return np.clip(setpoint + a[:3] * POS_STEP, WORKSPACE_LOW, WORKSPACE_HIGH)


def finger_command(action: np.ndarray) -> float:
    """Map the gripper command in [-1, 1] to a per-finger closing distance."""
    g = float(np.clip(action[3], -1.0, 1.0))
    return 0.5 * (g + 1.0) * FINGER_TRAVEL
