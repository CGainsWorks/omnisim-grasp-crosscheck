"""Scripted expert for the lift task: the source of every demonstration.

A four-phase state machine ported from ``ml-grasp-rl-imitation-sim`` with the
wrist and handle branches removed. It reads only the grip point, the object
position and the hold point and it runs unchanged in both simulators.

    APPROACH  move above the box, fingers open
    DESCEND   drop to the grasp height, fingers open
    CLOSE     squeeze for a fixed number of steps
    LIFT      rise to the hold point and stay there

It is deliberately imperfect: it never re-plans and never checks that the
grasp took before lifting, so a change that makes grasping harder shows up in
its success rate as well as in the clone's.
"""

from __future__ import annotations

import numpy as np

from crosscheck.task import GOAL_POS, GRIP_POS, OBJ_POS

APPROACH, DESCEND, CLOSE, LIFT = range(4)

GRASP_Z_OFFSET = 0.004
APPROACH_HEIGHT = 0.085
SPEED_CAP = {APPROACH: 1.0, DESCEND: 0.35, CLOSE: 0.20, LIFT: 0.55}


class ScriptedExpert:
    def __init__(self, kp: float = 8.0, lift_kp: float = 5.0, close_steps: int = 6,
                 approach_tol: float = 0.020, descend_tol: float = 0.012,
                 filter_alpha: float = 0.5, phase_timeout: tuple = (45, 30),
                 noise: float = 0.0, rng: np.random.Generator | None = None) -> None:
        self.kp = kp
        self.lift_kp = lift_kp
        self.close_steps = close_steps
        self.approach_tol = approach_tol
        self.descend_tol = descend_tol
        self.filter_alpha = filter_alpha
        self.phase_timeout = phase_timeout
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.phase = APPROACH
        self._phase_steps = 0
        self._close_counter = 0
        self._filtered = None

    def _advance(self, phase: int) -> None:
        self.phase = phase
        self._phase_steps = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if self._filtered is None:
            self._filtered = obs.copy()
        else:
            a = self.filter_alpha
            self._filtered = a * obs + (1.0 - a) * self._filtered
        grip = self._filtered[GRIP_POS]
        obj = self._filtered[OBJ_POS]
        goal = obs[GOAL_POS]
        self._phase_steps += 1

        if self.phase == APPROACH:
            target = np.array([obj[0], obj[1], obj[2] + APPROACH_HEIGHT])
            grip_cmd = -1.0
            if (np.linalg.norm(target - grip) < self.approach_tol
                    or self._phase_steps > self.phase_timeout[0]):
                self._advance(DESCEND)
        elif self.phase == DESCEND:
            target = np.array([obj[0], obj[1], obj[2] + GRASP_Z_OFFSET])
            grip_cmd = -1.0
            if (abs(target[2] - grip[2]) < self.descend_tol
                    or self._phase_steps > self.phase_timeout[1]):
                self._advance(CLOSE)
        elif self.phase == CLOSE:
            target = np.array([obj[0], obj[1], obj[2] + GRASP_Z_OFFSET])
            grip_cmd = 1.0
            self._close_counter += 1
            if self._close_counter >= self.close_steps:
                self._advance(LIFT)
        else:
            target = grip + (goal - obj)
            grip_cmd = 1.0

        gain = self.lift_kp if self.phase == LIFT else self.kp
        cap = SPEED_CAP[self.phase]
        delta = np.clip(gain * (target - grip), -cap, cap)
        action = np.concatenate([delta, [grip_cmd]])
        if self.noise > 0.0:
            action = action + self.rng.normal(0.0, self.noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
