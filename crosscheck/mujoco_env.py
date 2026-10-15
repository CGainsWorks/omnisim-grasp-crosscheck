"""MuJoCo reference environment for the lift task (``scenes/lift.xml``)."""

from __future__ import annotations

import os

import mujoco
import numpy as np

from crosscheck import task
from crosscheck.variants import VARIANTS

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scenes", "lift.xml")


class MujocoLiftEnv:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_substeps = int(round(task.CONTROL_DT / m.opt.timestep))
        self._gantry_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                          for n in ("gx", "gy", "gz")]
        self._finger_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                          for n in ("left_slide", "right_slide")]
        self._obj_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")]
        self._obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object")
        self._grip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._nominal_mass = float(m.body_mass[self._obj_body])
        self._nominal_inertia = m.body_inertia[self._obj_body].copy()
        self._nominal_friction = m.geom_friction[self._obj_geom].copy()

    def _apply_variant(self, variant: str) -> None:
        v = VARIANTS[variant]
        m = self.model
        scale = v["box_mass"] / self._nominal_mass
        m.body_mass[self._obj_body] = v["box_mass"]
        m.body_inertia[self._obj_body] = self._nominal_inertia * scale
        m.geom_friction[self._obj_geom] = [v["box_friction"], *self._nominal_friction[1:]]

    def _grip(self) -> np.ndarray:
        return self.data.site_xpos[self._grip_site].copy()

    def _opening(self) -> float:
        q = self.data.qpos
        return task.OPEN_GAP - q[self._finger_q[0]] - q[self._finger_q[1]]

    def _obs(self) -> np.ndarray:
        d = self.data
        grip = self._grip()
        obj_t = mujoco.mjtObj
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, obj_t.mjOBJ_SITE, self._grip_site, vel, 0)
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, obj_t.mjOBJ_BODY, self._obj_body, obj_vel, 0)
        opening = self._opening()
        rate = (opening - self._prev_opening) / task.CONTROL_DT
        self._prev_opening = opening
        return task.build_obs(grip, vel[3:], opening, rate, d.xpos[self._obj_body],
                              obj_vel[3:], self._goal)

    def reset(self, init: dict, variant: str) -> np.ndarray:
        self._apply_variant(variant)
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        carriage = init["grip"] + np.array([0.0, 0.0, task.GRIP_BELOW_PALM])
        for i, q in enumerate(self._gantry_q):
            d.qpos[q] = carriage[i]
        d.ctrl[:3] = carriage
        d.ctrl[3:5] = 0.0
        ox, oy = init["box_xy"]
        yaw = init["box_yaw"]
        d.qpos[self._obj_q:self._obj_q + 3] = [ox, oy, task.TABLE_Z + task.BOX_HALF + 0.001]
        d.qpos[self._obj_q + 3:self._obj_q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, d)
        self._setpoint = init["grip"].copy()
        self._goal = task.goal_for(init["box_xy"])
        self._prev_opening = self._opening()
        return self._obs()

    def step(self, action: np.ndarray) -> np.ndarray:
        self._setpoint = task.next_setpoint(self._setpoint, action)
        d = self.data
        d.ctrl[:3] = self._setpoint + np.array([0.0, 0.0, task.GRIP_BELOW_PALM])
        d.ctrl[3:5] = task.finger_command(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, d)
        return self._obs()
