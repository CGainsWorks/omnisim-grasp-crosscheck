"""OmniSim controller: runs a list of jobs against the lift world it is loaded in.

The engine starts this file as the gantry's supervisor controller. It wraps the
simulator in the same ``reset(init, variant)`` / ``step(action)`` interface as
``crosscheck/mujoco_env.py``, so the episode loop, the expert, the success test
and the policy code are the very files the MuJoCo reference runs.

Environment, set by ``scripts/run_omnisim.py``:

    CROSSCHECK_REPO   repository root inside the container (read only)
    CROSSCHECK_JOBS   JSON file with the job list
    CROSSCHECK_OUT    writable directory for results and demonstrations

A job is ``{"name", "actor", "seeds": [start, stop], "variant"}`` where actor is
``"expert"``, ``"expert_demo"`` (noisy expert, recorded) or a policy path.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.environ.get("CROSSCHECK_REPO", "/work/repo"))

from crosscheck import task  # noqa: E402
from crosscheck.bc import MLPPolicy  # noqa: E402
from crosscheck.expert import ScriptedExpert  # noqa: E402
from crosscheck.rollout import DEMO_NOISE, run_episode  # noqa: E402
from omnisim import Supervisor  # noqa: E402

HOME_Z = 0.70                 # must match omnisim/gen_worlds.py
CONTROL_MS = int(round(task.CONTROL_DT * 1000))
SETTLE_STEPS = 60             # control steps allowed to reach the start pose
SETTLE_TOL = 0.001


class OmniSimLiftEnv:
    def __init__(self, robot: Supervisor, variant: str) -> None:
        self.robot = robot
        self.variant = variant
        basic = int(robot.getBasicTimeStep())
        self.motors = [robot.getDevice(n) for n in ("gx", "gy", "gz", "finger_l", "finger_r")]
        self.sensors = [robot.getDevice(n + "_sensor")
                        for n in ("gx", "gy", "gz", "finger_l", "finger_r")]
        for s in self.sensors:
            s.enable(basic)
        self.hand = robot.getFromDef("HAND")
        self.box = robot.getFromDef("BOX")
        self.box_translation = self.box.getField("translation")
        self.box_rotation = self.box.getField("rotation")

    def _command(self, grip_setpoint: np.ndarray, finger: float) -> None:
        carriage = grip_setpoint + np.array([0.0, 0.0, task.GRIP_BELOW_PALM])
        self.motors[0].setPosition(float(carriage[0]))
        self.motors[1].setPosition(float(carriage[1]))
        self.motors[2].setPosition(float(carriage[2] - HOME_Z))
        self.motors[3].setPosition(float(finger))
        self.motors[4].setPosition(float(finger))

    def _grip(self) -> np.ndarray:
        return np.array(self.hand.getPosition()) - np.array([0.0, 0.0, task.GRIP_BELOW_PALM])

    def _opening(self) -> float:
        return task.OPEN_GAP - self.sensors[3].getValue() - self.sensors[4].getValue()

    def _obs(self) -> np.ndarray:
        opening = self._opening()
        rate = (opening - self._prev_opening) / task.CONTROL_DT
        self._prev_opening = opening
        return task.build_obs(self._grip(), self.hand.getVelocity()[:3], opening, rate,
                              self.box.getPosition(), self.box.getVelocity()[:3], self._goal)

    def _tick(self) -> None:
        if self.robot.step(CONTROL_MS) == -1:
            raise SystemExit(0)

    def reset(self, init: dict, variant: str) -> np.ndarray:
        if variant != self.variant:
            raise ValueError(f"world is {self.variant!r}, job asked for {variant!r}")
        ox, oy = init["box_xy"]
        # Writing a root Solid's pose also returns every joint to its default,
        # which parks the hand at the home pose, clear above the table.
        self.box_translation.setSFVec3f([float(ox), float(oy),
                                         task.TABLE_Z + task.BOX_HALF + 0.001])
        self.box_rotation.setSFRotation([0.0, 0.0, 1.0, float(init["box_yaw"])])
        self.box.setVelocity([0.0] * 6)
        self._command(init["grip"], 0.0)
        self._tick()
        # Drive to the start pose before the clock starts. MuJoCo writes the
        # start state directly; here it has to be reached, from above.
        for _ in range(SETTLE_STEPS):
            self._tick()
            err = np.linalg.norm(self._grip() - init["grip"])
            speed = np.linalg.norm(self.hand.getVelocity()[:3])
            if err < SETTLE_TOL and speed < 0.01:
                break
        self.settle_error = float(np.linalg.norm(self._grip() - init["grip"]))
        self._setpoint = init["grip"].copy()
        self._goal = task.goal_for(init["box_xy"])
        self._prev_opening = self._opening()
        return self._obs()

    def step(self, action: np.ndarray) -> np.ndarray:
        self._setpoint = task.next_setpoint(self._setpoint, action)
        self._command(self._setpoint, task.finger_command(action))
        self._tick()
        return self._obs()


def make_actor(spec: str, seed: int):
    if spec == "expert":
        return ScriptedExpert()
    if spec == "expert_demo":
        return ScriptedExpert(noise=DEMO_NOISE, rng=np.random.default_rng(seed))
    return MLPPolicy.load(spec)


def main() -> None:
    robot = Supervisor()
    out_dir = os.environ["CROSSCHECK_OUT"]
    os.makedirs(out_dir, exist_ok=True)
    variant = sys.argv[1] if len(sys.argv) > 1 else "nominal"
    status_path = os.path.join(out_dir, "status.json")
    try:
        with open(os.environ["CROSSCHECK_JOBS"], encoding="utf-8") as f:
            jobs = json.load(f)
        env = OmniSimLiftEnv(robot, variant)
        t0 = time.time()
        for job in jobs:
            episodes, demo_obs, demo_act = [], [], []
            policy = None if job["actor"].startswith("expert") else MLPPolicy.load(job["actor"])
            for seed in range(*job["seeds"]):
                actor = policy if policy is not None else make_actor(job["actor"], seed)
                record = job["actor"] == "expert_demo"
                ep = run_episode(env, actor, seed, job["variant"], record=record)
                ep["settle_error"] = env.settle_error
                if record:
                    if ep["success"]:
                        demo_obs.append(ep.pop("obs"))
                        demo_act.append(ep.pop("act"))
                    else:
                        ep.pop("obs")
                        ep.pop("act")
                episodes.append(ep)
            if demo_obs:
                np.savez_compressed(os.path.join(out_dir, job["name"] + "_demos.npz"),
                                    obs=np.concatenate(demo_obs), act=np.concatenate(demo_act))
            with open(os.path.join(out_dir, job["name"] + ".json"), "w", newline="\n") as f:
                json.dump({"job": job, "episodes": episodes}, f)
                f.write("\n")
            rate = np.mean([e["success"] for e in episodes])
            print(f"[lift_agent] {job['name']}: {rate:.3f} over {len(episodes)}", flush=True)
        with open(status_path, "w", newline="\n") as f:
            json.dump({"ok": True, "wall_seconds": round(time.time() - t0, 1),
                       "sim_seconds": round(robot.getTime(), 2)}, f)
        robot.simulationQuit(0)
    except Exception:
        with open(status_path, "w", newline="\n") as f:
            json.dump({"ok": False, "error": traceback.format_exc()}, f)
        traceback.print_exc()
        robot.simulationQuit(1)


main()
