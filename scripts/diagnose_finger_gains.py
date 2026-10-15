"""Exploratory, not preregistered: does OmniSim's finger gain explain the moved cliff?

OmniSim gives a position-controlled slider kp = 10 x maxForce and kv = 0.5 x
maxForce (``src/omnisim/nodes/OmBasicJoint.cpp`` at v9.2.4), so the 60 N
fingers get 600 and 30, against 300 and none in ``scenes/lift.xml``. This
reruns the expert over the friction ladder in MuJoCo with OmniSim's finger
gains and nothing else changed. The expert is used because it involves no
training, so the comparison has one variable.

Writes ``results/mujoco/finger_gain_diagnostic.json``.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from run_mujoco import evaluate  # noqa: E402

from crosscheck import protocol  # noqa: E402
from crosscheck.expert import ScriptedExpert  # noqa: E402
from crosscheck.mujoco_env import MujocoLiftEnv  # noqa: E402

LADDER = ["nominal", "friction_0.30", "friction_0.15", "friction_0.08", "friction_0.07",
          "friction_0.06"]
OMNISIM_FINGER_GAINS = (600.0, 30.0)


def main() -> None:
    env = MujocoLiftEnv(finger_gains=OMNISIM_FINGER_GAINS)
    out = {"finger_gains": {"kp": OMNISIM_FINGER_GAINS[0], "kv": OMNISIM_FINGER_GAINS[1]},
           "expert": {}}
    for world in LADDER:
        r = evaluate(env, ScriptedExpert(), world, protocol.EVAL_SEEDS)
        out["expert"][world] = round(r["success"], 3)
        print(world, out["expert"][world], flush=True)
    with open(os.path.join(REPO, "results", "mujoco", "finger_gain_diagnostic.json"), "w",
              newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
