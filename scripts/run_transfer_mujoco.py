"""Run the OmniSim-trained clones in the MuJoCo reference.

The other direction (MuJoCo clones in OmniSim) is part of
``scripts/run_omnisim_eval.py``. Writes ``results/mujoco/transfer.json``.
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
from crosscheck.bc import MLPPolicy  # noqa: E402
from crosscheck.mujoco_env import MujocoLiftEnv  # noqa: E402


def main() -> None:
    env = MujocoLiftEnv()
    out = {"omnisim_clones_in_mujoco": {}}
    for world in ("nominal", "friction_0.07"):
        rates = []
        for k in protocol.BC_SEEDS:
            policy = MLPPolicy.load(os.path.join(REPO, "results", "omnisim", "policies",
                                                 f"bc_s{k}.npz"))
            rates.append(round(evaluate(env, policy, world, protocol.EVAL_SEEDS)["success"], 3))
        out["omnisim_clones_in_mujoco"][world] = rates
        print(world, rates, flush=True)
    with open(os.path.join(REPO, "results", "mujoco", "transfer.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
