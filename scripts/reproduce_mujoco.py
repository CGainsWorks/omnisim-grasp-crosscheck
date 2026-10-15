"""Re-run part of the MuJoCo reference and compare it, episode by episode.

Runs the expert and the first committed clone on the first 25 held-out seeds
of the nominal world and of the preregistered one and compares each episode's
outcome with ``results/mujoco/results.json``. MuJoCo is not bit-reproducible
across platforms and the preregistered world sits on a friction cliff where a
last-digit difference can flip an episode, so a few flips are allowed there.
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
from crosscheck.expert import ScriptedExpert  # noqa: E402
from crosscheck.mujoco_env import MujocoLiftEnv  # noqa: E402

EPISODES = 25
MAX_FLIPS = {"nominal": 0, "friction_0.07": 4}


def main() -> int:
    with open(os.path.join(REPO, "results", "mujoco", "results.json")) as f:
        committed = json.load(f)
    env = MujocoLiftEnv()
    seeds = list(protocol.EVAL_SEEDS)[:EPISODES]
    policy = MLPPolicy.load(os.path.join(REPO, "results", "mujoco", "policies", "bc_s0.npz"))
    failed = False
    for world, limit in MAX_FLIPS.items():
        for name, actor, want in (
                ("expert", ScriptedExpert(), committed["expert"][world]["outcomes"]),
                ("clone s0", policy, committed["bc"]["s0"][world]["outcomes"])):
            got = evaluate(env, actor, world, seeds)["outcomes"]
            flips = sum(a != b for a, b in zip(got, want[:EPISODES]))
            print(f"{world:14s} {name:8s} {got.count('1')}/{EPISODES}, "
                  f"{flips} differ from committed (allowed {limit})")
            failed |= flips > limit
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
