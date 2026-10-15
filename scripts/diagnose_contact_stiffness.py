"""Exploratory, not preregistered: does OmniSim's contact stiffness move the cliff?

OmniSim gives every shape contact stiffness ke 2500 and damping kd 100 unless
the world or ``OMNISIM_NEWTON_CONTACT_KE`` / ``_KD`` says otherwise
(``src/omnisim/physics/omnisim_newton_runtime.py`` at v9.2.4). The MuJoCo
reference sets its contacts per geom, stiffer. This reruns the OmniSim expert
at box friction 0.08, where the two simulators disagree most (0.94 against
0.52), across a ladder of contact stiffness with everything else fixed.

Writes ``results/omnisim/contact_stiffness_diagnostic.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_omnisim import REPO, run_parallel  # noqa: E402

sys.path.insert(0, REPO)
from crosscheck import protocol  # noqa: E402

WORLD = "friction_0.08"
KE_LADDER = (2500, 10000, 40000, 160000)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    eval_seeds = [protocol.EVAL_SEEDS.start, protocol.EVAL_SEEDS.stop]
    out = {"world": WORLD, "contact_kd": 100, "expert": {}}
    for ke in KE_LADDER:
        tmp = os.path.join(REPO, "scratch", "contact_ke", str(ke))
        jobs = [{"name": "expert", "actor": "expert", "seeds": eval_seeds, "variant": WORLD}]
        status = run_parallel(WORLD, jobs, tmp, 3600, args.workers,
                              {"OMNISIM_NEWTON_CONTACT_KE": ke})
        if not status["ok"]:
            raise SystemExit(json.dumps(status, indent=1))
        with open(os.path.join(tmp, "expert.json")) as f:
            eps = json.load(f)["episodes"]
        out["expert"][str(ke)] = round(sum(e["success"] for e in eps) / len(eps), 3)
        print(f"ke {ke}: expert {out['expert'][str(ke)]}", flush=True)
    with open(os.path.join(REPO, "results", "omnisim", "contact_stiffness_diagnostic.json"),
              "w", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
