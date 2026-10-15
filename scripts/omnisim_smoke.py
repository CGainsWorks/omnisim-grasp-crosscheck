"""Short in-engine check against the committed OmniSim results.

Runs the expert and the first committed clone on the first few evaluation seeds
of the nominal world and compares each episode with the committed outcome.
OmniSim's CPU solver is deterministic run to run on one machine; across
machines the floating point can differ, so a small number of flipped episodes
is allowed and reported rather than failed.

    python scripts/omnisim_smoke.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_omnisim import REPO, run_parallel  # noqa: E402

RESULTS = os.path.join(REPO, "results", "omnisim")


def committed(name: str) -> dict:
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return {e["seed"]: e["success"] for e in json.load(f)["episodes"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-flips", type=int, default=2)
    args = p.parse_args()
    seeds = [10_000, 10_000 + args.episodes]
    policy = os.path.join(RESULTS, "policies", "bc_s0.npz")
    checks = {"expert_nominal": "expert", "bc_s0_nominal": policy}
    jobs = [{"name": n, "actor": a, "seeds": seeds, "variant": "nominal"}
            for n, a in checks.items()]
    with tempfile.TemporaryDirectory(dir=os.path.join(REPO, "scratch")
                                     if os.path.isdir(os.path.join(REPO, "scratch"))
                                     else None) as out:
        status = run_parallel("nominal", jobs, out, 1800, args.workers)
        if not status["ok"]:
            print(json.dumps(status, indent=1))
            return 1
        failed = False
        for name in checks:
            with open(os.path.join(out, name + ".json")) as f:
                got = {e["seed"]: e["success"] for e in json.load(f)["episodes"]}
            want = committed(name)
            flips = [s for s in got if got[s] != want[s]]
            rate = sum(got.values()) / len(got)
            print(f"{name}: {rate:.2f} over {len(got)}, {len(flips)} differ from committed")
            failed |= len(flips) > args.max_flips
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
