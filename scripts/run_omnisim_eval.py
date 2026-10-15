"""The OmniSim half of the experiment, end to end.

    python scripts/run_omnisim_eval.py --stage demos      # expert demos, nominal
    python scripts/run_omnisim_eval.py --stage train      # five clones (no simulator)
    python scripts/run_omnisim_eval.py --stage eval       # every world in the protocol

The eval stage runs, in each world, the expert and the five OmniSim-trained
clones on the 100 held-out seeds. In the nominal world and the preregistered
one it also runs the five MuJoCo-trained clones, which asks whether a policy
cloned in one simulator works in the other.

Results go to ``results/omnisim/``; per-worker engine logs stay in ``scratch/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_omnisim import REPO, run_parallel  # noqa: E402

sys.path.insert(0, REPO)
from crosscheck import protocol  # noqa: E402

OUT = os.path.join(REPO, "results", "omnisim")
SCRATCH = os.path.join(REPO, "scratch", "omnisim_eval")
WORLDS = ["nominal", "friction_0.07", "friction_0.30", "friction_0.15", "friction_0.08",
          "friction_0.06", "mass_x10"]
TRANSFER_WORLDS = ("nominal", "friction_0.07")
EVAL = [protocol.EVAL_SEEDS.start, protocol.EVAL_SEEDS.stop]


def compact(src: str, dst: str) -> None:
    """Keep what the tables need: seed, outcome, peak lift, final box position."""
    with open(src) as f:
        data = json.load(f)
    keep = ("seed", "success", "peak_lift", "final_obj", "settle_error")
    episodes = [{k: (round(v, 5) if isinstance(v, float) else
                     [round(x, 5) for x in v] if isinstance(v, list) else v)
                 for k, v in e.items() if k in keep} for e in data["episodes"]]
    job = dict(data["job"])
    if not job["actor"].startswith("expert"):
        # Repository-relative and POSIX: an absolute host path would leak the
        # machine's layout into committed results and break on another OS.
        job["actor"] = os.path.relpath(job["actor"], REPO).replace(os.sep, "/")
    with open(dst, "w", newline="\n") as f:
        json.dump({"job": job, "episodes": episodes}, f, indent=0)
        f.write("\n")


def stage_demos(workers: int) -> None:
    jobs = [{"name": "demos", "actor": "expert_demo",
             "seeds": [protocol.DEMO_SEEDS.start, protocol.DEMO_SEEDS.stop], "variant": "nominal"}]
    tmp = os.path.join(SCRATCH, "demos")
    status = run_parallel("nominal", jobs, tmp, 3600, workers)
    if not status["ok"]:
        raise SystemExit(json.dumps(status, indent=1))
    os.makedirs(OUT, exist_ok=True)
    shutil.copy2(os.path.join(tmp, "demos_demos.npz"), os.path.join(OUT, "demos.npz"))
    compact(os.path.join(tmp, "demos.json"), os.path.join(OUT, "demos_episodes.json"))


def stage_train() -> None:
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "train_bc.py"),
                    "--demos", os.path.join(OUT, "demos.npz"),
                    "--out", os.path.join(OUT, "policies")], check=True)


def stage_eval(workers: int, worlds: list) -> None:
    for world in worlds:
        jobs = [{"name": f"expert_{world}", "actor": "expert", "seeds": EVAL, "variant": world}]
        for k in protocol.BC_SEEDS:
            jobs.append({"name": f"bc_s{k}_{world}", "variant": world, "seeds": EVAL,
                         "actor": os.path.join(OUT, "policies", f"bc_s{k}.npz")})
            if world in TRANSFER_WORLDS:
                jobs.append({"name": f"mujoco_bc_s{k}_{world}", "variant": world, "seeds": EVAL,
                             "actor": os.path.join(REPO, "results", "mujoco", "policies",
                                                   f"bc_s{k}.npz")})
        tmp = os.path.join(SCRATCH, world)
        status = run_parallel(world, jobs, tmp, 7200, workers)
        if not status["ok"]:
            raise SystemExit(json.dumps(status, indent=1))
        for job in jobs:
            compact(os.path.join(tmp, job["name"] + ".json"),
                    os.path.join(OUT, job["name"] + ".json"))
        print(f"{world}: done", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["demos", "train", "eval"], required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--worlds", nargs="+", default=WORLDS)
    args = p.parse_args()
    if args.stage == "demos":
        stage_demos(args.workers)
    elif args.stage == "train":
        stage_train()
    else:
        stage_eval(args.workers, args.worlds)


if __name__ == "__main__":
    main()
