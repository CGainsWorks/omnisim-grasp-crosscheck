"""Run a list of lift jobs inside the OmniSim container and copy the results out.

    python scripts/run_omnisim.py --variant nominal --jobs jobs.json --out results/omnisim/raw

OmniSim steps this scene at about 0.4x real time on one core, so ``--workers``
splits every job's seeds into contiguous chunks and runs one container per
chunk. The chunks are merged back in seed order, so the output does not depend
on the worker count.

The container gets no network (``--network none``), the repository read only
and a named volume for its output. A writable bind mount works on Docker
Desktop and fails on a Linux CI runner, so results come back with
``docker cp`` instead. The image is built from source by
``scripts/build_omnisim_image.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = os.environ.get("OMNISIM_IMAGE", "omnisim-local:v9.2.4")


def to_container(path: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel.startswith(".."):
        raise ValueError(f"{path} is outside the repository, so the container cannot see it")
    return "/work/repo/" + rel.replace(os.sep, "/")


def docker(*args: str, check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, **kw)


def run(variant: str, jobs: list, out: str, duration: int, extra_env: dict | None = None) -> dict:
    world_file = os.path.join(REPO, "omnisim", "worlds", f"lift_{variant}.wbt")
    if not os.path.exists(world_file):
        raise FileNotFoundError(f"{world_file} is missing; run omnisim/gen_worlds.py")
    for job in jobs:
        if not job["actor"].startswith("expert"):
            job["actor"] = to_container(job["actor"])
    stage = os.path.join(REPO, "scratch", "omnisim_jobs")
    os.makedirs(stage, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    jobs_path = os.path.join(stage, f"jobs_{variant}_{tag}.json")
    with open(jobs_path, "w", newline="\n") as f:
        json.dump(jobs, f, indent=1)
    volume = f"crosscheck-out-{tag}"
    docker("volume", "create", volume, stdout=subprocess.DEVNULL)
    t0 = time.time()
    try:
        world = f"/work/repo/omnisim/worlds/lift_{variant}.wbt"
        proc = docker(
            "run", "--rm", "--network", "none",
            "-v", f"{REPO}:/work/repo:ro", "-v", f"{volume}:/work/out",
            "-e", "CROSSCHECK_REPO=/work/repo",
            "-e", f"CROSSCHECK_JOBS={to_container(jobs_path)}",
            "-e", "CROSSCHECK_OUT=/work/out",
            "-e", "OMNISIM_LOG_PATH=/work/out/omnisim_log.txt",
            *[x for k, v in (extra_env or {}).items() for x in ("-e", f"{k}={v}")],
            IMAGE, "run-headless", world, "--duration", str(duration),
            check=False, capture_output=True, text=True)
        os.makedirs(out, exist_ok=True)
        holder = f"crosscheck-cp-{tag}"
        docker("create", "--name", holder, "-v", f"{volume}:/work/out", IMAGE,
               stdout=subprocess.DEVNULL)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                docker("cp", f"{holder}:/work/out/.", tmp)
                for name in os.listdir(tmp):
                    shutil.copy2(os.path.join(tmp, name), os.path.join(out, name))
        finally:
            docker("rm", holder, stdout=subprocess.DEVNULL)
    finally:
        docker("volume", "rm", "--force", volume, stdout=subprocess.DEVNULL, check=False)
        os.remove(jobs_path)
    with open(os.path.join(out, f"runner_{variant}.log"), "w", newline="\n") as f:
        f.write(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    status_path = os.path.join(out, "status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {"ok": False}
    status["runner_exit"] = proc.returncode
    status["host_seconds"] = round(time.time() - t0, 1)
    return status


def split(jobs: list, workers: int) -> list:
    """One job list per worker, each job cut to a contiguous chunk of its seeds."""
    plans = [[] for _ in range(workers)]
    for job in jobs:
        start, stop = job["seeds"]
        edges = np.linspace(start, stop, workers + 1).round().astype(int)
        for k in range(workers):
            if edges[k] < edges[k + 1]:
                plans[k].append({**job, "seeds": [int(edges[k]), int(edges[k + 1])]})
    return [p for p in plans if p]


def run_parallel(variant: str, jobs: list, out: str, duration: int, workers: int,
                 extra_env: dict | None = None) -> dict:
    plans = split(jobs, workers)
    dirs = [os.path.join(out, "workers", f"w{k}") for k in range(len(plans))]
    with ThreadPoolExecutor(len(plans)) as pool:
        statuses = list(pool.map(lambda a: run(variant, *a, duration, extra_env),
                                 zip(plans, dirs)))
    for job in jobs:
        episodes, obs, act = [], [], []
        for d in dirs:
            path = os.path.join(d, job["name"] + ".json")
            if os.path.exists(path):
                episodes += json.load(open(path))["episodes"]
            demo = os.path.join(d, job["name"] + "_demos.npz")
            if os.path.exists(demo):
                data = np.load(demo)
                obs.append(data["obs"])
                act.append(data["act"])
        episodes.sort(key=lambda e: e["seed"])
        with open(os.path.join(out, job["name"] + ".json"), "w", newline="\n") as f:
            json.dump({"job": job, "episodes": episodes}, f)
            f.write("\n")
        if obs:
            np.savez_compressed(os.path.join(out, job["name"] + "_demos.npz"),
                                obs=np.concatenate(obs), act=np.concatenate(act))
    return {"ok": all(s.get("ok") and s["runner_exit"] == 0 for s in statuses),
            "workers": statuses}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--jobs", required=True, help="JSON list of jobs")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--duration", type=int, default=3600,
                   help="wall-clock cap in seconds; the controller quits when done")
    args = p.parse_args()
    with open(args.jobs) as f:
        jobs = json.load(f)
    status = run_parallel(args.variant, jobs, args.out, args.duration, args.workers)
    print(json.dumps(status))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
