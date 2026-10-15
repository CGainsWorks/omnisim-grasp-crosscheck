"""MuJoCo reference: demonstrations, behaviour cloning and evaluation.

    python scripts/run_mujoco.py --variants nominal friction_0.15 --out results/mujoco

Writes ``demos.npz``, ``policies/bc_s<k>.npz`` and ``results.json`` under --out.
``--eval-episodes`` trims the evaluation set for a quick check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crosscheck import bc, protocol  # noqa: E402
from crosscheck.expert import ScriptedExpert  # noqa: E402
from crosscheck.mujoco_env import MujocoLiftEnv  # noqa: E402
from crosscheck.rollout import collect_demos, run_episode  # noqa: E402


def evaluate(env, actor, variant, seeds):
    eps = [run_episode(env, actor, s, variant) for s in seeds]
    return {"success": float(np.mean([e["success"] for e in eps])),
            "n": len(eps),
            "mean_peak_lift": float(np.mean([e["peak_lift"] for e in eps])),
            "outcomes": "".join("1" if e["success"] else "0" for e in eps)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["nominal"])
    p.add_argument("--out", default="results/mujoco")
    p.add_argument("--demo-episodes", type=int, default=len(protocol.DEMO_SEEDS))
    p.add_argument("--eval-episodes", type=int, default=len(protocol.EVAL_SEEDS))
    p.add_argument("--bc-seeds", type=int, default=len(protocol.BC_SEEDS))
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "policies"), exist_ok=True)
    env = MujocoLiftEnv()
    t0 = time.time()

    demo_seeds = list(protocol.DEMO_SEEDS)[:args.demo_episodes]
    obs, act, outcomes = collect_demos(env, demo_seeds)
    np.savez_compressed(os.path.join(args.out, "demos.npz"), obs=obs, act=act)
    print(f"demos: {sum(outcomes)}/{len(outcomes)} expert successes, {len(obs)} samples")

    eval_seeds = list(protocol.EVAL_SEEDS)[:args.eval_episodes]
    results = {"simulator": "mujoco", "demo_successes": int(sum(outcomes)),
               "demo_episodes": len(outcomes), "samples": int(len(obs)),
               "eval_episodes": len(eval_seeds), "expert": {}, "bc": {}}
    for v in args.variants:
        results["expert"][v] = evaluate(env, ScriptedExpert(), v, eval_seeds)
        print(f"expert {v}: {results['expert'][v]['success']:.3f}")

    for k in list(protocol.BC_SEEDS)[:args.bc_seeds]:
        policy, train_curve, val_curve = bc.fit(obs, act, np.random.default_rng(k),
                                                epochs=args.epochs)
        policy.save(os.path.join(args.out, "policies", f"bc_s{k}.npz"))
        row = {"final_val_mse": val_curve[-1]}
        for v in args.variants:
            row[v] = evaluate(env, policy, v, eval_seeds)
        results["bc"][f"s{k}"] = row
        print(f"bc s{k}: " + " ".join(f"{v}={row[v]['success']:.2f}" for v in args.variants))

    results["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "results.json"), "w", newline="\n") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
