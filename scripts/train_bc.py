"""Train the behaviour-cloning policies from a demonstrations file.

    python scripts/train_bc.py --demos results/omnisim/demos.npz --out results/omnisim/policies

Writes ``bc_s<k>.npz`` for each training seed and ``training.json`` with the
final validation error of each. Pure numpy, no simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crosscheck import bc, protocol  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bc-seeds", type=int, default=len(protocol.BC_SEEDS))
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    data = np.load(args.demos)
    os.makedirs(args.out, exist_ok=True)
    summary = {"samples": int(len(data["obs"])), "policies": {}}
    for k in list(protocol.BC_SEEDS)[:args.bc_seeds]:
        policy, _, val = bc.fit(data["obs"], data["act"], np.random.default_rng(k),
                                epochs=args.epochs)
        policy.save(os.path.join(args.out, f"bc_s{k}.npz"))
        summary["policies"][f"s{k}"] = {"final_val_mse": round(val[-1], 6)}
        print(f"bc s{k}: val mse {val[-1]:.5f}", flush=True)
    with open(os.path.join(args.out, "training.json"), "w", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
