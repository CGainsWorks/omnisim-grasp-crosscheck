"""Build the results tables and grade the preregistered predictions.

Reads only committed result files, so it needs no simulator:

    python scripts/summarise.py           # write results/summary.json and the README tables
    python scripts/summarise.py --check   # fail if either is out of date
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
README = os.path.join(REPO, "README.md")
BC_SEEDS = range(5)
WORLD_ORDER = ["nominal", "friction_0.30", "friction_0.15", "friction_0.08", "friction_0.07",
               "friction_0.06", "mass_x4", "mass_x10"]
MARKERS = {
    "results": ("<!-- results:start -->", "<!-- results:end -->"),
    "predictions": ("<!-- predictions:start -->", "<!-- predictions:end -->"),
    "transfer": ("<!-- transfer:start -->", "<!-- transfer:end -->"),
}


def load(path):
    with open(path) as f:
        return json.load(f)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def omnisim_rate(name: str):
    path = os.path.join(RES, "omnisim", name + ".json")
    if not os.path.exists(path):
        return None
    eps = load(path)["episodes"]
    return sum(e["success"] for e in eps) / len(eps), len(eps)


def collect() -> dict:
    mj = load(os.path.join(RES, "mujoco", "results.json"))
    table = {}
    for world in WORLD_ORDER:
        row = {}
        if world in mj["expert"]:
            e = mj["expert"][world]
            row["mujoco"] = {
                "expert": round(e["success"], 3),
                "clones": [round(mj["bc"][f"s{k}"][world]["success"], 3) for k in BC_SEEDS]}
        ex = omnisim_rate(f"expert_{world}")
        if ex is not None:
            clones = [omnisim_rate(f"bc_s{k}_{world}") for k in BC_SEEDS]
            row["omnisim"] = {"expert": round(ex[0], 3),
                              "clones": [round(c[0], 3) for c in clones if c]}
        if row:
            table[world] = row
    for row in table.values():
        for sim in row.values():
            n = 100
            sim["expert_ci95"] = [round(x, 3) for x in wilson(round(sim["expert"] * n), n)]
            sim["clone_mean"] = round(sum(sim["clones"]) / len(sim["clones"]), 3)

    transfer = {}
    for world in ("nominal", "friction_0.07"):
        mj_in_om = [omnisim_rate(f"mujoco_bc_s{k}_{world}") for k in BC_SEEDS]
        if all(mj_in_om):
            transfer.setdefault(world, {})["mujoco_clones_in_omnisim"] = [
                round(r[0], 3) for r in mj_in_om]
    tpath = os.path.join(RES, "mujoco", "transfer.json")
    if os.path.exists(tpath):
        for world, rates in load(tpath)["omnisim_clones_in_mujoco"].items():
            transfer.setdefault(world, {})["omnisim_clones_in_mujoco"] = rates
    return {"table": table, "transfer": transfer,
            "predictions": grade(table, load(os.path.join(RES, "preregistration.json")))}


def grade(table: dict, prereg: dict) -> list:
    def om(world):
        return table.get(world, {}).get("omnisim")

    checks = {}
    nom, f07, mass, f08, f06 = (om(w) for w in
                                ("nominal", "friction_0.07", "mass_x10", "friction_0.08",
                                 "friction_0.06"))
    if nom:
        checks["P1"] = (nom["expert"] >= 0.95 and nom["clone_mean"] >= 0.90,
                        f"expert {nom['expert']:.2f}, mean clone {nom['clone_mean']:.2f}")
    if f07:
        checks["P2"] = (f07["expert"] < 0.50, f"expert {f07['expert']:.2f}")
        checks["P3"] = (f07["clone_mean"] > f07["expert"],
                        f"mean clone {f07['clone_mean']:.2f} against expert {f07['expert']:.2f}")
    if nom and mass:
        de = abs(mass["expert"] - nom["expert"])
        dc = abs(mass["clone_mean"] - nom["clone_mean"])
        checks["P4"] = (de <= 0.05 and dc <= 0.05,
                        f"expert moves {de:.2f}, mean clone moves {dc:.2f}")
    if f08 and f06:
        checks["P5"] = (f08["expert"] >= 0.90 and f06["expert"] <= 0.10,
                        f"expert {f08['expert']:.2f} at 0.08, {f06['expert']:.2f} at 0.06")
    out = []
    for p in prereg["predictions"]:
        held, detail = checks.get(p["id"], (None, "not run"))
        out.append({"id": p["id"], "claim": p["claim"],
                    "verdict": "pending" if held is None else ("held" if held else "failed"),
                    "observed": detail})
    return out


def label(world: str) -> str:
    if world == "nominal":
        return "nominal"
    kind, value = world.split("_")
    return f"box friction {value}" if kind == "friction" else f"box mass {value.lstrip('x')}x"


def fmt_clones(c):
    return f"{sum(c) / len(c):.2f} ({min(c):.2f} to {max(c):.2f})"


def render(summary: dict) -> dict:
    rows = ["| world | MuJoCo expert | MuJoCo clones | OmniSim expert | OmniSim clones |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for world, row in summary["table"].items():
        cells = [f"**{label(world)}**" if world == "friction_0.07" else label(world)]
        for sim in ("mujoco", "omnisim"):
            r = row.get(sim)
            cells += ([f"{r['expert']:.2f}", fmt_clones(r["clones"])] if r else ["", ""])
        rows.append("| " + " | ".join(cells) + " |")
    preds = ["| | claim | observed in OmniSim | verdict |", "| --- | --- | --- | --- |"]
    for p in summary["predictions"]:
        preds.append(f"| {p['id']} | {p['claim']} | {p['observed']} | **{p['verdict']}** |")
    trans = ["| world | MuJoCo clones run in OmniSim | OmniSim clones run in MuJoCo |",
             "| --- | ---: | ---: |"]
    for world, t in summary["transfer"].items():
        a = t.get("mujoco_clones_in_omnisim")
        b = t.get("omnisim_clones_in_mujoco")
        trans.append(f"| {label(world)} | {fmt_clones(a) if a else ''} | "
                     f"{fmt_clones(b) if b else ''} |")
    return {"results": "\n".join(rows), "predictions": "\n".join(preds),
            "transfer": "\n".join(trans)}


def splice(text: str, blocks: dict) -> str:
    for key, (start, end) in MARKERS.items():
        if start not in text:
            continue
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{blocks[key]}\n{end}{tail}"
    return text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    summary = collect()
    summary_text = json.dumps(summary, indent=2) + "\n"
    readme = open(README, encoding="utf-8").read() if os.path.exists(README) else ""
    new_readme = splice(readme, render(summary))
    spath = os.path.join(RES, "summary.json")
    if args.check:
        stale = []
        if not os.path.exists(spath) or open(spath, encoding="utf-8").read() != summary_text:
            stale.append("results/summary.json")
        if new_readme != readme:
            stale.append("README.md")
        if stale:
            print("out of date, rerun scripts/summarise.py: " + ", ".join(stale))
            return 1
        print("summary and README tables match the raw results")
        return 0
    with open(spath, "w", encoding="utf-8", newline="\n") as f:
        f.write(summary_text)
    if readme:
        with open(README, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
