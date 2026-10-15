# Findings

What the cross-check showed, what was done to explain it and what is still
open. Numbers are success rates over the 100 held-out initial states.
Everything in the diagnostics section is exploratory: it was run after the
preregistered results were in and was not predicted.

## 1. Imitation transfers between the simulators where the task is easy

In the nominal world both experts score 1.00, all ten clones score 1.00 and
each simulator's clones score 1.00 when run unchanged in the other. OmniSim
reproduces the MuJoCo result here without any tuning beyond matching the
geometry and ten times the box mass changes nothing in either.

## 2. The clone beats its teacher on the friction cliff, in both simulators

This held as predicted (P3). In MuJoCo at friction 0.07 the expert scores 0.02
and the clones 0.48 on average. In OmniSim at 0.07 it is 0.52 against 0.71 and
at 0.06 it is 0.03 against 0.54. Two simulators that disagree about where the
cliff is agree that the clone is more robust than the expert on it, which
makes this the most transferable result in the repository.

The mechanism is not established. Measured in MuJoCo at 0.07, the clones start
lifting sooner after closing (2.4 steps against the expert's fixed 6),
command a slightly softer grip (0.82 to 0.86 against 1.0) and accelerate the
box upwards less hard on average (2.1 against 2.6 m/s²). None of these ranks
the five clones by success, so none of them is the explanation on its own.

## 3. The cliff is in a different place and the obvious causes do not explain it

Two predictions failed (P2, P5). MuJoCo has a step: the expert is at 0.94 at
friction 0.08 and 0.02 at 0.07. OmniSim has a slope: the clones are already
down to 0.95 at 0.15, the expert is at 0.52 at both 0.08 and 0.07 and 0.03 at
0.06. The same solver, MuJoCo 3.11, runs underneath both.

Two differences between the scenes were found in OmniSim's source and tested.

**Finger gains.** OmniSim derives a position-controlled slider's gains from
its force limit, kp = 10 x maxForce and kv = 0.5 x maxForce
(`src/omnisim/nodes/OmBasicJoint.cpp`), so the 60 N fingers get 600 and 30
where the MuJoCo scene has 300 and none. The fingers therefore squeeze about
twice as hard in OmniSim. Rerunning the MuJoCo expert with OmniSim's finger
gains (`scripts/diagnose_finger_gains.py`):

| friction | 1.00 | 0.30 | 0.15 | 0.08 | 0.07 | 0.06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MuJoCo, own gains | 1.00 | 1.00 | 1.00 | 0.94 | 0.02 | 0.00 |
| MuJoCo, OmniSim's finger gains | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.90 |
| OmniSim | 1.00 | 1.00 | 0.99 | 0.52 | 0.52 | 0.03 |

The harder squeeze moves the MuJoCo cliff the other way, past 0.06. So the gain
mapping explains why OmniSim holds on at 0.06 at all and makes its weakness at
0.08 to 0.15 harder to explain, not easier.

**Contact stiffness.** OmniSim gives every shape contact stiffness ke 2500 and
damping kd 100 by default, where the MuJoCo scene sets stiffer contacts per
geom. Rerunning the OmniSim expert at friction 0.08 with only
`OMNISIM_NEWTON_CONTACT_KE` changed (`scripts/diagnose_contact_stiffness.py`):

| contact ke | 2,500 (default) | 10,000 | 40,000 | 160,000 |
| --- | ---: | ---: | ---: | ---: |
| OmniSim expert | 0.52 | 0.31 | 0.04 | 0.23 |

Stiffer is worse and not monotonically. That rules out "OmniSim is simply too
soft". It also matches OmniSim's own documentation, which warns that raising
contact stiffness ejects grasped parts. The 2,500 run reproduced the main
result exactly, episode for episode.

**Still open.** The remaining differences, none of them tested yet:
- OmniSim does not apply declared joint damping unless
  `OMNISIM_NEWTON_JOINT_DYNAMICS=1` is set and says so in its log.
- OmniSim maps ke and kd to MuJoCo's solref and solimp in a way this
  repository has not inspected.
- The MuJoCo pads are softer than the box (their own solref). OmniSim cannot
  express that per shape.

## 4. Practical observations about OmniSim

These are the notes the outreach asked for: where it became unclear.

- **Determinism holds.** Two separate 12-container runs of the 200
  demonstrations were bit-identical in all 20,000 rows and a rerun of a
  diagnostic reproduced its earlier result episode for episode. That claim in
  OmniSim's README is accurate.
- **A Solid without a collider gets a 0.12 m sphere.** The gantry carriages
  needed no geometry. OmniSim gave each a default sphere collider, which sat
  inside the table and stopped the hand reaching its start pose by up to 6 cm.
  It is logged as a warning but the run otherwise succeeds. The fix here is a
  1 cm box parked 1.2 m above the scene.
- **Mass and friction are fixed when the world loads.** There is no
  environment override and a runtime field write does not reach the solver, so
  a parameter sweep is one generated world file per value
  (`omnisim/gen_worlds.py`).
- **Moving a root Solid resets every joint.** It is useful for episode resets
  once known but it is documented only in the source.
- **`run-headless` needs `--duration`.** Without it the run is a load check
  that stops after half a second, before the controller has done anything.
- **Speed.** About 1.8x real time per container with twelve running at once,
  for a seven-body scene at a 2 ms step. That is roughly 2.5 s per four-second
  episode, against 0.03 s calling MuJoCo directly. Parallel containers make it
  workable: the whole OmniSim side ran in under an hour.
- **Getting it running safely took real effort.** Only Windows has a prebuilt
  package and it cannot be tied to the source. A source build in Docker
  (a few minutes on a 16-core machine) is the way to run audited code; see
  [security-review.md](security-review.md).
- **Not tested:** the plain-language-to-scene feature. It needs the paid
  OmniLink agent and an account and was out of scope.
