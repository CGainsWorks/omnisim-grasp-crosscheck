# Preregistration

Written on 15 October 2026, after the MuJoCo screen and before any OmniSim run
of a world other than nominal. The machine-readable copy, which the tests
read, is [`results/preregistration.json`](../results/preregistration.json).
The commit that adds both files comes before the commit that adds any OmniSim
result for a changed world, so the order can be checked in the history.

## The one controlled change

**Box friction lowered from 1.0 to 0.07.** Nothing else in the world changes.

Two other candidate kinds were screened on MuJoCo and dropped. Box mass does
not bite: four and ten times the nominal 80 g leave the expert at 1.00, because
a 60 N squeeze has friction to spare. Moderate friction does not bite either:
0.30 and 0.15 leave everything at 1.00. The effect lives in a narrow band:

| box friction | expert | five clones |
| ---: | ---: | --- |
| 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 |
| 0.08 | 0.94 | 0.97 0.73 0.97 0.97 0.98 |
| **0.07** | **0.02** | **0.12 0.31 0.52 0.58 0.87** |
| 0.06 | 0.00 | 0.00 0.00 0.00 0.00 0.00 |

The floor is structural. MuJoCo combines two surfaces' friction by taking the
larger value and the pads are 0.05, so no box value below that changes the
pad-box contact.

0.07 was chosen because it is where the most can go wrong in a second
simulator. It sits on a cliff and in MuJoCo the clones outperform the expert
they were cloned from. A change that leaves everything at 1.00 in both
simulators would agree trivially and show nothing.

## Predictions for OmniSim

| | Claim |
| --- | --- |
| P1 | Nominal world: the expert scores at least 0.95 and the mean clone at least 0.90. |
| P2 | Friction 0.07: the expert scores below 0.50. |
| P3 | Friction 0.07: the mean clone scores above the expert. |
| P4 | Ten times the mass moves neither the expert nor the mean clone by more than 0.05. |
| P5 | The friction cliff lies between 0.08 and 0.06, as in MuJoCo. |

P5 is the weakest. Both scenes run MuJoCo 3.11 underneath but MuJoCo's
contact softness is set per geom (the pads are softer than the box) and
OmniSim exposes it only per world. If the cliff moves, that difference is
the first suspect.

The secondary runs (friction 0.30, 0.15, 0.08 and 0.06 and ten times the
mass) locate the cliff for P4 and P5. They are not extra chances at P2 and P3.
