"""The experimental protocol, fixed in one place so both simulators run it alike."""

DEMO_SEEDS = range(0, 200)          # 200 expert episodes in the nominal world
EVAL_SEEDS = range(10_000, 10_100)  # 100 held-out initial states per world
BC_SEEDS = (0, 1, 2, 3, 4)          # five independently trained clones
