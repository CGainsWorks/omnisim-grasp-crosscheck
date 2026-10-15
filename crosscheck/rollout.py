"""Episode loop shared by both simulators.

An environment needs two methods: ``reset(init, variant) -> obs`` and
``step(action) -> obs``. ``init`` comes from :func:`crosscheck.task.sample_reset`
and ``variant`` names the world (``"nominal"`` or the controlled change), so a
seed means the same box position, yaw and hand start in MuJoCo and in OmniSim.
"""

from __future__ import annotations

import numpy as np

from crosscheck import task
from crosscheck.expert import ScriptedExpert

DEMO_NOISE = 0.02


def run_episode(env, actor, seed: int, variant: str, record: bool = False) -> dict:
    """Run one episode. ``actor`` has ``act(obs)`` and optionally ``reset()``."""
    init = task.sample_reset(np.random.default_rng(seed))
    obs = env.reset(init, variant)
    if hasattr(actor, "reset"):
        actor.reset()
    observations, actions = [], []
    peak_lift = 0.0
    rest_z = float(obs[task.OBJ_POS][2])
    for _ in range(task.MAX_STEPS):
        action = actor.act(obs)
        if record:
            observations.append(obs.astype(np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
        obs = env.step(action)
        peak_lift = max(peak_lift, float(obs[task.OBJ_POS][2]) - rest_z)
    success = task.is_success(obs[task.OBJ_POS], obs[task.GRIP_POS], obs[task.GOAL_POS])
    out = {"seed": seed, "variant": variant, "success": success,
           "peak_lift": peak_lift, "final_obj": obs[task.OBJ_POS].tolist()}
    if record:
        out["obs"] = np.asarray(observations)
        out["act"] = np.asarray(actions)
    return out


def collect_demos(env, seeds, variant: str = "nominal", noise: float = DEMO_NOISE):
    """Expert demonstrations. Failed episodes are kept out of the training set.

    The expert's actions carry a little Gaussian noise, as in the original
    repository, so the demonstrations visit states slightly off the expert's
    own path and the clone sees how to recover from them.
    """
    obs, act, outcomes = [], [], []
    for seed in seeds:
        expert = ScriptedExpert(noise=noise, rng=np.random.default_rng(seed))
        ep = run_episode(env, expert, seed, variant, record=True)
        outcomes.append(ep["success"])
        if ep["success"]:
            obs.append(ep["obs"])
            act.append(ep["act"])
    return np.concatenate(obs), np.concatenate(act), outcomes
