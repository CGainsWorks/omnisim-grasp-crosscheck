import json
import os
import subprocess
import sys

import numpy as np
import pytest

from crosscheck import bc, task
from crosscheck.expert import ScriptedExpert
from crosscheck.rollout import collect_demos, run_episode
from crosscheck.variants import VARIANTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def test_reset_draw_is_deterministic():
    a = task.sample_reset(np.random.default_rng(7))
    b = task.sample_reset(np.random.default_rng(7))
    assert np.allclose(a["grip"], b["grip"]) and a["box_yaw"] == b["box_yaw"]


def test_setpoint_stays_in_workspace():
    sp = task.next_setpoint(task.WORKSPACE_HIGH.copy(), np.ones(4))
    assert np.allclose(sp, task.WORKSPACE_HIGH)


def test_success_needs_the_box_in_the_hand():
    goal = np.array([0.0, 0.0, task.TABLE_Z + task.HOLD_HEIGHT])
    assert task.is_success(goal, goal, goal)
    assert not task.is_success(goal, goal + [0, 0, 0.1], goal)


def test_bc_fits_a_smooth_mapping():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(2000, task.OBS_DIM))
    act = np.tanh(obs[:, :task.ACT_DIM] * 0.5)
    _, train, val = bc.fit(obs, act, rng, epochs=15)
    assert val[-1] < 0.1 * val[0]


def test_policy_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    p = bc.MLPPolicy.init(task.OBS_DIM, task.ACT_DIM, 16, rng)
    path = tmp_path / "p.npz"
    p.save(str(path))
    q = bc.MLPPolicy.load(str(path))
    x = rng.normal(size=task.OBS_DIM)
    assert np.allclose(p.act(x), q.act(x))


def test_committed_worlds_are_current():
    r = subprocess.run([sys.executable, os.path.join(REPO, "omnisim", "gen_worlds.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_worker_split_covers_every_seed_once():
    from run_omnisim import split
    jobs = [{"name": "a", "actor": "expert", "seeds": [10, 107], "variant": "nominal"}]
    plans = split(jobs, 12)
    seeds = [s for plan in plans for job in plan for s in range(*job["seeds"])]
    assert seeds == list(range(10, 107))


def test_preregistered_variant_exists():
    with open(os.path.join(REPO, "results", "preregistration.json")) as f:
        assert json.load(f)["variant"] in VARIANTS


mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def env():
    from crosscheck.mujoco_env import MujocoLiftEnv
    return MujocoLiftEnv()


def test_expert_lifts_the_box(env):
    eps = [run_episode(env, ScriptedExpert(), s, "nominal") for s in range(10_000, 10_005)]
    assert all(e["success"] for e in eps)


def test_variant_changes_the_model(env):
    obs = env.reset(task.sample_reset(np.random.default_rng(0)), "friction_0.08")
    gid = env._obj_geom
    assert env.model.geom_friction[gid][0] == pytest.approx(0.08)
    env.reset(task.sample_reset(np.random.default_rng(0)), "nominal")
    assert env.model.geom_friction[gid][0] == pytest.approx(task.BOX_FRICTION)
    assert obs.shape == (task.OBS_DIM,)


def test_demo_collection_shapes(env):
    obs, act, outcomes = collect_demos(env, range(2))
    assert obs.shape[1] == task.OBS_DIM and act.shape[1] == task.ACT_DIM
    assert len(obs) == task.MAX_STEPS * sum(outcomes)
