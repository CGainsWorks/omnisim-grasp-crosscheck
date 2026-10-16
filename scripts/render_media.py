"""Render the README media: the same episodes in MuJoCo and OmniSim, side by side.

    python scripts/render_media.py

Each clip is one held-out seed run by one policy in one world, rendered from the
same camera pose in both simulators (``crosscheck/camera.py``). The OmniSim
frames come from a Camera device inside the engine, in the ``_render`` twin of
the world, with rendering switched on under the container's virtual display.
The caption under each half is the outcome that episode actually had.

Needs Docker and the OmniSim image (as ``scripts/run_omnisim.py``), plus Pillow
and glfw for the MuJoCo side (``requirements-media.txt``). Rendering is slow in
the container, which draws on the CPU: about a second per frame.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from run_omnisim import run  # noqa: E402

from crosscheck import camera  # noqa: E402
from crosscheck.bc import MLPPolicy  # noqa: E402
from crosscheck.expert import ScriptedExpert  # noqa: E402
from crosscheck.rollout import run_episode  # noqa: E402

MEDIA = os.path.join(REPO, "media")
SCRATCH = os.path.join(REPO, "scratch", "media")
SEED = 10_000
CLIPS = [
    # (file name, world, actor, title)
    ("nominal_expert", "nominal", "expert", "nominal world, scripted expert"),
    ("friction_0.07_expert", "friction_0.07", "expert", "box friction 0.07, scripted expert"),
    ("friction_0.07_clone", "friction_0.07", "bc_s4", "box friction 0.07, cloned policy s4"),
]
FRAME_STEP = 2      # keep every second control step: 50 frames, 12.5 fps of a 4 s episode
SCALE = 0.75


def actor_for(name: str, sim: str):
    if name == "expert":
        return ScriptedExpert()
    return MLPPolicy.load(os.path.join(REPO, "results", sim, "policies", name + ".npz"))


class RecordingEnv:
    """Wraps the MuJoCo env and renders one frame after every step."""

    def __init__(self, env, renderer, cam):
        self.env, self.renderer, self.cam, self.frames = env, renderer, cam, []

    def reset(self, init, variant):
        self.frames = []
        return self.env.reset(init, variant)

    def step(self, action):
        obs = self.env.step(action)
        self.renderer.update_scene(self.env.data, camera=self.cam)
        self.frames.append(Image.fromarray(self.renderer.render()))
        return obs


def mujoco_clips() -> dict:
    import mujoco

    from crosscheck.mujoco_env import MujocoLiftEnv

    env = MujocoLiftEnv()
    env.model.vis.global_.fovy = camera.mujoco_fovy_degrees()
    renderer = mujoco.Renderer(env.model, camera.HEIGHT, camera.WIDTH)
    cam = mujoco.MjvCamera()
    pose = camera.mujoco_free_camera()
    cam.lookat[:] = pose["lookat"]
    cam.distance, cam.azimuth, cam.elevation = pose["distance"], pose["azimuth"], pose["elevation"]
    rec = RecordingEnv(env, renderer, cam)
    out = {}
    for name, world, actor, _ in CLIPS:
        ep = run_episode(rec, actor_for(actor, "mujoco"), SEED, world)
        out[name] = (rec.frames, ep["success"])
    renderer.close()
    return out


def omnisim_clips() -> dict:
    by_world = {}
    for name, world, actor, _ in CLIPS:
        spec = actor if actor == "expert" else os.path.join(
            REPO, "results", "omnisim", "policies", actor + ".npz")
        by_world.setdefault(world, []).append(
            {"name": name, "actor": spec, "seeds": [SEED, SEED + 1], "variant": world})

    def go(item):
        world, jobs = item
        out = os.path.join(SCRATCH, "omnisim", world)
        status = run(world, jobs, out, 3600, {"CROSSCHECK_FRAMES": "/work/out/frames"},
                     render=True)
        if not (status.get("ok") and status["runner_exit"] == 0):
            raise SystemExit(f"OmniSim render of {world} failed: {json.dumps(status)}")
        return world

    with ThreadPoolExecutor(len(by_world)) as pool:
        list(pool.map(go, by_world.items()))
    clips = {}
    for name, world, _, _ in CLIPS:
        base = os.path.join(SCRATCH, "omnisim", world)
        files = sorted(glob.glob(os.path.join(base, "frames", f"{name}_{SEED}_*.png")))
        with open(os.path.join(base, name + ".json")) as f:
            success = json.load(f)["episodes"][0]["success"]
        clips[name] = ([Image.open(p).convert("RGB") for p in files], success)
    return clips


def caption(img: Image.Image, text: str, ok: bool) -> Image.Image:
    bar = 26
    out = Image.new("RGB", (img.width, img.height + bar), (24, 26, 30))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=15)
    draw.text((8, img.height + 5), text, fill=(230, 230, 230), font=font)
    verdict = "success" if ok else "failed"
    colour = (110, 200, 120) if ok else (235, 110, 100)
    w = draw.textlength(verdict, font=font)
    draw.text((img.width - w - 8, img.height + 5), verdict, fill=colour, font=font)
    return out


def compose(name: str, title: str, mj: tuple, om: tuple) -> None:
    (mj_frames, mj_ok), (om_frames, om_ok) = mj, om
    n = min(len(mj_frames), len(om_frames))
    frames = []
    for i in range(0, n, FRAME_STEP):
        left = caption(mj_frames[i], "MuJoCo 3.11", mj_ok)
        right = caption(om_frames[i], "OmniSim 9.2.4", om_ok)
        head = 28
        frame = Image.new("RGB", (left.width + right.width + 4, left.height + head), (24, 26, 30))
        ImageDraw.Draw(frame).text((10, 6), f"{title}, seed {SEED}", fill=(240, 240, 240),
                                   font=ImageFont.load_default(size=16))
        frame.paste(left, (0, head))
        frame.paste(right, (left.width + 4, head))
        frames.append(frame.resize((int(frame.width * SCALE), int(frame.height * SCALE)),
                                   Image.LANCZOS))
    frames[0].save(os.path.join(MEDIA, name + ".gif"), save_all=True,
                   append_images=frames[1:], duration=int(1000 * 0.04 * FRAME_STEP), loop=0,
                   optimize=True)
    # A still from late in the episode, for places a GIF does not play.
    frames[-1].save(os.path.join(MEDIA, name + ".png"), optimize=True)


def main() -> None:
    os.makedirs(MEDIA, exist_ok=True)
    om = omnisim_clips()
    mj = mujoco_clips()
    for name, _, _, title in CLIPS:
        compose(name, title, mj[name], om[name])
        print(f"{name}: MuJoCo {'success' if mj[name][1] else 'failed'}, "
              f"OmniSim {'success' if om[name][1] else 'failed'}", flush=True)
    # Record what the frames show against the committed per-episode results, so
    # a caption can never disagree with the tables without someone noticing.
    with open(os.path.join(MEDIA, "clips.json"), "w", newline="\n") as f:
        json.dump({"seed": SEED, "clips": {n: {"mujoco_success": bool(mj[n][1]),
                                               "omnisim_success": bool(om[n][1])}
                                           for n, *_ in CLIPS}}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
