#!/usr/bin/env python3
"""Parity check of render_shift.py against lanai-made (keti) camera-shift datasets.

Expects any layout under --keti-root that contains LeRobot dirs with
extras/dataset_meta.json, extras/episode_XXXXXX/{states.npz, model.xml(.gz), ep_meta.json},
meta/camera_config.json (shifted variants) and videos/.../episode_XXXXXX.mp4.

For each found dataset: re-render the episode with our conventions and report the
mean |diff| (0-255) per camera vs the keti mp4. If a shifted camera mismatches,
search sign flips x Euler orders on a few frames and report the best convention.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import av  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import regen_common as RC  # noqa: E402
import render_shift as RS  # noqa: E402
from regen_replay import CAMERAS  # noqa: E402
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils  # noqa: E402
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils  # noqa: E402

VARIANTS = ["combined_shift_v2", "wrist_shift_v2", "front_shift_v2"]


def read_frames(p):
    with av.open(str(p)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def find_datasets(root: Path):
    for dm in root.rglob("extras/dataset_meta.json"):
        ds = dm.parent.parent
        for ep_dir in sorted((ds / "extras").glob("episode_*")):
            if (ep_dir / "states.npz").exists():
                yield ds, int(ep_dir.name.split("_")[-1])


def load_episode(ds, ep):
    ex = ds / "extras" / f"episode_{ep:06d}"
    states = np.load(ex / "states.npz")["states"]
    if (ex / "model.xml.gz").exists():
        xml = gzip.open(ex / "model.xml.gz", "rt").read()
    else:
        xml = open(ex / "model.xml").read()
    return states, xml, json.load(open(ex / "ep_meta.json"))


def robot_of(ds, xml):
    for r in RC.ROBOT_GRIPPERS:
        if r in str(ds):
            return r
    dm = json.load(open(ds / "extras" / "dataset_meta.json"))
    return dm["env_args"]["env_kwargs"]["robots"]


def variant_of(ds):
    for v in VARIANTS:
        if v.replace("_v2", "") in str(ds):
            return v
    return "canonical"


def make_env(ds, robot):
    dm = json.load(open(ds / "extras" / "dataset_meta.json"))
    em = dm["env_args"]
    em["env_kwargs"]["robots"] = robot
    em["env_kwargs"]["gripper_types"] = RC.ROBOT_GRIPPERS[robot]
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    return EnvUtils.create_env_for_data_processing(env_meta=em, camera_names=CAMERAS, camera_height=256,
                                                   camera_width=256, reward_shaping=False)


def render(env, states, xml, ep_meta, spec, frames_idx, rot_fn=None):
    env.reset()
    env.reset_to({"states": states[0], "model": xml, "ep_meta": json.dumps(ep_meta)})
    sim = env.env.sim
    if rot_fn is None:
        RS.apply_shifts(sim, spec)
    else:
        for cam, s in spec.items():
            cid = sim.model.camera_name2id(cam)
            q = sim.model.cam_quat[cid].copy()
            r0 = R.from_quat([q[1], q[2], q[3], q[0]])
            r1 = r0 * rot_fn(s)
            x, y, z, w = r1.as_quat()
            sim.model.cam_quat[cid] = [w, x, y, z]
            sim.model.cam_pos[cid] = sim.model.cam_pos[cid] + r0.apply([0, 0, s.get("back_m", 0.0)])
        sim.forward()
    out = {c: [] for c in CAMERAS}
    for t in frames_idx:
        obs = env.reset_to({"states": states[t]})
        for c in CAMERAS:
            out[c].append(RS.to_u8(np.asarray(obs[f"{c}_image"])))
    return out


def diff(a, b):
    return float(np.mean([np.abs(x.astype(np.float32) - y.astype(np.float32)).mean() for x, y in zip(a, b)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keti-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tol", type=float, default=6.0)
    a = p.parse_args()
    results = []
    for ds, ep in find_datasets(a.keti_root):
        states, xml, ep_meta = load_episode(ds, ep)
        robot, variant = robot_of(ds, xml), variant_of(ds)
        spec = {}
        if variant != "canonical":
            if (ds / "meta" / "camera_config.json").exists():
                print("keti camera_config:", json.dumps(json.load(open(ds / "meta" / "camera_config.json")))[:600])
            spec = RS.shift_spec(robot, variant, json.load(open(RS.SHIFTS)))
        keti = {c: read_frames(next(ds.glob(f"videos/*/observation.images.{c}/episode_{ep:06d}.mp4"))) for c in CAMERAS}
        T = min(len(states), len(keti[CAMERAS[0]]))
        idx = list(np.linspace(0, T - 1, 5).astype(int))
        env = make_env(ds, robot)
        ours = render(env, states, xml, ep_meta, spec, idx)
        res = dict(dataset=str(ds), episode=ep, robot=robot, variant=variant,
                   diff={c: round(diff(ours[c], [keti[c][i] for i in idx]), 2) for c in CAMERAS})
        bad = [c for c in CAMERAS if res["diff"][c] > a.tol]
        if bad and spec:
            # convention search on the mismatching shifted cams
            best = None
            for signs in itertools.product([1, -1], repeat=3):
                for order in itertools.permutations("yxz"):
                    def rot_fn(s, signs=signs, order=order):
                        ang = {"y": signs[0] * s["yaw"], "x": signs[1] * s["pitch"], "z": signs[2] * s["roll"]}
                        r = R.identity()
                        for ax in order:
                            r = r * R.from_euler(ax, ang[ax], degrees=True)
                        return r
                    o = render(env, states, xml, ep_meta, {c: spec[c] for c in bad if c in spec}, idx[:2], rot_fn)
                    d = np.mean([diff(o[c], [keti[c][i] for i in idx[:2]]) for c in bad if c in spec])
                    if best is None or d < best[0]:
                        best = (float(d), signs, "".join(order))
            res["convention_search_best"] = dict(diff=round(best[0], 2), signs_yaw_pitch_roll=best[1],
                                                 order=best[2])
        res["ok"] = not bad
        print(json.dumps(res), flush=True)
        results.append(res)
        env.env.close()
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
