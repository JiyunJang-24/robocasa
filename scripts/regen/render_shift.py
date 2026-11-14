#!/usr/bin/env python3
"""Camera-shifted re-render of a canonical regen LeRobot dataset.

Re-renders the SAME saved episodes from extras/episode_*/states.npz + model.xml
with camera mount poses changed per camera_shifts_v2.json. Only videos differ;
data/, meta/ (+camera_config.json) and extras/ are byte-identical copies, with 1:1
episode ids.

  render   : --episodes ...  -> writes <out>/videos/... for those episodes
  finalize : copies data/meta/extras, writes meta/camera_config.json,
             provenance.jsonl, validation_report.json, MANIFEST.md5
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import regen_common as RC  # noqa: E402,F401  (registers grippers / robots)
from regen_replay import CAMERAS, encode_mp4  # noqa: E402
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils  # noqa: E402
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils  # noqa: E402

HERE = Path(__file__).resolve().parent
SHIFTS = HERE / "camera_shifts_v2.json"


def shift_spec(robot: str, variant: str, shifts: dict) -> dict:
    spec = {}
    for part in shifts["variants"][variant]:
        spec.update(shifts["robots"][robot][part])
    return spec


def local_rot(s):
    """Ry(yaw) @ Rx(pitch) @ Rz(roll), degrees, in the camera's own frame."""
    return (R.from_euler("y", s["yaw"], degrees=True) * R.from_euler("x", s["pitch"], degrees=True)
            * R.from_euler("z", s["roll"], degrees=True))


def apply_shifts(sim, spec):
    """Modify camera mount poses (cam_pos/cam_quat, relative to the parent body)."""
    applied = {}
    for cam, s in spec.items():
        cid = sim.model.camera_name2id(cam)
        q = sim.model.cam_quat[cid].copy()  # wxyz
        r0 = R.from_quat([q[1], q[2], q[3], q[0]])
        r1 = r0 * local_rot(s)
        x, y, z, w = r1.as_quat()
        pos = sim.model.cam_pos[cid].copy() + r0.apply([0.0, 0.0, s.get("back_m", 0.0)])
        sim.model.cam_quat[cid] = [w, x, y, z]
        sim.model.cam_pos[cid] = pos
        applied[cam] = dict(parent_body=sim.model.body_id2name(int(sim.model.cam_bodyid[cid])),
                            orig_quat_wxyz=[float(v) for v in q],
                            new_pos=pos.tolist(), new_quat_wxyz=[float(w), float(x), float(y), float(z)])
    sim.forward()
    return applied


def make_env(canonical: Path, img: int):
    dm = json.load(open(canonical / "extras" / "dataset_meta.json"))
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
    return EnvUtils.create_env_for_data_processing(
        env_meta=dm["env_args"], camera_names=CAMERAS, camera_height=img, camera_width=img,
        reward_shaping=False)


def render_episode(env, canonical: Path, ep: int, spec: dict):
    ex = canonical / "extras" / f"episode_{ep:06d}"
    states = np.load(ex / "states.npz")["states"]
    with gzip.open(ex / "model.xml.gz", "rt") as f:
        xml = f.read()
    ep_meta = json.load(open(ex / "ep_meta.json"))
    env.reset()
    env.reset_to({"states": states[0], "model": xml, "ep_meta": json.dumps(ep_meta)})
    applied = apply_shifts(env.env.sim, spec)
    frames = {c: [] for c in CAMERAS}
    for t in range(len(states)):
        obs = env.reset_to({"states": states[t]})
        for c in CAMERAS:
            frames[c].append(np.asarray(obs[f"{c}_image"]).copy())
    return {c: np.stack(v) for c, v in frames.items()}, applied


def video_path(root: Path, ep: int, cam: str) -> Path:
    return root / "videos" / f"chunk-{ep // 1000:03d}" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"


def to_u8(x):
    if x.dtype == np.uint8:
        return x
    return np.clip(x * (255 if x.max() <= 1.0 else 1), 0, 255).astype(np.uint8)


def cmd_render(a):
    shifts = json.load(open(SHIFTS))
    spec = shift_spec(a.robot, a.variant, shifts)
    env = make_env(a.canonical, a.img)
    for ep in a.episodes:
        done_flag = a.out / ".render" / f"episode_{ep:06d}.json"
        if done_flag.exists():
            continue
        frames, applied = render_episode(env, a.canonical, ep, spec)
        for c in CAMERAS:
            # unshifted cams are re-rendered too (keeps the encoder identical across cams)
            encode_mp4(video_path(a.out, ep, c), to_u8(frames[c]))
        done_flag.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(episode=ep, frames=int(len(frames[CAMERAS[0]])), applied=applied), open(done_flag, "w"))
        print(f"[shift {a.variant}] {a.robot} ep {ep:06d} done", flush=True)


def md5(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mean_abs_diff(p1: Path, p2: Path, n=5):
    import av

    def read(p):
        with av.open(str(p)) as c:
            return [f.to_ndarray(format="rgb24").astype(np.float32) for f in c.decode(video=0)]
    a, b = read(p1), read(p2)
    idx = np.linspace(0, min(len(a), len(b)) - 1, n).astype(int)
    return float(np.mean([np.abs(a[i] - b[i]).mean() for i in idx])), len(a), len(b)


def cmd_finalize(a):
    shifts = json.load(open(SHIFTS))
    spec = shift_spec(a.robot, a.variant, shifts)
    info = json.load(open(a.canonical / "meta" / "info.json"))
    n_eps = info["total_episodes"]
    missing = [e for e in range(n_eps) if not (a.out / ".render" / f"episode_{e:06d}.json").exists()]
    assert not missing, f"episodes not rendered: {missing[:10]}..."
    for sub in ("data", "meta", "extras"):
        if (a.out / sub).exists():
            shutil.rmtree(a.out / sub)
        shutil.copytree(a.canonical / sub, a.out / sub)
    cam_cfg = dict(variant=a.variant, robot=a.robot, shifts=spec, convention=shifts["convention"],
                   unshifted=[c for c in CAMERAS if c not in spec], image_size=[a.img, a.img])
    json.dump(cam_cfg, open(a.out / "meta" / "camera_config.json", "w"), indent=2)
    # validation: frame counts, byte-identity of copied files, shifted cams differ / unshifted ~equal
    lengths = [json.loads(l)["length"] for l in open(a.canonical / "meta" / "episodes.jsonl")]
    report = dict(variant=a.variant, robot=a.robot, episodes=n_eps, problems=[], cam_diff_vs_canonical={})
    for e in range(n_eps):
        r = json.load(open(a.out / ".render" / f"episode_{e:06d}.json"))
        if r["frames"] != lengths[e]:
            report["problems"].append(f"ep {e}: {r['frames']} frames vs parquet {lengths[e]}")
    for e in sorted(set([0, n_eps // 2, n_eps - 1])):
        for c in CAMERAS:
            d, la, lb = mean_abs_diff(video_path(a.out, e, c), video_path(a.canonical, e, c))
            report["cam_diff_vs_canonical"].setdefault(c, []).append(round(d, 2))
            shifted = c in spec
            if la != lb:
                report["problems"].append(f"ep {e} {c}: {la} vs {lb} decoded frames")
            if shifted and d < 3.0:
                report["problems"].append(f"ep {e} {c}: shifted cam nearly identical to canonical ({d:.2f})")
            if not shifted and d > 3.0:
                report["problems"].append(f"ep {e} {c}: unshifted cam differs from canonical ({d:.2f})")
    for sub in ("data", "extras"):
        for p in (a.canonical / sub).rglob("*"):
            if p.is_file() and md5(p) != md5(a.out / p.relative_to(a.canonical)):
                report["problems"].append(f"not byte-identical: {p.relative_to(a.canonical)}")
    report["ok"] = not report["problems"]
    json.dump(report, open(a.out / "validation_report.json", "w"), indent=2)
    try:
        commit = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    with open(a.out / "provenance.jsonl", "a") as f:
        f.write(json.dumps(dict(time=dt.datetime.utcnow().isoformat() + "Z", canonical=str(a.canonical),
                                variant=a.variant, robot=a.robot, git_commit=commit,
                                shifts_file="scripts/regen/camera_shifts_v2.json", host=os.uname().nodename)) + "\n")
    shutil.rmtree(a.out / ".render")
    with open(a.out / "MANIFEST.md5", "w") as f:
        for p in sorted(a.out.rglob("*")):
            if p.is_file() and p.name != "MANIFEST.md5":
                f.write(f"{md5(p)}  {p.relative_to(a.out)}\n")
    print(f"[finalize] {a.out} ok={report['ok']} problems={report['problems'][:5]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["render", "finalize"])
    p.add_argument("--canonical", type=Path, required=True, help=".../<Robot>/<Task>/lerobot (canonical)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--variant", required=True, choices=["combined_shift_v2", "wrist_shift_v2", "front_shift_v2"])
    p.add_argument("--episodes", type=int, nargs="*", default=[])
    p.add_argument("--img", type=int, default=256)
    a = p.parse_args()
    cmd_render(a) if a.mode == "render" else cmd_finalize(a)


if __name__ == "__main__":
    main()
