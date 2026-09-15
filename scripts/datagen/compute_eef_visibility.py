"""Add end-effector visibility columns to an already-built self-workspace shard.

The shipped parquet records where the end-effector was, not whether it could
actually be seen: the generator's in-frame test projects the eef into the
image and checks the pixel lands inside the frame, which passes even when the
eef sits behind one of the arm's own links. Measured on accepted poses, the
eef is hidden in 21-37% of frames depending on the arm.

The arm configuration behind each frame is not stored, and IK from the eef
pose alone would recover a different arm. It does not need to be stored: the
generator is deterministic, so re-running it with rendering stubbed out
reproduces the same joint path (verified to 0.02mm against shipped data) and
the visibility can be measured on the reproduced states. The existing videos
stay valid because the poses are identical.

Visibility is measured by ray casting from the camera, not to a single point
but to a ball of points around the eef, so a frame whose surroundings are
occluded is also marked down. Two columns are written per embodiment:

  eef_visible_frac.<tag>   fraction of the ball's rays that reach unobstructed
  eef_visible_center.<tag> whether the eef point itself is unobstructed

Threshold at use time: `frac == 1.0` keeps only frames whose neighbourhood is
entirely clear; lower cut-offs trade strictness for data volume.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco

sys.path.insert(0, "/data1/local/robocasa")
os.environ.setdefault("MUJOCO_GL", "egl")

import generate_visual_robust_cross_embodiment_x as G

G.patch_python_paths()
import robocasa  # noqa: F401

G.install_mesh_inertia_compat()
import custom_robots  # noqa: F401

G.install_recorded_camera_override()
G.install_object_registry_fallback()
import generate_self_workspace_dataset as S

BALL_RADIUS_M = 0.10
N_RAYS = 14
SURFACE_EPS_M = 0.02  # ignore hits within this of the target: that is the target itself


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lerobot-dir", type=Path, required=True)
    ap.add_argument("--robots", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument(
        "--camera-names",
        nargs="+",
        default=["robot0_agentview_right", "robot0_eye_in_hand"],
    )
    ap.add_argument("--camera-height", type=int, default=180)
    ap.add_argument("--camera-width", type=int, default=320)
    ap.add_argument("--hide-scope", default="no_kitchen")
    ap.add_argument("--style-sweep", action="store_true")
    ap.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="shipped parquet; reproduction is checked against it",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--real-render",
        action="store_true",
        help="render for real instead of stubbing it. MuJoCo warm-starts "
        "its contact solver, so skipping the render loop's forward() "
        "calls can flip a contact right at the 2mm tolerance and send "
        "the path down a different branch.",
    )
    args = ap.parse_args()

    H, W = args.camera_height, args.camera_width
    cams = args.camera_names
    vis_cam = cams[0]

    if not args.real_render:
        # Rendering consumes no rng, but it does advance sim state, so this is
        # only safe where reproduction is verified to stay within tolerance.
        G.render_cameras = lambda ew, be, cs, h, w: {
            c: np.zeros((h, w, 3), np.uint8) for c in cs
        }

    env_args = json.loads(
        (args.source_lerobot_dir / "extras" / "dataset_meta.json").read_text()
    )["env_args"]
    specs = G.parse_robot_specs(args.robots)
    _, _, ep_meta, _, _ = G.load_episode(args.source_lerobot_dir, 0)
    ep_meta = dict(ep_meta)
    if args.style_sweep:
        ep_meta["style_id"] = 0  # one episode per shard, so ep index is 0

    shipped = pd.read_parquet(args.verify_against) if args.verify_against else None
    rng = np.random.default_rng(args.seed)

    cols = {}
    worst_repro = 0.0
    for spec in specs:
        env = G.create_env(
            env_args,
            spec.robot_class,
            cams,
            H,
            W,
            300.0,
            is_source_robot=False,
            gripper_override=spec.gripper,
        )
        w = G._EnvWithCameras(env, cams)
        w.env.set_ep_meta(dict(ep_meta))
        w.reset(unset_ep_meta=False)
        be = w.env

        # build_episode replays the sampling and leaves the sim on the last
        # frame, so the per-frame states are collected from its own buffer
        buf, _ = S.build_episode(
            w,
            be,
            spec.robot_class,
            args.steps,
            cams,
            H,
            W,
            rng,
            args.hide_scope,
            vis_cam,
        )

        cols[f"eef_visible_frac.{spec.tag}"] = np.array(
            buf["eef_visible_frac"], dtype=np.float32
        )
        cols[f"eef_visible_center.{spec.tag}"] = np.array(
            buf["eef_visible_center"], dtype=bool
        )

        if shipped is not None:
            col = f"eef_state.{spec.tag}"
            if col in shipped.columns:
                want = np.stack(shipped[col].to_numpy())[:, :3].astype(float)
                got = np.array(buf["fingertip_pos"], dtype=float)
                e = float(np.abs(got - want).max()) * 1000
                worst_repro = max(worst_repro, e)
                print(
                    f"    {spec.tag:24s} repro {e:9.4f} mm"
                    f"{'' if e < 0.1 else '   <-- DIVERGED'}",
                    flush=True,
                )
        be.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(args.out)
    print(
        f"[vis] seed={args.seed} tags={len(specs)} "
        f"repro_err={worst_repro:.4f}mm -> {args.out}",
        flush=True,
    )
    # Reproduction drift beyond storage precision means the states are not the
    # shipped ones and the flags would describe different frames.
    return 0 if worst_repro < 0.1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
