"""Add the end-effector's image position to a recorded dataset.

Two layouts are handled, because the two datasets store different things:

  pretrain (barx_panda_ur5e_iiwa)
      observation.state carries the eef in the mobilebase0_support frame, and
      each episode's ep_meta gives the camera in that same frame, so the
      projection needs no simulator at all -- the base pose cancels.
      Verified against the simulator's own camera: 0.0000 px over 108 frames
      spanning three robots.

  cross-embodiment (new_barx)
      observation.state is the eef in world coordinates and the camera was
      pinned in world space at generation time and not recorded, so it is
      recovered per episode by rebuilding the source robot with that
      episode's ep_meta.

Two columns are written:
  eef_pixel     float32[2]  (u, v) in pixels, valid only where in-frame
  eef_in_frame  bool        the eef is in front of the camera and inside the
                            image; (u, v) is left as the raw projection
                            otherwise so it can still be used as a direction
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data1/local/robocasa")
os.environ.setdefault("MUJOCO_GL", "egl")


def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def project_many(pts, cam_p, cam_R, fovy, h, w):
    """Vectorised pinhole projection; MuJoCo cameras look down -Z, +Y up."""
    f = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    pc = (np.asarray(pts, dtype=float) - cam_p) @ cam_R  # == (R^T @ v)^T
    depth = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = w / 2.0 + f * (pc[:, 0] / depth)
        v = h / 2.0 - f * (pc[:, 1] / depth)
    ok = (depth > 1e-9) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = np.stack([u, v], axis=1)
    uv[~np.isfinite(uv)] = -1.0
    return uv.astype(np.float32), ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=Path, required=True, help="lerobot dir to annotate"
    )
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="dataset holding extras/episode_*/ep_meta.json",
    )
    ap.add_argument("--layout", choices=["pretrain", "cross"], required=True)
    ap.add_argument("--robot", default="UR5eOmron", help="cross layout: source robot")
    ap.add_argument("--camera", default="robot0_agentview_right")
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument(
        "--episode-map",
        type=Path,
        default=None,
        help="cross layout: JSON {new_barx_ep: source_ep}. new_barx "
        "episode N is NOT source episode N -- 48 of 108 differ "
        "even in length -- so the camera must be looked up "
        "through this mapping. Episodes absent from it get "
        "eef_in_frame=False and eef_pixel=nan rather than a "
        "guessed camera.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    H, W = args.height, args.width
    pqs = sorted(args.dataset.glob("data/chunk-*/episode_*.parquet"))
    if not pqs:
        print("no parquet found", file=sys.stderr)
        return 2

    ep_map = None
    if args.episode_map is not None:
        ep_map = {
            int(k): int(v)
            for k, v in json.loads(args.episode_map.read_text())["mapping"].items()
        }
    world_cam = None
    if args.layout == "cross":
        import generate_visual_robust_cross_embodiment_x as G

        G.patch_python_paths()
        import robocasa  # noqa: F401

        G.install_mesh_inertia_compat()
        import custom_robots  # noqa: F401

        G.install_recorded_camera_override()
        G.install_object_registry_fallback()
        env_args = json.loads(
            (args.source / "extras" / "dataset_meta.json").read_text()
        )["env_args"]

        def world_cam(ep):
            states, _, ep_meta, _, _ = G.load_episode(args.source, ep)
            env = G.create_env(
                env_args, args.robot, [args.camera], H, W, 300.0, is_source_robot=True
            )
            w = G._EnvWithCameras(env, [args.camera])
            w.env.set_ep_meta(dict(ep_meta))
            w.reset(unset_ep_meta=False)
            be = w.env
            be.sim.set_state_from_flattened(states[0])
            be.sim.forward()
            m, d = be.sim.model, be.sim.data
            cid = m.camera_name2id(args.camera)
            res = (
                np.array(d.cam_xpos[cid], dtype=float),
                np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3),
                float(m.cam_fovy[cid]),
            )
            be.close()
            return res

    n_in = n_tot = 0
    for pq in pqs:
        ep = int(pq.stem.split("_")[1])
        df = pd.read_parquet(pq)
        st = np.stack(df["observation.state"].to_numpy())

        if args.layout == "pretrain":
            # eef and camera share the mobilebase0_support frame
            meta = json.loads(
                (
                    args.source / "extras" / f"episode_{ep:06d}" / "ep_meta.json"
                ).read_text()
            )
            cc = meta["cam_configs"][args.camera]
            cam_p = np.array(cc["pos"], dtype=float)
            cam_R = quat_wxyz_to_mat(cc["quat"])
            fovy = float(cc["camera_attribs"]["fovy"])
            pts = st[:, 7:10]
        else:
            src_ep = ep_map.get(ep) if ep_map is not None else ep
            if src_ep is None:
                # no camera could be established for this episode; ship an
                # explicit gap instead of a plausible-looking wrong pixel
                uv = np.full((len(st), 2), np.nan, dtype=np.float32)
                ok = np.zeros(len(st), dtype=bool)
                df["eef_pixel"] = list(uv)
                df["eef_in_frame"] = ok
                out = args.out / pq.relative_to(args.dataset)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
                n_tot += len(ok)
                print(f"  ep {ep}: no camera -> pixels marked invalid", flush=True)
                continue
            cam_p, cam_R, fovy = world_cam(src_ep)
            pts = st[:, 0:3]  # already world coordinates

        uv, ok = project_many(pts, cam_p, cam_R, fovy, H, W)
        df["eef_pixel"] = list(uv)
        df["eef_in_frame"] = ok
        out = args.out / pq.relative_to(args.dataset)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        n_in += int(ok.sum())
        n_tot += len(ok)
        if ep % 100 == 0:
            print(
                f"  ep {ep}: {len(df)} frames, in-frame {100*ok.mean():.1f}%",
                flush=True,
            )

    # carry meta across, declaring the new columns
    import shutil

    for m in sorted((args.dataset / "meta").glob("*")):
        dst = args.out / "meta" / m.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if m.name == "info.json":
            info = json.loads(m.read_text())
            info.setdefault("features", {})
            info["features"]["eef_pixel"] = {
                "dtype": "float32",
                "shape": [2],
                "names": ["u", "v"],
            }
            info["features"]["eef_in_frame"] = {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            }
            dst.write_text(json.dumps(info, indent=4))
        else:
            shutil.copy2(m, dst)

    print(
        f"done: {len(pqs)} episodes, {n_tot} frames, "
        f"in-frame {100*n_in/max(n_tot,1):.1f}% -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
