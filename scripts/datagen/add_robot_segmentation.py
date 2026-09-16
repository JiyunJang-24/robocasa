#!/usr/bin/env python3
"""Add a robot segmentation mask video to an existing single-embodiment atomic
LeRobot dataset (the format `generate_mimicgen_ur5e_dataset.py` /
`atomic_new` produce -- one robot, video-backed image columns).

Why this can be done post-hoc, cheaply and exactly: `extras/episode_*/states.npz`
already holds the FULL simulator state for every frame (not just the initial
one), because that is what `generate_visual_robust_cross_embodiment_x.py`
loads to seed cross-embodiment replay. Restoring `states[t]` with
`sim.set_state_from_flattened` and rendering reproduces the exact physical
configuration the original frame was rendered from -- no action replay, so
none of the open-loop drift documented in that script's `replay_embodiment`
(measured up to ~90cm by end-of-episode) applies here. The recorded episode's
own model XML is used unmodified (no embodiment/gripper swap), so there is no
DOF-layout mismatch to work around either.

Segmentation itself is `sim.render(..., segmentation=True)`, which returns
(H, W, 2) int32 (geom_type, geom_id) per pixel -- this is intentionally the
low-level MjSim call, not the env-constructor `camera_segmentations` kwarg:
robocasa_x's Kitchen wrapper does not forward that kwarg (`TypeError:
unexpected keyword argument`), while the low-level render works on any already
-built env. A geom belongs to the robot if its parent body's name starts with
`robot0_`, `gripper0_`, or `mobilebase0_`, minus the bodies in
`MASK_EXCLUDED_BODIES` (see the comment there for why the mobile base's
support column is real geometry that is nonetheless left out).

Output: a copy of the dataset (originals untouched) with one new video key
per source camera, `observation.images.<camera>_segmentation`, holding a
single-channel (replicated to 3) 0/255 mask, plus the corresponding entries
appended to `meta/info.json`'s `features`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "datagen"))

FPS = 20
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")


_JOINT_SIZES = None  # filled in lazily, needs mujoco's enum


def prepare_joint_restore(old_model_xml: str, sim):
    """One-time (per episode) setup for `restore_robot_state_by_joint_name`.

    `create_env`'s scene is resampled through robocasa's Kitchen object
    sampler, not loaded from the recorded model.xml verbatim, so its object
    set (hence total nq/nv) does not match the exact recording even for the
    un-swapped source robot -- confirmed here (78 vs 90 qvel on the first
    tested episode). A flat `set_state_from_flattened` therefore fails.
    Object pose does not matter for a *robot* mask, so the fix is to copy
    only joints whose name is under the robot's own bodies, by name (same
    technique as `restore_arm_state_by_joint_name` in the generator script,
    extended to the gripper too -- that function skips it deliberately for a
    different purpose, establishing a cross-embodiment reference trajectory;
    here matching the recorded gripper open/close state is exactly the point).

    Parses the recorded XML and resolves every joint address ONCE per episode
    -- `mujoco.MjModel.from_xml_string` on a full kitchen scene is not free,
    and calling it per-frame (the first version of this function did) turned
    a ~1-2 minute episode into an 11-minute one for no benefit, since none of
    this depends on the frame index.
    """
    global _JOINT_SIZES
    import mujoco

    if _JOINT_SIZES is None:
        _JOINT_SIZES = {
            mujoco.mjtJoint.mjJNT_FREE: (7, 6),
            mujoco.mjtJoint.mjJNT_BALL: (4, 3),
            mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
            mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
        }

    old = mujoco.MjModel.from_xml_string(old_model_xml)
    nq_old, nv_old = old.nq, old.nv

    model = sim.model
    plan = []  # (qa_old, va_old, new_qa, new_va, nq, nv)
    for j in range(old.njnt):
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j)
        if (
            not name
            or not name.startswith(ROBOT_BODY_PREFIXES)
            or name not in model.joint_names
        ):
            continue
        nq, nv = _JOINT_SIZES[mujoco.mjtJoint(old.jnt_type[j])]
        new_qa = model.get_joint_qpos_addr(name)
        new_va = model.get_joint_qvel_addr(name)
        plan.append(
            (
                old.jnt_qposadr[j],
                old.jnt_dofadr[j],
                new_qa[0] if isinstance(new_qa, tuple) else new_qa,
                new_va[0] if isinstance(new_va, tuple) else new_va,
                nq,
                nv,
            )
        )
    return {"nq_old": nq_old, "nv_old": nv_old, "plan": plan}


def restore_robot_state_by_joint_name(
    sim, restore_plan: dict, flat_state: np.ndarray
) -> int:
    state = np.asarray(flat_state, dtype=np.float64).reshape(-1)
    nq_old = restore_plan["nq_old"]
    qpos_old = state[1 : 1 + nq_old]
    qvel_old = state[1 + nq_old : 1 + nq_old + restore_plan["nv_old"]]

    data = sim.data
    for qa_o, va_o, new_qa, new_va, nq, nv in restore_plan["plan"]:
        data.qpos[new_qa : new_qa + nq] = qpos_old[qa_o : qa_o + nq]
        data.qvel[new_va : new_va + nv] = qvel_old[va_o : va_o + nv]
    sim.forward()
    return len(restore_plan["plan"])


# Excluded from the MASK only (not from joint restoration -- this body's own
# joint still has to be positioned correctly, since it sets the height the
# arm mounts at). `mobilebase0_support`/`mobilebase0_fixed_support` is the
# Omron base's real connecting column between the wheeled base and the arm
# mount, verified by connected-component analysis of the mask: an isolated
# ~20x70px sliver made of nothing but this one geom, decoupled from every
# other robot component. It is genuine geometry, not a rendering artifact --
# but from this camera angle most of it sits directly behind the arm links,
# so only a thin, disconnected edge peeks out on either side. A blob that
# shape reads as a stray pointer/indicator line in the mask rather than as
# "part of the robot," so it is dropped here even though it is real.
MASK_EXCLUDED_BODIES = ("mobilebase0_support", "mobilebase0_fixed_support")


def robot_geom_ids(model) -> set[int]:
    ids = set()
    for gi in range(model.ngeom):
        bname = model.body_id2name(model.geom_bodyid[gi])
        if (
            bname
            and bname.startswith(ROBOT_BODY_PREFIXES)
            and bname not in MASK_EXCLUDED_BODIES
        ):
            ids.add(gi)
    return ids


def write_mask_video(path: Path, masks: list[np.ndarray]) -> None:
    import imageio_ffmpeg

    h, w = masks[0].shape
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(w, h),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="gray",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "18"],  # masks are flat regions; keep edges crisp
    )
    writer.send(None)
    try:
        for m in masks:
            writer.send(np.ascontiguousarray((m.astype(np.uint8) * 255)))
    finally:
        writer.close()


def process_episode(
    source_dir: Path,
    out_dir: Path,
    episode_index: int,
    camera_names: list[str],
    save_preview_to: Path | None = None,
) -> int:
    from generate_visual_robust_cross_embodiment_x import (
        create_env,
        install_mesh_inertia_compat,
        install_recorded_camera_override,
        load_episode,
        patch_python_paths,
        resolve_env_name,
    )

    patch_python_paths()
    install_mesh_inertia_compat()
    install_recorded_camera_override()
    import robocasa  # noqa: F401 - registers robocasa_x envs once paths are patched

    states, raw_actions, ep_meta, model_xml, env_args = load_episode(
        source_dir, episode_index
    )
    source_robot = env_args["env_kwargs"]["robots"]
    if isinstance(source_robot, (list, tuple)):
        source_robot = source_robot[0]

    env = create_env(
        env_args,
        source_robot,
        camera_names,
        180,
        320,
        osc_kp=300.0,
        is_source_robot=True,
        gripper_override=None,
    )
    # create_env's own construction resets once with no ep_meta set, so
    # set_cameras() (even patched) has nothing recorded to read yet and the
    # camera stays at robocasa_x's freshly-derived default -- for
    # "robot0_agentview_right" specifically, that default sits on the
    # opposite side from what atomic_new actually recorded under that name.
    # Setting ep_meta and resetting again (same sequence
    # replay_embodiment/process_episode uses in the main generator) is what
    # makes the override actually take effect.
    env.env.set_ep_meta(dict(ep_meta))
    env.reset(unset_ep_meta=False)
    # env.reset() rebuilds the sim -- any handle grabbed before this point is
    # dangling now, so sim/model must be (re)fetched only after it.
    sim = env.env.sim
    model = sim.model
    rgeoms = robot_geom_ids(model)

    restore_plan = prepare_joint_restore(model_xml, sim)
    if not restore_plan["plan"]:
        raise RuntimeError(
            f"episode {episode_index}: no robot joints matched by name between "
            "the recorded model.xml and this env build -- check ROBOT_BODY_PREFIXES"
        )

    masks = {cam: [] for cam in camera_names}
    for t in range(len(states)):
        restore_robot_state_by_joint_name(sim, restore_plan, states[t])
        for cam in camera_names:
            seg = sim.render(width=320, height=180, camera_name=cam, segmentation=True)
            mask = np.isin(seg[..., 1], list(rgeoms))
            masks[cam].append(
                np.flipud(mask)
            )  # match observation.image's flipud storage

    env.env.close()

    for cam in camera_names:
        dst = out_dir / "videos" / "chunk-000" / f"{cam}_segmentation"
        dst.mkdir(parents=True, exist_ok=True)
        write_mask_video(dst / f"episode_{episode_index:06d}.mp4", masks[cam])

    if save_preview_to is not None:
        import imageio.v2 as iio

        cam = camera_names[0]
        rgb_video = (
            source_dir
            / "videos"
            / "chunk-000"
            / cam
            / f"episode_{episode_index:06d}.mp4"
        )
        r = iio.get_reader(str(rgb_video))
        n = min(r.count_frames(), len(masks[cam]))
        idx = np.linspace(0, n - 1, 4).astype(int)
        rows = []
        for i in idx:
            rgb = r.get_data(i)
            m = masks[cam][i]
            overlay = rgb.copy()
            overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([255, 0, 0])).astype(
                np.uint8
            )
            rows.append(
                np.hstack([rgb, np.dstack([m * 255] * 3).astype(np.uint8), overlay])
            )
        r.close()
        iio.imwrite(str(save_preview_to), np.vstack(rows))
        print(f"[preview] wrote {save_preview_to}")

    return len(states)


def copy_episode_data(source_dir: Path, out_dir: Path, episode_index: int) -> None:
    chunk = episode_index // 1000
    rel = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    if not (out_dir / rel).exists():
        shutil.copy2(source_dir / rel, out_dir / rel)
    for cam_dir in (source_dir / "videos" / f"chunk-{chunk:03d}").iterdir():
        if not cam_dir.is_dir():
            continue
        dst_dir = out_dir / "videos" / f"chunk-{chunk:03d}" / cam_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_file = cam_dir / f"episode_{episode_index:06d}.mp4"
        if src_file.exists() and not (dst_dir / src_file.name).exists():
            shutil.copy2(src_file, dst_dir / src_file.name)


def finalize_meta(source_dir: Path, out_dir: Path, camera_names: list[str]) -> None:
    meta_src, meta_dst = source_dir / "meta", out_dir / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for f in meta_src.iterdir():
        if f.is_file() and not (meta_dst / f.name).exists():
            shutil.copy2(f, meta_dst / f.name)
    info_path = meta_dst / "info.json"
    info = json.loads(info_path.read_text())
    for cam in camera_names:
        key = f"observation.images.{cam}_segmentation"
        if key not in info["features"]:
            base = info["features"][f"observation.images.{cam}"]
            seg_feat = json.loads(json.dumps(base))
            seg_feat["names"] = ["height", "width", "channel"]
            info["features"][key] = seg_feat
    info_path.write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lerobot-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument(
        "--camera-names",
        nargs="+",
        default=["robot0_agentview_right", "robot0_eye_in_hand"],
    )
    ap.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="write an RGB|mask|overlay contact sheet per episode here",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes:
        preview = None
        if args.preview_dir is not None:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            preview = args.preview_dir / f"episode_{ep:06d}_check.png"
        n = process_episode(
            args.source_lerobot_dir,
            args.output_dir,
            ep,
            args.camera_names,
            save_preview_to=preview,
        )
        copy_episode_data(args.source_lerobot_dir, args.output_dir, ep)
        print(f"[done] episode {ep}: {n} frames, mask videos + copied data/rgb written")

    finalize_meta(args.source_lerobot_dir, args.output_dir, args.camera_names)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
