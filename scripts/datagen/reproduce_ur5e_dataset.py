#!/usr/bin/env python3
"""
Final production script: reproduce RoboCasa atomic PandaOmron demos on UR5eOmron.

Per episode:
  1. Reset the source env to the recorded initial state (states.npz + model.xml + ep_meta).
  2. Reset the target UR5eOmron env with the same ep_meta, align base x/y/yaw to the
     source low-dim state.
  3. Collision-aware base backoff: if any robot-vs-scene contact exists after IK at
     the source EEF pose, retreat the base along its own facing axis in 5 cm steps
     (up to --max-backoff) until IK converges with zero contacts. Most atomic tasks
     (57/66) have zero base motion, so a static backoff start is a valid initial
     condition; the world-frame EEF trajectory is still reproduced by tracking.
  4. Closed-loop EEF tracking replay: each step reads the source's *recorded* state
     (no source dynamics drift), computes the OSC delta that chases the source EEF
     pose, and copies base/torso/gripper/base_mode from the source action.
  5. Keep the episode only if the target env's own success check passes
     (unless --keep-failed).

Outputs (robomimic-style):
  <output-root>/<target-robot>/<task>/<date>/raw.hdf5
  <output-root>/<target-robot>/<task>/<date>/videos/episode_XXXXXX_success-K.mp4
  <output-root>/<target-robot>/<task>/<date>/summary.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

import robosuite.utils.transform_utils as T

from dataset_reproduce import (
    init_obs_utils,
    load_episode_payload,
    load_modality,
    read_episode_df,
    episode_paths,
    build_target_env_meta,
    create_env,
    reset_source_to_episode,
    reset_target_to_episode,
    align_base_to_source_lowdim,
    solve_ik_to_eef_pose,
    robot_eef_pose,
    reorder_lerobot_actions_to_hdf5,
    flatten_index_container,
    write_episode_video,
    write_raw_hdf5,
    render_frame,
)
from test_dataset import (
    sync_controller_goals,
    retarget_action,
    source_eef_pose_from_saved_state,
)
import robocasa.utils.env_utils as CoreEnvUtils

ATOMIC_ROOT = Path("/data1/local/robocasa/datasets/v1.0/pretrain/atomic")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, help="Atomic task name, e.g. CloseDrawer"
    )
    parser.add_argument(
        "--source-lerobot-dir",
        type=Path,
        default=None,
        help="Override source dir; default resolves under the atomic root.",
    )
    parser.add_argument("--target-robot", default="UR5eOmron")
    parser.add_argument("--target-gripper", default="default")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data1/local/robocasa/datasets/reproduced_ur5e"),
    )
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--eef-gain", type=float, default=1.0)
    parser.add_argument(
        "--eef-rot-gain",
        type=float,
        default=2.5,
        help="Rotation tracking gain. 2.5 drops rot error from ~3-8 deg to <1 deg "
        "with no position cost (measured on AdjustToasterOvenTemperature).",
    )
    parser.add_argument(
        "--no-twist-align",
        action="store_true",
        help="Disable gripper opening-axis alignment (Rz twist of the desired orientation).",
    )
    parser.add_argument(
        "--no-pad-friction-boost",
        action="store_true",
        help="Keep the target gripper's stock pad friction instead of matching Panda's.",
    )
    parser.add_argument(
        "--grasp-depth-extra",
        type=float,
        default=0.01,
        help="Extra approach-depth (m) added to the grasp-center offset; compensates the "
        "Robotiq linkage swinging its pads forward as it closes.",
    )
    parser.add_argument("--ik-iters", type=int, default=250)
    parser.add_argument("--max-backoff", type=float, default=0.30)
    parser.add_argument("--backoff-step", type=float, default=0.05)
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Also store failed rollouts in the hdf5 (marked success=0).",
    )
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_source_dir(args) -> Path:
    if args.source_lerobot_dir is not None:
        return args.source_lerobot_dir
    candidates = sorted((ATOMIC_ROOT / args.task).glob("*/lerobot"))
    if not candidates:
        raise FileNotFoundError(
            f"No lerobot dir found for task {args.task} under {ATOMIC_ROOT}"
        )
    return candidates[-1]


def count_robot_scene_contacts(env, arm_only=False):
    """Robot-vs-scene contact count, ignoring the mobile base.

    arm_only=True counts only arm-link (robot0_*) contacts and ignores the
    gripper: task contact (gripper touching the drawer/handle/knob) is expected
    and desired, but arm links pressing into counters/cabinet doors is exactly
    the failure mode that pins the robot and breaks tracking.
    """
    sim = env.env.sim
    n = 0
    pairs = []
    for ci in range(sim.data.ncon):
        con = sim.data.contact[ci]
        g1 = sim.model.geom_id2name(con.geom1) or ""
        g2 = sim.model.geom_id2name(con.geom2) or ""
        robot_prefixes = ("robot0",) if arm_only else ("robot0", "gripper0")
        r1 = g1.startswith(robot_prefixes)
        r2 = g2.startswith(robot_prefixes)
        if r1 == r2:
            continue
        scene = g2 if r1 else g1
        if scene.startswith(("mobilebase0", "gripper0")):
            continue
        n += 1
        pairs.append((g1, g2))
    return n, pairs


def base_world_pose(env):
    sim = env.env.sim
    bid = sim.model.body_name2id("mobilebase0_support")
    pos = sim.data.body_xpos[bid].copy()
    yaw = T.mat2euler(sim.data.body_xmat[bid].reshape(3, 3))[2]
    return pos, yaw


def move_base_backward(env, backoff: float, base_z: float):
    pos, yaw = base_world_pose(env)
    new_pos = np.array(
        [
            pos[0] - backoff * np.cos(yaw),
            pos[1] - backoff * np.sin(yaw),
            base_z,
        ]
    )
    CoreEnvUtils.set_robot_to_position(env.env, new_pos)


ROBOT_NAME_PREFIXES = ("robot0", "gripper0", "mobilebase0")


def opening_axis_in_site_frame(env):
    """Unit vector (site frame, approach-axis component removed) along which the
    finger pads separate. Panda opens along one site axis, Robotiq85 along the
    other (90 deg twist) -- aligning these makes the target straddle objects the
    same way the source did instead of pinching them from a rotated direction."""
    sim = env.env.sim
    sp, sR = robot_eef_pose(env)
    groups = {}
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ""
        if (
            name.startswith("gripper0")
            and "pad" in name.lower()
            and "visual" not in name
        ):
            side = "l" if ("left" in name or "finger2" in name) else "r"
            groups.setdefault(side, []).append(sim.data.geom_xpos[gid].copy())
    if len(groups) < 2:
        return np.array([0.0, 1.0, 0.0])
    d_site = sR.T @ (np.mean(groups["r"], axis=0) - np.mean(groups["l"], axis=0))
    d_site[2] = 0.0
    n = np.linalg.norm(d_site)
    return d_site / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def gripper_twist_matrix(source_env, target_env):
    """Rz(angle) aligning the target gripper's opening axis with the source's,
    folded into [-90, 90] deg (a parallel gripper is symmetric under 180 deg)."""
    ax_src = opening_axis_in_site_frame(source_env)
    ax_tgt = opening_axis_in_site_frame(target_env)
    ang = np.arctan2(ax_tgt[1], ax_tgt[0]) - np.arctan2(ax_src[1], ax_src[0])
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def boost_pad_friction(env, friction=(2.0, 0.05, 1.0e-4)):
    """Raise the target gripper pad friction to the source Panda pad level.

    The robosuite Robotiq85 pads ship with sliding friction 0.8 / torsional
    5e-3 vs the Panda pads' 2.0 / 5e-2. Combined with the knob joint's
    frictionloss, the slick pads slip where the Panda grips; measured knob
    rotation went from 0.000 to +0.06 rad after boosting.
    """
    sim = env.env.sim
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ""
        if (
            name.startswith("gripper0")
            and "pad" in name.lower()
            and "visual" not in name
        ):
            sim.model.geom_friction[gid] = np.asarray(friction)


def grasp_center_in_site_frame(env):
    """Fingertip-pad midpoint expressed in the eef-site frame.

    Different grippers put their grip_site at different depths relative to the
    fingertip pads (measured: Panda -0.4 cm vs Robotiq85 -1.4 cm along the
    approach axis). Tracking raw site poses therefore leaves the target's pads
    ~1 cm shy of where the source's pads were -- enough for a knob grasp to
    pinch the rim and slip instead of wrapping the knob. We align the pad
    midpoints ("grasp centers") instead.
    """
    sim = env.env.sim
    sp, sR = robot_eef_pose(env)
    pads = []
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ""
        if (
            name.startswith("gripper0")
            and "pad" in name.lower()
            and "visual" not in name
        ):
            pads.append(sim.data.geom_xpos[gid].copy())
    if not pads:
        return np.zeros(3)
    return sR.T @ (np.mean(pads, axis=0) - sp)


def sync_scene_to_source(source_env, target_env):
    """Force the target env's scene to match the source episode's scene exactly.

    Why this is necessary: robocasa re-samples some scene placements on every
    env.reset() -- notably appliance fixture groups like the toaster oven,
    which are positioned via model body poses (not free joints). Only the very
    first reset of a freshly-seeded env happens to reproduce the source
    episode's placements; every later reset draws NEW positions, so the
    (correctly tracked!) source EEF trajectory ends up manipulating empty air
    tens of cm away from where the appliance now is. Verified empirically:
    the toaster oven moved 13-33 cm between retry-ladder attempts while all
    joint qpos still matched.

    Fix: copy every non-robot body pose (model.body_pos/quat) and every
    non-robot joint qpos from the source env (which was restored exactly via
    reset_to(states[0] + model.xml)) into the target model/data by name.
    """
    ssim, tsim = source_env.env.sim, target_env.env.sim
    for tbid in range(tsim.model.nbody):
        name = tsim.model.body_id2name(tbid) or ""
        if (not name) or name == "world" or name.startswith(ROBOT_NAME_PREFIXES):
            continue
        try:
            sbid = ssim.model.body_name2id(name)
        except Exception:
            continue
        tsim.model.body_pos[tbid] = ssim.model.body_pos[sbid]
        tsim.model.body_quat[tbid] = ssim.model.body_quat[sbid]
    for tjid in range(tsim.model.njnt):
        name = tsim.model.joint_id2name(tjid) or ""
        if (not name) or name.startswith(ROBOT_NAME_PREFIXES):
            continue
        try:
            sjid = ssim.model.joint_name2id(name)
        except Exception:
            continue
        jtype = int(tsim.model.jnt_type[tjid])
        width = 7 if jtype == 0 else (4 if jtype == 1 else 1)
        tadr = tsim.model.jnt_qposadr[tjid]
        sadr = ssim.model.jnt_qposadr[sjid]
        tsim.data.qpos[tadr : tadr + width] = ssim.data.qpos[sadr : sadr + width]
    tsim.data.qvel[:] = 0.0
    tsim.forward()


def place_target_at_backoff(
    target_env, ep_meta, lowdim0, backoff, src_pos, src_xmat, args, source_env=None
):
    """Reset + scene sync + base align (+ optional backoff) + IK to the source
    initial EEF pose.

    Returns (ik_residual, init_arm_contacts).
    """
    reset_target_to_episode(target_env, ep_meta)
    if source_env is not None:
        sync_scene_to_source(source_env, target_env)
    align_base_to_source_lowdim(target_env, lowdim0)
    if backoff > 0.0:
        move_base_backward(target_env, backoff, base_z=float(lowdim0[2]))
    solve_ik_to_eef_pose(target_env, src_pos, src_xmat, n_iters=args.ik_iters)
    tp, _ = robot_eef_pose(target_env)
    residual = float(np.linalg.norm(tp - src_pos))
    ncon, _ = count_robot_scene_contacts(target_env, arm_only=True)
    sync_controller_goals(target_env)
    return residual, ncon


def main():
    args = parse_args()
    init_obs_utils()

    source_dir = resolve_source_dir(args)
    with open(source_dir / "extras" / "dataset_meta.json") as f:
        dataset_meta = json.load(f)
    source_env_meta = dataset_meta["env_args"]
    target_env_meta = build_target_env_meta(
        source_env_meta, args.target_robot, args.target_gripper
    )
    modality = load_modality(source_dir)

    task_name = source_env_meta["env_name"]
    date_tag = dt.datetime.now().strftime("%Y%m%d")
    output_dir = args.output_root / args.target_robot / task_name / date_tag
    raw_path = output_dir / "raw.hdf5"
    video_dir = output_dir / "videos"

    print(f"[source] {source_dir}")
    print(f"[target] {args.target_robot}/{args.target_gripper}")
    print(f"[output] {output_dir}")

    source_env = create_env(source_env_meta, seed=args.seed)
    target_env = create_env(target_env_meta, seed=args.seed)

    episodes = []
    stats = []
    try:
        paths = episode_paths(source_dir, args.start_episode, args.max_episodes)
        for parquet_path in tqdm(paths, desc=f"reproduce-{task_name}"):
            episode_num = int(parquet_path.stem.split("_")[-1])
            try:
                episode_df = read_episode_df(parquet_path)
                source_states, ep_meta, model_xml = load_episode_payload(
                    source_dir, episode_num
                )
                source_actions = reorder_lerobot_actions_to_hdf5(
                    np.stack(episode_df["action"].to_numpy()), modality=modality
                )
                lowdim0 = np.asarray(
                    episode_df.iloc[0]["observation.state"], dtype=np.float64
                )

                reset_source_to_episode(
                    source_env, source_states[0], ep_meta, model_xml
                )
                sync_controller_goals(source_env)
                src_pos0, src_xmat0 = robot_eef_pose(source_env)
                grasp_center_src = grasp_center_in_site_frame(source_env)

                # Retry ladder over base backoffs, using REAL rollouts.
                # A kinematic-sweep prefilter proved over-conservative (IK-branch
                # phantom contacts disqualified rollouts that actually succeed,
                # and large backoffs push the arm to its reach limit), so the
                # only reliable feasibility oracle is the rollout itself: try the
                # smallest backoff first, escalate only on failure. Initial-pose
                # arm collision is still used to skip clearly impossible starts.
                backoff_candidates = list(
                    np.arange(0.0, args.max_backoff + 1e-9, args.backoff_step)
                )
                best_attempt = (
                    None  # keep the best (success first, then lowest track err)
                )
                attempts_meta = []
                for backoff in backoff_candidates:
                    ik_residual, init_ncon = place_target_at_backoff(
                        target_env,
                        ep_meta,
                        lowdim0,
                        backoff,
                        src_pos0,
                        src_xmat0,
                        args,
                        source_env=source_env,
                    )
                    if ik_residual > 2e-3 or init_ncon > 0:
                        attempts_meta.append(
                            {
                                "backoff_m": float(backoff),
                                "skipped": True,
                                "ik_residual_m": float(ik_residual),
                                "init_contacts": int(init_ncon),
                            }
                        )
                        continue

                    model_file = target_env.get_state()["model"]
                    if not args.no_pad_friction_boost:
                        boost_pad_friction(target_env)
                    # grasp-center alignment: constant site-frame offset so the
                    # target's fingertip pads land where the source's pads were,
                    # plus extra approach depth for the Robotiq closing swing
                    grasp_offset = (
                        grasp_center_src
                        - grasp_center_in_site_frame(target_env)
                        + np.array([0.0, 0.0, args.grasp_depth_extra])
                    )
                    # opening-axis (twist) alignment about the approach axis
                    if args.no_twist_align:
                        twist = np.eye(3)
                    else:
                        twist = gripper_twist_matrix(source_env, target_env)
                    target_states, target_actions, frames = [], [], []
                    track_errors = []
                    for step_idx, source_action in enumerate(source_actions):
                        desired_pos, desired_xmat = source_eef_pose_from_saved_state(
                            source_env, source_states, step_idx, source_action
                        )
                        desired_pos = desired_pos + desired_xmat @ grasp_offset
                        desired_xmat = desired_xmat @ twist
                        target_action = retarget_action(
                            source_action=source_action,
                            source_robot=source_env.env.robots[0],
                            target_env=target_env,
                            desired_pos=desired_pos,
                            desired_xmat=desired_xmat,
                            gain=args.eef_gain,
                            rot_gain=args.eef_rot_gain,
                        )
                        target_env.step(target_action)
                        tp, _ = robot_eef_pose(target_env)
                        track_errors.append(float(np.linalg.norm(desired_pos - tp)))
                        target_actions.append(target_action)
                        target_states.append(target_env.get_state()["states"].copy())
                        if not args.no_videos and (
                            step_idx % args.video_skip == 0
                            or step_idx == len(source_actions) - 1
                        ):
                            frames.append(
                                render_frame(
                                    target_env,
                                    camera_name=args.camera,
                                    height=args.camera_height,
                                    width=args.camera_width,
                                )
                            )
                    # restore source to initial state for a possible next attempt
                    source_env.reset_to(
                        {"states": np.asarray(source_states[0]).reshape(-1)}
                    )

                    success = bool(target_env.is_success().get("task", False))
                    attempt = {
                        "backoff": float(backoff),
                        "success": success,
                        "ik_residual": float(ik_residual),
                        "track_errors": track_errors,
                        "states": target_states,
                        "actions": target_actions,
                        "frames": frames,
                        "model_file": model_file,
                    }
                    attempts_meta.append(
                        {
                            "backoff_m": float(backoff),
                            "skipped": False,
                            "success": success,
                            "track_mean_m": float(np.mean(track_errors)),
                            "track_max_m": float(np.max(track_errors)),
                        }
                    )
                    if success:
                        best_attempt = attempt
                        break
                    if best_attempt is None or (
                        np.mean(track_errors) < np.mean(best_attempt["track_errors"])
                    ):
                        best_attempt = attempt

                if best_attempt is None:
                    stats.append(
                        {
                            "episode": episode_num,
                            "success": False,
                            "error": "no feasible backoff (all skipped)",
                            "attempts": attempts_meta,
                        }
                    )
                    print(f"[episode {episode_num:06d}] no feasible backoff, skipped")
                    continue

                success = best_attempt["success"]
                track_errors = best_attempt["track_errors"]
                stat = {
                    "episode": episode_num,
                    "success": success,
                    "backoff_m": best_attempt["backoff"],
                    "ik_residual_m": best_attempt["ik_residual"],
                    "track_mean_m": float(np.mean(track_errors)),
                    "track_max_m": float(np.max(track_errors)),
                    "track_final_m": float(track_errors[-1]),
                    "attempts": attempts_meta,
                }
                stats.append(stat)
                print(
                    f"[episode {episode_num:06d}] success={int(success)} "
                    f"backoff={best_attempt['backoff']:.2f}m "
                    f"attempts={len([a for a in attempts_meta if not a.get('skipped')])} "
                    f"track mean/max/final={stat['track_mean_m']*100:.1f}/"
                    f"{stat['track_max_m']*100:.1f}/{stat['track_final_m']*100:.1f}cm"
                )

                if not args.no_videos:
                    video_path = (
                        video_dir
                        / f"episode_{episode_num:06d}_success-{int(success)}.mp4"
                    )
                    write_episode_video(video_path, best_attempt["frames"])

                if success or args.keep_failed:
                    episodes.append(
                        {
                            "source_episode": episode_num,
                            "states": np.asarray(
                                best_attempt["states"], dtype=np.float64
                            ),
                            "actions": np.asarray(
                                best_attempt["actions"], dtype=np.float64
                            ),
                            "model_file": best_attempt["model_file"],
                            "ep_meta": ep_meta,
                            "success": success,
                            "initial_eef_pos_error": best_attempt["ik_residual"],
                            "initial_eef_rot_error": 0.0,
                        }
                    )
            except Exception as exc:
                stats.append(
                    {
                        "episode": episode_num,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[fail] episode {episode_num:06d}: {exc}")
                traceback.print_exc()
    finally:
        source_env.env.close()
        target_env.env.close()

    n_ok = sum(1 for s in stats if s.get("success"))
    summary = {
        "task": task_name,
        "target_robot": args.target_robot,
        "target_gripper": args.target_gripper,
        "episodes_attempted": len(stats),
        "episodes_succeeded": n_ok,
        "success_rate": (n_ok / len(stats)) if stats else 0.0,
        "episodes_saved": len(episodes),
        "per_episode": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if episodes:
        write_raw_hdf5(raw_path, episodes, target_env_meta)
        print(f"[raw] {raw_path}")
    else:
        print("[raw] no episodes saved (all failed and --keep-failed not set)")
    print(
        f"[summary] attempted={len(stats)} succeeded={n_ok} "
        f"rate={summary['success_rate']*100:.1f}% saved={len(episodes)}"
    )
    print(f"[summary-file] {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
