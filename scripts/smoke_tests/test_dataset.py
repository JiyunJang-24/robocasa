#!/usr/bin/env python3
"""
Retarget a RoboCasa dataset to UR5eOmron by following the source EEF trajectory.

This is different from simply replaying the source action vector. For every
source step, the script reads the next source EEF pose, computes the OSC delta
action that should move the target EEF toward that pose, executes that target
action, and saves the target states/actions as a raw robomimic-style HDF5 file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from dataset_reproduce import (
    DEFAULT_SOURCE,
    action_dict_from_vector,
    align_base_to_source_lowdim,
    build_target_env_meta,
    create_env,
    episode_paths,
    init_obs_utils,
    load_episode_payload,
    load_modality,
    read_episode_df,
    render_frame,
    reorder_lerobot_actions_to_hdf5,
    reset_source_to_episode,
    reset_target_to_episode,
    restore_gripper_from_lowdim,
    robot_eef_pose,
    solve_ik_to_eef_pose,
    write_episode_video,
    write_raw_hdf5,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lerobot-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-robot", default="UR5eOmron")
    parser.add_argument("--target-gripper", default="default")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ik-iters", type=int, default=250)
    parser.add_argument("--eef-gain", type=float, default=1.0)
    parser.add_argument(
        "--restore-gripper",
        action="store_true",
        help=(
            "Copy source lowdim gripper qpos into the target before rollout. "
            "This is off by default because Panda and Robotiq qpos conventions "
            "are different and direct copying can destabilize the simulation."
        ),
    )
    parser.add_argument(
        "--align-base",
        action="store_true",
        help=(
            "Also force the target mobile base from source lowdim state before IK. "
            "This is off by default because the lowdim base convention can differ "
            "from mobile base qpos and can move the target base incorrectly."
        ),
    )
    parser.add_argument("--drop-failed", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--camera-width", type=int, default=512)
    return parser.parse_args()


def sync_controller_goals(env):
    for robot in env.env.robots:
        robot.composite_controller.update_state()
        robot.composite_controller.reset()


def inverse_scale_action(controller, scaled_delta: np.ndarray) -> np.ndarray:
    if controller.action_scale is None:
        controller.scale_action(np.zeros(controller.control_dim))
    return (
        scaled_delta - controller.action_output_transform
    ) / controller.action_scale + controller.action_input_transform


def pos_in_controller_origin(env, pos: np.ndarray) -> np.ndarray:
    robot = env.env.robots[0]
    arm = robot.arms[0]
    controller = robot.part_controllers[arm]
    robot.composite_controller.update_state()
    if controller.input_ref_frame == "base":
        return controller.world_to_origin_frame(pos)
    if controller.input_ref_frame == "world":
        return pos
    raise ValueError(f"Unsupported input_ref_frame: {controller.input_ref_frame}")


def xmat_in_controller_origin(env, xmat: np.ndarray) -> np.ndarray:
    robot = env.env.robots[0]
    arm = robot.arms[0]
    controller = robot.part_controllers[arm]
    robot.composite_controller.update_state()
    if controller.input_ref_frame == "base":
        return controller.origin_ori.T @ xmat
    if controller.input_ref_frame == "world":
        return xmat
    raise ValueError(f"Unsupported input_ref_frame: {controller.input_ref_frame}")


def arm_action_to_eef_pose(
    env,
    desired_pos: np.ndarray,
    desired_xmat: np.ndarray,
    gain: float,
    rot_gain: float | None = None,
) -> np.ndarray:
    robot = env.env.robots[0]
    arm = robot.arms[0]
    controller = robot.part_controllers[arm]
    env.env.sim.forward()
    controller.update(force=True)

    current_pos, current_xmat = robot_eef_pose(env)
    pos_delta = pos_in_controller_origin(env, desired_pos) - pos_in_controller_origin(
        env, current_pos
    )

    desired_xmat_origin = xmat_in_controller_origin(env, desired_xmat)
    current_xmat_origin = xmat_in_controller_origin(env, current_xmat)
    rot_delta = desired_xmat_origin @ current_xmat_origin.T
    ori_delta = Rotation.from_matrix(rot_delta).as_rotvec()

    if rot_gain is None:
        rot_gain = gain
    scaled_delta = np.concatenate([gain * pos_delta, rot_gain * ori_delta])
    return np.clip(inverse_scale_action(controller, scaled_delta), -1.0, 1.0)


def retarget_action(
    source_action: np.ndarray,
    source_robot,
    target_env,
    desired_pos,
    desired_xmat,
    gain,
    rot_gain: float | None = None,
):
    action_dict = action_dict_from_vector(
        source_robot, np.asarray(source_action, dtype=np.float64)
    )
    target_arm = target_env.env.robots[0].arms[0]
    action_dict[target_arm] = arm_action_to_eef_pose(
        target_env, desired_pos, desired_xmat, gain, rot_gain=rot_gain
    )
    return target_env.env.robots[0].create_action_vector(action_dict)


def source_eef_pose_from_saved_state(
    source_env, source_states, step_idx, source_action
):
    next_idx = step_idx + 1
    if next_idx < len(source_states):
        source_env.reset_to({"states": np.asarray(source_states[next_idx]).reshape(-1)})
    else:
        source_env.step(source_action)
    return robot_eef_pose(source_env)


def default_output_dir(args, task_name: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    date_tag = dt.datetime.now().strftime("%Y%m%d")
    return (
        Path("/data1/local/robocasa/datasets/eef_retargeted")
        / args.target_robot
        / task_name
        / date_tag
    )


def make_dataset(args):
    init_obs_utils()
    with open(args.source_lerobot_dir / "extras" / "dataset_meta.json", "r") as f:
        dataset_meta = json.load(f)
    source_env_meta = dataset_meta["env_args"]
    target_env_meta = build_target_env_meta(
        source_env_meta, args.target_robot, args.target_gripper
    )
    modality = load_modality(args.source_lerobot_dir)

    output_dir = default_output_dir(args, source_env_meta["env_name"])
    raw_path = output_dir / "raw.hdf5"
    video_dir = output_dir / "videos"

    print(f"[source] {args.source_lerobot_dir}")
    print(f"[target] {args.target_robot}/{args.target_gripper}")
    print("[mode] source saved EEF trajectory -> target OSC delta action")
    print(f"[output] {output_dir}")

    source_env = create_env(source_env_meta, seed=args.seed)
    target_env = create_env(target_env_meta, seed=args.seed)
    episodes = []
    failures = []

    try:
        paths = episode_paths(
            args.source_lerobot_dir, args.start_episode, args.max_episodes
        )
        for parquet_path in tqdm(paths, desc="eef-retarget"):
            episode_num = int(parquet_path.stem.split("_")[-1])
            try:
                episode_df = read_episode_df(parquet_path)
                source_states, ep_meta, model_xml = load_episode_payload(
                    args.source_lerobot_dir, episode_num
                )
                source_actions = reorder_lerobot_actions_to_hdf5(
                    np.stack(episode_df["action"].to_numpy()),
                    modality=modality,
                )

                reset_source_to_episode(
                    source_env, source_states[0], ep_meta, model_xml
                )
                sync_controller_goals(source_env)
                source_eef_pos, source_eef_xmat = robot_eef_pose(source_env)

                reset_target_to_episode(target_env, ep_meta)
                source_lowdim0 = np.asarray(
                    episode_df.iloc[0]["observation.state"], dtype=np.float64
                )
                if args.align_base:
                    align_base_to_source_lowdim(target_env, source_lowdim0)
                solve_ik_to_eef_pose(
                    target_env, source_eef_pos, source_eef_xmat, n_iters=args.ik_iters
                )
                sync_controller_goals(target_env)
                if args.restore_gripper:
                    restore_gripper_from_lowdim(target_env, source_lowdim0)
                    sync_controller_goals(target_env)

                target_eef_pos, target_eef_xmat = robot_eef_pose(target_env)
                initial_pos_error = np.linalg.norm(source_eef_pos - target_eef_pos)
                initial_rot_error = np.linalg.norm(source_eef_xmat - target_eef_xmat)

                target_states = []
                target_actions = []
                frames = []
                eef_pos_errors = []
                eef_rot_errors = []
                model_file = target_env.get_state()["model"]

                for step_idx, source_action in enumerate(source_actions):
                    desired_pos, desired_xmat = source_eef_pose_from_saved_state(
                        source_env,
                        source_states,
                        step_idx,
                        source_action,
                    )
                    target_action = retarget_action(
                        source_action=source_action,
                        source_robot=source_env.env.robots[0],
                        target_env=target_env,
                        desired_pos=desired_pos,
                        desired_xmat=desired_xmat,
                        gain=args.eef_gain,
                    )
                    target_env.step(target_action)

                    target_pos, target_xmat = robot_eef_pose(target_env)
                    eef_pos_errors.append(np.linalg.norm(desired_pos - target_pos))
                    eef_rot_errors.append(np.linalg.norm(desired_xmat - target_xmat))
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

                success = bool(target_env.is_success().get("task", False))
                if (not success) and args.drop_failed:
                    print(
                        f"[skip] episode {episode_num:06d}: target rollout did not succeed"
                    )
                    failures.append((episode_num, "not_success"))
                    continue

                if not args.no_videos:
                    video_path = (
                        video_dir
                        / f"episode_{episode_num:06d}_success-{int(success)}.mp4"
                    )
                    write_episode_video(video_path, frames)
                    print(f"[video] {video_path}")

                episodes.append(
                    {
                        "source_episode": episode_num,
                        "states": np.asarray(target_states, dtype=np.float64),
                        "actions": np.asarray(target_actions, dtype=np.float64),
                        "model_file": model_file,
                        "ep_meta": ep_meta,
                        "success": success,
                        "initial_eef_pos_error": initial_pos_error,
                        "initial_eef_rot_error": initial_rot_error,
                    }
                )
                print(
                    f"[episode {episode_num:06d}] success={int(success)} "
                    f"init_pos={initial_pos_error:.5f} "
                    f"final_pos={eef_pos_errors[-1]:.5f} "
                    f"max_pos={max(eef_pos_errors):.5f} "
                    f"final_rot={eef_rot_errors[-1]:.5f} "
                    f"max_rot={max(eef_rot_errors):.5f}"
                )
            except Exception as exc:
                failures.append((episode_num, f"{type(exc).__name__}: {exc}"))
                print(f"[fail] episode {episode_num:06d}: {exc}")
                traceback.print_exc()
    finally:
        source_env.env.close()
        target_env.env.close()

    if not episodes:
        raise RuntimeError(f"No episodes retargeted. Failures: {failures[:5]}")

    write_raw_hdf5(raw_path, episodes, target_env_meta)
    print(f"[raw] {raw_path}")
    print(f"[summary] episodes={len(episodes)} failures={len(failures)}")
    if failures:
        print(f"[summary] first failures={failures[:5]}")
    return raw_path


def main():
    make_dataset(parse_args())


if __name__ == "__main__":
    main()
