#!/usr/bin/env python3
"""
Replay existing atomic LeRobot datasets into a target robot/gripper configuration.

This script reads the source atomic dataset stored under:
    datasets/v1.0/<split>/atomic/<task>/<date>/lerobot

It then:
  1. rebuilds the task environment with the requested robot / gripper;
  2. replays the recorded actions in that environment;
  3. saves a raw robomimic-style HDF5 dataset;
  4. optionally converts the raw HDF5 into a LeRobot dataset;
  5. writes mp4 videos for visual inspection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import shutil
import traceback
from copy import deepcopy
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
from tqdm import tqdm

import robocasa
from robocasa.utils import env_utils as CoreEnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite import __version__ as robosuite_version

from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
from robocasa.utils.dataset_registry_utils import get_ds_path


ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def _read_episode_df(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    return pd.read_parquet(parquet_path)


def _load_episode_payload(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    state_path = episode_dir / "states.npz"
    meta_path = episode_dir / "ep_meta.json"
    model_path = episode_dir / "model.xml.gz"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    states = np.load(state_path, allow_pickle=True)["states"]
    ep_meta = json.load(open(meta_path, "r"))
    payload = {}
    if model_path.exists():
        with gzip.open(model_path, "rt") as f:
            payload["model"] = f.read()
    return states, ep_meta, payload


def _resolve_episode_paths(source_lerobot_dir: Path) -> list[Path]:
    episode_paths = sorted(source_lerobot_dir.glob("data/chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(
            f"No episode parquet files found in {source_lerobot_dir}"
        )
    return episode_paths


def _resolve_source_lerobot_dir(
    source_root: Path, task_name: str, split: str
) -> Path | None:
    candidates = sorted(
        (source_root / split / "atomic" / task_name).glob("*/lerobot"),
        reverse=True,
    )
    if candidates:
        return candidates[0]

    registry_path = get_ds_path(task=task_name, source="human", split=split)
    if registry_path is None:
        return None
    return Path(registry_path)


def _load_modality_dict(source_lerobot_dir: Path) -> dict:
    modality_path = source_lerobot_dir / "meta" / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(modality_path)
    with open(modality_path, "r") as f:
        return json.load(f)


def _reorder_lerobot_actions_to_raw(
    actions: np.ndarray, modality_dict: dict
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    action_info = modality_dict["action"]
    raw_dim = max(end for _, end in ACTION_KEY_ORDERING_HDF5.values())
    raw_actions = np.zeros((actions.shape[0], raw_dim), dtype=np.float64)

    for key, hdf5_slice in ACTION_KEY_ORDERING_HDF5.items():
        if key not in action_info:
            continue
        lerobot_start = action_info[key]["start"]
        lerobot_end = action_info[key]["end"]
        hdf5_start, hdf5_end = hdf5_slice
        raw_actions[:, hdf5_start:hdf5_end] = actions[:, lerobot_start:lerobot_end]

    return raw_actions


def _build_target_env_meta(
    source_env_meta: dict, robot: str, gripper_type: str, controller: str | None
) -> dict:
    env_meta = deepcopy(source_env_meta)
    env_kwargs = deepcopy(env_meta["env_kwargs"])
    env_kwargs["robots"] = robot
    normalized_gripper = {
        "Robotiq85": "Robotiq85Gripper",
        "Robotiq140": "Robotiq140Gripper",
    }.get(gripper_type, gripper_type)
    env_kwargs["gripper_types"] = normalized_gripper
    env_kwargs["translucent_robot"] = False

    if robot == "UR5eOmron":
        # Keep the source mobile-base controller semantics. UR5e in this path is treated as a
        # mobile robot mounted on Omron, not as the fixed-base default UR5e mount.
        env_kwargs["controller_configs"] = deepcopy(
            source_env_meta["env_kwargs"]["controller_configs"]
        )
    else:
        # If no controller was explicitly requested, prefer the target robot's default controller.
        source_controller = controller
        try:
            env_kwargs["controller_configs"] = load_composite_controller_config(
                controller=source_controller,
                robot=robot,
            )
        except Exception as exc:
            print(
                f"[warn] could not rebuild controller config for robot={robot} "
                f"controller={source_controller}: {exc}. Falling back to target robot default."
            )
            env_kwargs["controller_configs"] = load_composite_controller_config(
                robot=robot
            )

    env_meta["env_kwargs"] = env_kwargs
    return env_meta


def _vector_to_action_dict(robot, action: np.ndarray) -> dict:
    controller = getattr(robot, "composite_controller", None)
    if controller is not None and hasattr(
        controller, "create_action_dict_from_action_vector"
    ):
        return controller.create_action_dict_from_action_vector(action)

    action_dict = {}
    for part_name, (start_idx, end_idx) in robot._action_split_indexes.items():
        action_dict[part_name] = action[start_idx:end_idx]
    return action_dict


def _action_adapter(
    raw_action: np.ndarray, source_robot, target_robot, target_dim: int
) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    source_dim = raw_action.shape[0]

    try:
        action_dict = _vector_to_action_dict(source_robot, raw_action)
        return target_robot.create_action_vector(action_dict)
    except Exception as exc:
        print(f"[warn] robosuite cross-embodiment action adapter failed: {exc}")
        if source_dim == target_dim:
            return raw_action
        if source_dim == 12 and target_dim == 7:
            return raw_action[0:7]

    raise ValueError(
        f"Unsupported action mapping: source_dim={source_dim}, target_dim={target_dim}. "
        "Use a compatible robot/controller pair or extend the adapter."
    )


def _safe_reset_to_source_state(
    env,
    source_state: np.ndarray,
    source_ep_meta: dict,
    source_model: str | None = None,
) -> bool:
    """
    Best-effort state restore.
    We first try to restore the exact simulator state when the flattened sizes match.
    If that fails, we fall back to a plain env reset with ep_meta already installed.
    """
    try:
        if source_model is not None:
            env.reset_to(
                {
                    "states": source_state,
                    "model": source_model,
                    "ep_meta": json.dumps(source_ep_meta),
                }
            )
            return True

        env.reset()
        current_shape = env.get_state()["states"].shape[0]
        if source_state.shape[0] == current_shape:
            env.reset_to({"states": source_state})
            return True
    except Exception as exc:
        print(f"[warn] exact state restore failed: {exc}")

    try:
        env.reset()
    except Exception as exc:
        print(f"[warn] env.reset() failed after state restore failure: {exc}")
        raise
    return False


def _render_frame(env, camera_names: list[str], height: int, width: int) -> np.ndarray:
    frames = []
    for camera_name in camera_names:
        try:
            frame = env.render(
                mode="rgb_array",
                height=height,
                width=width,
                camera_name=camera_name,
            )
            frames.append(frame)
        except Exception as exc:
            print(f"[warn] skipping unavailable camera {camera_name}: {exc}")
    if len(frames) == 1:
        return frames[0]
    return np.concatenate(frames, axis=1)


def _is_wrist_camera(camera_name: str) -> bool:
    return "eye_in_hand" in camera_name or "wrist" in camera_name


def _capture_camera_world_poses(
    env, camera_names: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sim = env.env.sim
    sim.forward()
    poses = {}
    for camera_name in camera_names:
        if _is_wrist_camera(camera_name):
            continue
        try:
            cam_id = sim.model.camera_name2id(camera_name)
        except Exception:
            continue
        poses[camera_name] = (
            sim.data.cam_xpos[cam_id].copy(),
            sim.data.cam_xmat[cam_id].reshape(3, 3).copy(),
        )
    return poses


def _apply_world_fixed_camera_poses(
    env, camera_poses: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    sim = env.env.sim
    sim.forward()
    for camera_name, (world_pos, world_xmat) in camera_poses.items():
        try:
            cam_id = sim.model.camera_name2id(camera_name)
        except Exception:
            continue
        body_id = sim.model.cam_bodyid[cam_id]
        body_pos = sim.data.xpos[body_id].copy()
        body_xmat = sim.data.xmat[body_id].reshape(3, 3).copy()
        local_pos = body_xmat.T @ (world_pos - body_pos)
        local_xmat = body_xmat.T @ world_xmat
        sim.model.cam_pos[cam_id] = local_pos
        sim.model.cam_quat[cam_id] = T.convert_quat(T.mat2quat(local_xmat), to="wxyz")
    sim.forward()


def _resize_frame_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx][:, x_idx]


def _resolve_source_video_path(
    source_lerobot_dir: Path,
    episode_num: int,
    source_camera: str,
) -> Path | None:
    rel = (
        "videos"
        f"/chunk-{episode_num // 1000:03d}"
        f"/observation.images.{source_camera}"
        f"/episode_{episode_num:06d}.mp4"
    )
    video_path = source_lerobot_dir / rel
    if video_path.exists():
        return video_path

    matches = sorted(
        source_lerobot_dir.glob(
            f"videos/chunk-*/observation.images.*/episode_{episode_num:06d}.mp4"
        )
    )
    return matches[0] if matches else None


def _read_sampled_video_frames(
    video_path: Path,
    frame_indices: list[int],
    height: int,
    width: int,
) -> list[np.ndarray]:
    wanted = set(frame_indices)
    frames_by_idx = {}
    reader = imageio.get_reader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if idx in wanted:
                frames_by_idx[idx] = _resize_frame_nearest(
                    np.asarray(frame),
                    height=height,
                    width=width,
                )
            if len(frames_by_idx) == len(wanted):
                break
    finally:
        reader.close()

    if not frames_by_idx:
        raise ValueError(f"No frames read from source video {video_path}")

    last_frame = None
    sampled_frames = []
    for idx in frame_indices:
        if idx in frames_by_idx:
            last_frame = frames_by_idx[idx]
        sampled_frames.append(last_frame)

    first_frame = next(frame for frame in sampled_frames if frame is not None)
    return [frame if frame is not None else first_frame for frame in sampled_frames]


def _write_side_by_side_video(
    source_lerobot_dir: Path,
    episode_num: int,
    source_cameras: list[str],
    target_frames: list[np.ndarray],
    target_frame_indices: list[int],
    video_path: Path,
    fps: int = 20,
) -> bool:
    if not target_frames:
        return False

    target_height, target_width = target_frames[0].shape[:2]
    source_frame_groups = []
    for source_camera in source_cameras:
        source_video_path = _resolve_source_video_path(
            source_lerobot_dir=source_lerobot_dir,
            episode_num=episode_num,
            source_camera=source_camera,
        )
        if source_video_path is None:
            print(
                f"[warn] source video not found for camera={source_camera} episode={episode_num}"
            )
            continue

        source_frame_groups.append(
            _read_sampled_video_frames(
                video_path=source_video_path,
                frame_indices=target_frame_indices,
                height=target_height,
                width=target_width,
            )
        )

    if not source_frame_groups:
        print(f"[warn] no source videos found for episode={episode_num}")
        return False

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        separator = np.zeros((target_height, 16, 3), dtype=np.uint8)
        for frame_idx, target_frame in enumerate(target_frames):
            source_frames = [
                group[frame_idx].astype(np.uint8) for group in source_frame_groups
            ]
            writer.append_data(
                np.concatenate(
                    [
                        *source_frames,
                        separator,
                        target_frame.astype(np.uint8),
                    ],
                    axis=1,
                )
            )
    finally:
        writer.close()

    print(f"[comparison] {video_path}")
    return True


def _flatten_index_container(value) -> np.ndarray:
    if isinstance(value, dict):
        parts = [np.asarray(v, dtype=np.int64).reshape(-1) for v in value.values()]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(parts).astype(np.int64)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _solve_ik_to_target_pose(
    env, target_pos: np.ndarray, target_xmat: np.ndarray, n_iters: int = 200
) -> None:
    sim = env.env.sim
    robot = env.env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    jpos_idx = _flatten_index_container(robot._ref_joint_pos_indexes)
    jvel_idx = _flatten_index_container(robot._ref_joint_vel_indexes)

    for _ in range(n_iters):
        sim.forward()
        cur_pos = sim.data.site_xpos[site_id].copy()
        cur_xmat = sim.data.site_xmat[site_id].reshape(3, 3).copy()
        pos_err = target_pos - cur_pos
        rot_err = 0.5 * (
            np.cross(cur_xmat[:, 0], target_xmat[:, 0])
            + np.cross(cur_xmat[:, 1], target_xmat[:, 1])
            + np.cross(cur_xmat[:, 2], target_xmat[:, 2])
        )
        if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 5e-3:
            break

        jacp = np.zeros((3, sim.model.nv), dtype=np.float64)
        jacr = np.zeros((3, sim.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            sim.model._model if hasattr(sim.model, "_model") else sim.model,
            sim.data._data if hasattr(sim.data, "_data") else sim.data,
            jacp,
            jacr,
            int(site_id),
        )

        J = np.vstack((jacp[:, jvel_idx], jacr[:, jvel_idx]))
        err = np.concatenate((pos_err, rot_err))
        A = J @ J.T + (1e-2**2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(A, err)
        q = sim.data.qpos[jpos_idx].copy()
        sim.data.qpos[jpos_idx] = q + 0.5 * dq
    sim.forward()


def _apply_lowdim_initial_state(
    env, source_lowdim: np.ndarray, allow_mobile_base: bool = True
) -> None:
    source_lowdim = np.asarray(source_lowdim, dtype=np.float64).reshape(-1)
    if source_lowdim.shape[0] < 16:
        raise ValueError(
            f"Expected at least 16 low-dim state values, got {source_lowdim.shape[0]}"
        )

    base_pos = source_lowdim[0:3]
    base_quat = source_lowdim[3:7]
    rel_eef_pos = source_lowdim[7:10]
    rel_eef_quat = source_lowdim[10:14]
    gripper_qpos = source_lowdim[14:16]

    base_R = T.quat2mat(base_quat)
    target_pos = base_pos + base_R @ rel_eef_pos
    target_xmat = base_R @ T.quat2mat(rel_eef_quat)

    # Attempt to place the mobile base only when the target robot actually has one.
    if allow_mobile_base:
        try:
            env.init_robot_base_pos_anchor = base_pos
            env.init_robot_base_ori_anchor = T.mat2euler(base_R)
            CoreEnvUtils.set_robot_to_position(env.env, base_pos)
        except Exception as exc:
            print(f"[warn] base placement helper failed: {exc}")

    # Then use IK to place the eef.
    try:
        _solve_ik_to_target_pose(env, target_pos=target_pos, target_xmat=target_xmat)
    except Exception as exc:
        print(f"[warn] IK initialization failed: {exc}")

    # Best-effort gripper restore.
    try:
        robot = env.env.robots[0]
        gripper_idx = _flatten_index_container(
            getattr(robot, "_ref_gripper_joint_pos_indexes", [])
        )
        if gripper_idx.size > 0:
            values = np.zeros(gripper_idx.shape[0], dtype=np.float64)
            count = min(values.shape[0], gripper_qpos.shape[0])
            values[:count] = gripper_qpos[:count]
            env.env.sim.data.qpos[gripper_idx] = values
    except Exception as exc:
        print(f"[warn] gripper restore failed: {exc}")

    # Freeze any residual velocity so the initial no-op state stays still.
    try:
        sim = env.env.sim
        sim.data.qvel[:] = 0.0
        if hasattr(sim.data, "qacc"):
            sim.data.qacc[:] = 0.0
        if hasattr(sim.data, "ctrl"):
            sim.data.ctrl[:] = 0.0
        sim.forward()
    except Exception as exc:
        print(f"[warn] zero-velocity freeze failed: {exc}")


def _hold_static_state(env) -> None:
    """
    Keep the simulator in a no-op configuration without integrating dynamics.
    This is used for the initial settle phase so a zero action cannot drift the pose.
    """
    try:
        sim = env.env.sim
        sim.data.qvel[:] = 0.0
        if hasattr(sim.data, "qacc"):
            sim.data.qacc[:] = 0.0
        if hasattr(sim.data, "ctrl"):
            sim.data.ctrl[:] = 0.0
        sim.forward()
    except Exception as exc:
        print(f"[warn] static hold failed: {exc}")


def _write_raw_hdf5(raw_path: Path, episodes: list[dict], env_meta: dict) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "w") as f:
        data_grp = f.create_group("data")
        now = dt.datetime.now()
        data_grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        data_grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        data_grp.attrs["robocasa_version"] = robocasa.__version__
        data_grp.attrs["robosuite_version"] = robosuite_version
        data_grp.attrs["mujoco_version"] = mujoco.__version__
        data_grp.attrs["env"] = env_meta["env_name"]
        data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
        data_grp.attrs["total"] = int(sum(ep["actions"].shape[0] for ep in episodes))

        for idx, episode in enumerate(episodes):
            demo_grp = data_grp.create_group(f"demo_{idx}")
            demo_grp.attrs["model_file"] = episode["model_file"]
            demo_grp.attrs["ep_meta"] = json.dumps(episode["ep_meta"], indent=4)
            demo_grp.attrs["num_samples"] = int(episode["actions"].shape[0])
            demo_grp.create_dataset(
                "states", data=np.asarray(episode["states"], dtype=np.float64)
            )
            demo_grp.create_dataset(
                "actions", data=np.asarray(episode["actions"], dtype=np.float64)
            )


def replay_task(
    task_name: str,
    source_lerobot_dir: Path,
    output_task_dir: Path,
    robot: str,
    gripper_type: str,
    controller: str | None,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    settle_steps: int,
    max_episodes: int | None,
    save_videos: bool,
    video_skip: int,
    save_comparison_videos: bool,
    comparison_source_cameras: list[str],
    retarget_initial_eef: bool,
    align_scene_cameras: bool,
):
    dataset_meta_path = source_lerobot_dir / "extras" / "dataset_meta.json"
    with open(dataset_meta_path, "r") as f:
        dataset_meta = json.load(f)

    modality_dict = _load_modality_dict(source_lerobot_dir)
    source_env_meta = dataset_meta["env_args"]
    target_env_meta = _build_target_env_meta(
        source_env_meta,
        robot=robot,
        gripper_type=gripper_type,
        controller=controller,
    )

    episode_paths = _resolve_episode_paths(source_lerobot_dir)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    output_task_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_task_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    comparison_video_dir = output_task_dir / "comparison_videos"
    comparison_video_dir.mkdir(parents=True, exist_ok=True)
    raw_hdf5_path = output_task_dir / "raw.hdf5"

    successful_episodes = []
    replayed_episodes = []

    env = EnvUtils.create_env_from_metadata(
        env_meta=target_env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
        seed=0,
    )
    source_action_env = EnvUtils.create_env_from_metadata(
        env_meta=source_env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
        seed=0,
    )
    source_env = None
    if align_scene_cameras:
        source_env = EnvUtils.create_env_from_metadata(
            env_meta=source_env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=False,
            seed=0,
        )

    try:
        for ep_idx, parquet_path in enumerate(
            tqdm(episode_paths, desc=f"Replaying {task_name}")
        ):
            episode_df = _read_episode_df(parquet_path)
            episode_num = int(parquet_path.stem.split("_")[-1])
            episode_dir = source_lerobot_dir / "extras" / f"episode_{episode_num:06d}"
            source_states, source_ep_meta, source_payload = _load_episode_payload(
                episode_dir
            )
            source_actions = _reorder_lerobot_actions_to_raw(
                np.stack(episode_df["action"].to_numpy()),
                modality_dict=modality_dict,
            )
            target_actions = []
            episode_states = []
            video_frames = []
            video_frame_indices = []

            # Install episode metadata first so object/layout/state sampling match the source episode.
            if hasattr(env.env, "set_attrs_from_ep_meta"):
                env.env.set_attrs_from_ep_meta(source_ep_meta)
            elif hasattr(env.env, "set_ep_meta"):
                env.env.set_ep_meta(source_ep_meta)

            if hasattr(source_action_env.env, "set_attrs_from_ep_meta"):
                source_action_env.env.set_attrs_from_ep_meta(source_ep_meta)
            elif hasattr(source_action_env.env, "set_ep_meta"):
                source_action_env.env.set_ep_meta(source_ep_meta)
            _safe_reset_to_source_state(
                source_action_env,
                np.asarray(source_states[0]).reshape(-1),
                source_ep_meta,
            )

            source_camera_poses = {}
            if source_env is not None:
                if hasattr(source_env.env, "set_attrs_from_ep_meta"):
                    source_env.env.set_attrs_from_ep_meta(source_ep_meta)
                elif hasattr(source_env.env, "set_ep_meta"):
                    source_env.env.set_ep_meta(source_ep_meta)
                _safe_reset_to_source_state(
                    source_env,
                    np.asarray(source_states[0]).reshape(-1),
                    source_ep_meta,
                    source_model=source_payload.get("model"),
                )
                source_camera_poses = _capture_camera_world_poses(
                    source_env,
                    camera_names=camera_names,
                )

            _safe_reset_to_source_state(
                env, np.asarray(source_states[0]).reshape(-1), source_ep_meta
            )
            if retarget_initial_eef:
                source_lowdim = np.asarray(
                    episode_df.iloc[0]["observation.state"],
                    dtype=np.float64,
                ).reshape(-1)
                _apply_lowdim_initial_state(
                    env,
                    source_lowdim,
                    allow_mobile_base=False,
                )
            if source_camera_poses:
                _apply_world_fixed_camera_poses(env, source_camera_poses)
            _hold_static_state(env)
            model_file = env.get_state()["model"]

            episode_success = False
            try:
                for step_idx, source_action in enumerate(source_actions):
                    adapted_action = _action_adapter(
                        source_action,
                        source_robot=source_action_env.env.robots[0],
                        target_robot=env.env.robots[0],
                        target_dim=env.action_dimension,
                    )
                    target_actions.append(adapted_action)
                    if step_idx < settle_steps and np.allclose(adapted_action, 0.0):
                        _hold_static_state(env)
                    else:
                        env.step(adapted_action)

                    episode_states.append(env.get_state()["states"].copy())
                    if save_videos:
                        if source_camera_poses:
                            _apply_world_fixed_camera_poses(env, source_camera_poses)
                        frame = _render_frame(
                            env,
                            camera_names=camera_names,
                            height=camera_height,
                            width=camera_width,
                        )
                        if (
                            step_idx % video_skip == 0
                            or step_idx == len(source_actions) - 1
                        ):
                            video_frames.append(frame)
                            video_frame_indices.append(step_idx)

                episode_success = env.is_success().get("task", False)
            except Exception as exc:
                print(f"[warn] replay failed for {parquet_path.name}: {exc}")
                traceback.print_exc()
                continue

            if save_videos and video_frames:
                video_path = (
                    video_dir
                    / f"episode_{episode_num:06d}_success-{int(episode_success)}.mp4"
                )
                writer = imageio.get_writer(video_path, fps=20)
                for frame in video_frames:
                    writer.append_data(frame)
                writer.close()
                print(f"[video] {video_path}")

                if save_comparison_videos:
                    comparison_path = (
                        comparison_video_dir
                        / f"episode_{episode_num:06d}_success-{int(episode_success)}.mp4"
                    )
                    try:
                        _write_side_by_side_video(
                            source_lerobot_dir=source_lerobot_dir,
                            episode_num=episode_num,
                            source_cameras=comparison_source_cameras,
                            target_frames=video_frames,
                            target_frame_indices=video_frame_indices,
                            video_path=comparison_path,
                        )
                    except Exception as exc:
                        print(
                            f"[warn] comparison video failed for episode={episode_num}: {exc}"
                        )

            if not episode_success:
                print(
                    f"[skip] task={task_name} episode={episode_num} did not succeed in target env"
                )
                continue

            replayed_episodes.append(
                {
                    "states": np.asarray(episode_states, dtype=np.float64),
                    "actions": np.asarray(target_actions, dtype=np.float64),
                    "ep_meta": source_ep_meta,
                    "model_file": model_file,
                }
            )
            successful_episodes.append(episode_num)

    finally:
        if (
            source_action_env is not None
            and hasattr(source_action_env, "env")
            and hasattr(source_action_env.env, "close")
        ):
            source_action_env.env.close()
        if (
            source_env is not None
            and hasattr(source_env, "env")
            and hasattr(source_env.env, "close")
        ):
            source_env.env.close()
        if hasattr(env, "env") and hasattr(env.env, "close"):
            env.env.close()

    if not replayed_episodes:
        print(f"[warn] no successful episodes produced for task={task_name}")
        return None

    _write_raw_hdf5(raw_hdf5_path, replayed_episodes, target_env_meta)
    print(f"[raw] {raw_hdf5_path}")

    return raw_hdf5_path


def maybe_convert_to_lerobot(raw_hdf5_path: Path, robot: str) -> Path | None:
    if robot != "PandaOmron":
        print(
            f"[skip] LeRobot conversion is PandaOmron-specific in this repo; leaving {raw_hdf5_path} as raw HDF5"
        )
        return None

    try:
        from robocasa.scripts.dataset_scripts.convert_hdf5_lerobot import (
            main as convert_main,
        )
    except Exception as exc:
        print(f"[warn] LeRobot converter import failed: {exc}")
        return None

    try:

        class Args:
            pass

        args = Args()
        args.raw_dataset_path = str(raw_hdf5_path)
        args.camera_names = [
            "robot0_eye_in_hand",
            "robot0_agentview_left",
            "robot0_agentview_right",
        ]
        args.camera_height = 256
        args.camera_width = 256
        convert_main(args)
        return raw_hdf5_path.parent / "lerobot"
    except Exception as exc:
        print(f"[warn] LeRobot conversion failed for {raw_hdf5_path}: {exc}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/data1/local/robocasa/datasets/v1.0")
    parser.add_argument(
        "--output-root", default="/data1/local/robocasa/datasets/generated"
    )
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--robot", default="PandaOmron")
    parser.add_argument("--gripper-type", default="default")
    parser.add_argument("--controller", default=None)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=["robot0_eye_in_hand", "robot0_agentview_left"],
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--no-comparison-videos", action="store_true")
    parser.add_argument("--no-retarget-initial-eef", action="store_true")
    parser.add_argument("--no-align-scene-cameras", action="store_true")
    parser.add_argument(
        "--comparison-source-cameras",
        nargs="+",
        default=["robot0_eye_in_hand", "robot0_agentview_left"],
    )
    args = parser.parse_args()

    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        )
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    tasks = [args.task_name] if args.task_name else list(ATOMIC_TASK_DATASETS.keys())
    date_tag = dt.datetime.now().strftime("%Y%m%d")
    output_root = Path(args.output_root)
    source_root = Path(args.source_root)

    for task_name in tasks:
        source_lerobot_dir = _resolve_source_lerobot_dir(
            source_root=source_root,
            task_name=task_name,
            split=args.split,
        )
        if source_lerobot_dir is None:
            print(f"[skip] no source dataset registered for task={task_name}")
            continue

        output_task_dir = (
            output_root
            / args.robot
            / args.gripper_type
            / args.split
            / task_name
            / date_tag
        )

        print(f"[task] {task_name}")
        print(f"[source] {source_lerobot_dir}")
        print(f"[output] {output_task_dir}")

        raw_hdf5_path = replay_task(
            task_name=task_name,
            source_lerobot_dir=source_lerobot_dir,
            output_task_dir=output_task_dir,
            robot=args.robot,
            gripper_type=args.gripper_type,
            controller=args.controller,
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            settle_steps=args.settle_steps,
            max_episodes=args.max_episodes,
            save_videos=not args.no_videos,
            video_skip=args.video_skip,
            save_comparison_videos=not args.no_comparison_videos,
            comparison_source_cameras=args.comparison_source_cameras,
            retarget_initial_eef=not args.no_retarget_initial_eef,
            align_scene_cameras=not args.no_align_scene_cameras,
        )

        if raw_hdf5_path is None:
            continue

        maybe_convert_to_lerobot(raw_hdf5_path, robot=args.robot)


if __name__ == "__main__":
    main()
