#!/usr/bin/env python3
"""
Reproduce a RoboCasa LeRobot dataset on a different Omron-mounted robot.

Default use case:
  - source dataset: AdjustToasterOvenTemperature / pretrain / 20250820
  - source robot: PandaOmron
  - target robot: UR5eOmron

For each source episode, this script:
  1. restores the source episode initial state;
  2. reads the source robot EEF world pose;
  3. resets a target robot env with the same episode metadata;
  4. uses damped-least-squares IK to match the target EEF to the source EEF;
  5. maps each source action through robosuite's action dict API;
  6. steps the target env and writes a raw robomimic-style HDF5 dataset.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
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
import robosuite
import robosuite.utils.transform_utils as T


DEFAULT_SOURCE = Path(
    "/data1/local/robocasa/datasets/v1.0/pretrain/atomic/"
    "AdjustToasterOvenTemperature/20250820/lerobot"
)

ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lerobot-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to datasets/reproduced/<robot>/<task>/<date>.",
    )
    parser.add_argument("--target-robot", default="UR5eOmron")
    parser.add_argument("--target-gripper", default="default")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--camera-width", type=int, default=512)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--drop-failed",
        action="store_true",
        help="Drop target rollouts that do not satisfy the task success check.",
    )
    parser.add_argument("--no-align-base", action="store_true")
    parser.add_argument("--ik-iters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def init_obs_utils():
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )


def read_episode_df(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    return pd.read_parquet(parquet_path)


def episode_paths(
    source_lerobot_dir: Path, start_episode: int, max_episodes: int | None
):
    paths = sorted(source_lerobot_dir.glob("data/chunk-*/episode_*.parquet"))
    paths = [p for p in paths if int(p.stem.split("_")[-1]) >= start_episode]
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(
            f"No episode parquet files found in {source_lerobot_dir}"
        )
    return paths


def load_episode_payload(source_lerobot_dir: Path, episode_num: int):
    episode_dir = source_lerobot_dir / "extras" / f"episode_{episode_num:06d}"
    state_path = episode_dir / "states.npz"
    meta_path = episode_dir / "ep_meta.json"
    model_path = episode_dir / "model.xml.gz"

    states = np.load(state_path, allow_pickle=True)["states"]
    ep_meta = json.load(open(meta_path, "r"))
    model_xml = None
    if model_path.exists():
        with gzip.open(model_path, "rt") as f:
            model_xml = f.read()
    return states, ep_meta, model_xml


def load_modality(source_lerobot_dir: Path):
    with open(source_lerobot_dir / "meta" / "modality.json", "r") as f:
        return json.load(f)


def reorder_lerobot_actions_to_hdf5(actions: np.ndarray, modality: dict) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    raw_dim = max(end for _, end in ACTION_KEY_ORDERING_HDF5.values())
    raw_actions = np.zeros((actions.shape[0], raw_dim), dtype=np.float64)

    for key, (raw_start, raw_end) in ACTION_KEY_ORDERING_HDF5.items():
        if key not in modality["action"]:
            continue
        src_start = modality["action"][key]["start"]
        src_end = modality["action"][key]["end"]
        raw_actions[:, raw_start:raw_end] = actions[:, src_start:src_end]
    return raw_actions


def build_target_env_meta(
    source_env_meta: dict, target_robot: str, target_gripper: str
) -> dict:
    env_meta = deepcopy(source_env_meta)
    env_kwargs = deepcopy(env_meta["env_kwargs"])
    env_kwargs["robots"] = target_robot
    env_kwargs["gripper_types"] = target_gripper
    env_kwargs["translucent_robot"] = False

    # Preserve the source PandaOmron HYBRID_MOBILE_BASE controller shape so actions
    # remain semantically aligned: right arm, right gripper, base, torso.
    controller_configs = deepcopy(source_env_meta["env_kwargs"]["controller_configs"])
    if "composite_controller_specific_configs" in controller_configs:
        controller_configs["composite_controller_specific_configs"][
            "body_part_ordering"
        ] = [
            "right",
            "right_gripper",
            "base",
            "torso",
        ]
    env_kwargs["controller_configs"] = controller_configs
    env_meta["env_kwargs"] = env_kwargs
    return env_meta


def create_env(env_meta: dict, seed: int):
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
        seed=seed,
    )


def install_ep_meta(env, ep_meta: dict):
    if hasattr(env.env, "set_attrs_from_ep_meta"):
        env.env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env.env, "set_ep_meta"):
        env.env.set_ep_meta(ep_meta)


def reset_source_to_episode(
    env, state: np.ndarray, ep_meta: dict, model_xml: str | None
):
    reset_state = {"states": np.asarray(state).reshape(-1)}
    if model_xml is not None:
        reset_state["model"] = model_xml
        reset_state["ep_meta"] = json.dumps(ep_meta)
    return env.reset_to(reset_state)


def reset_target_to_episode(env, ep_meta: dict):
    install_ep_meta(env, ep_meta)
    return env.reset(unset_ep_meta=False)


def flatten_index_container(value) -> np.ndarray:
    if isinstance(value, dict):
        parts = [np.asarray(v, dtype=np.int64).reshape(-1) for v in value.values()]
        return (
            np.concatenate(parts).astype(np.int64)
            if parts
            else np.zeros(0, dtype=np.int64)
        )
    return np.asarray(value, dtype=np.int64).reshape(-1)


def robot_eef_pose(env):
    sim = env.env.sim
    robot = env.env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    sim.forward()
    return (
        sim.data.site_xpos[int(site_id)].copy(),
        sim.data.site_xmat[int(site_id)].reshape(3, 3).copy(),
    )


def solve_ik_to_eef_pose(
    env,
    target_pos: np.ndarray,
    target_xmat: np.ndarray,
    n_iters: int,
    nullspace_gain: float = 0.05,
):
    """Damped-least-squares IK with a nullspace bias toward the current (reset) pose.

    Without the nullspace term the DLS iterations tend to wander into contorted
    elbow-flipped solutions; from those configurations the downstream OSC
    controller stalls against the counter and cannot track the source EEF.
    The bias projects a pull toward the natural reset configuration into the
    task-space nullspace, so the EEF pose is still matched exactly while the
    arm stays close to its natural configuration.
    """
    sim = env.env.sim
    robot = env.env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]

    qpos_idx = flatten_index_container(robot._ref_joint_pos_indexes)
    qvel_idx = flatten_index_container(robot._ref_joint_vel_indexes)
    if qpos_idx.size == 0 or qvel_idx.size == 0:
        raise ValueError("Target robot has no controllable joint indexes for IK")

    seed_qpos = sim.data.qpos[qpos_idx].copy()

    for _ in range(n_iters):
        sim.forward()
        cur_pos = sim.data.site_xpos[int(site_id)].copy()
        cur_xmat = sim.data.site_xmat[int(site_id)].reshape(3, 3).copy()
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
        J = np.vstack((jacp[:, qvel_idx], jacr[:, qvel_idx]))
        err = np.concatenate((pos_err, rot_err))
        damping = 1e-2
        JJt_inv = np.linalg.solve(J @ J.T + damping * damping * np.eye(6), np.eye(6))
        dq = J.T @ (JJt_inv @ err)
        # nullspace pull toward the seed configuration
        if nullspace_gain > 0.0:
            J_pinv = J.T @ JJt_inv
            null_proj = np.eye(J.shape[1]) - J_pinv @ J
            dq = dq + null_proj @ (
                nullspace_gain * (seed_qpos - sim.data.qpos[qpos_idx])
            )
        sim.data.qpos[qpos_idx] = sim.data.qpos[qpos_idx] + 0.5 * dq

    sim.data.qvel[:] = 0.0
    if hasattr(sim.data, "qacc"):
        sim.data.qacc[:] = 0.0
    if hasattr(sim.data, "ctrl"):
        sim.data.ctrl[:] = 0.0
    sim.forward()


def align_base_to_source_lowdim(env, source_lowdim: np.ndarray):
    base_pos = np.asarray(source_lowdim[0:3], dtype=np.float64)
    base_quat = np.asarray(source_lowdim[3:7], dtype=np.float64)
    try:
        env.env.init_robot_base_pos_anchor = base_pos
        env.env.init_robot_base_ori_anchor = T.mat2euler(T.quat2mat(base_quat))
        CoreEnvUtils.set_robot_to_position(env.env, base_pos)
        align_base_yaw_to_source(env, base_quat)
    except Exception as exc:
        print(f"[warn] base alignment failed: {exc}")


def align_base_yaw_to_source(env, base_quat: np.ndarray):
    """Rotate the mobile-base yaw joint so the target base world yaw matches the source.

    set_robot_to_position only aligns x/y; the yaw joint keeps whatever random
    spawn deviation the target env.reset() sampled, so we correct it here.
    """
    sim = env.env.sim
    yaw_addr = sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
    source_yaw = T.mat2euler(T.quat2mat(np.asarray(base_quat, dtype=np.float64)))[2]
    base_body_id = sim.model.body_name2id("mobilebase0_support")
    current_yaw = T.mat2euler(sim.data.body_xmat[base_body_id].reshape(3, 3))[2]
    delta = np.arctan2(
        np.sin(source_yaw - current_yaw), np.cos(source_yaw - current_yaw)
    )
    sim.data.qpos[yaw_addr] += delta
    sim.forward()


def restore_gripper_from_lowdim(env, source_lowdim: np.ndarray, source_env=None):
    """Match the target gripper opening to the source.

    NOTE: never write the source robot's raw gripper qpos into a different
    gripper model. The Robotiq85 is a closed-loop linkage with equality
    constraints; forcing Panda finger values into 2 of its 8 joints creates an
    inconsistent state whose constraint forces explode (QACC -> NaN), after
    which MuJoCo auto-resets the sim and teleports the robot to the staging
    area. If the gripper models differ we leave the target gripper at its
    reset (open) state -- the replayed gripper actions take over from there.
    """
    if source_env is None:
        return
    try:
        source_gripper_model = type(source_env.env.robots[0].gripper["right"]).__name__
        target_gripper_model = type(env.env.robots[0].gripper["right"]).__name__
    except Exception:
        return
    if source_gripper_model != target_gripper_model:
        return
    source_gripper = np.asarray(source_lowdim[14:16], dtype=np.float64)
    try:
        robot = env.env.robots[0]
        gripper_idx = flatten_index_container(
            getattr(robot, "_ref_gripper_joint_pos_indexes", [])
        )
        if gripper_idx.size == 0:
            return
        values = np.zeros(gripper_idx.shape[0], dtype=np.float64)
        count = min(values.shape[0], source_gripper.shape[0])
        values[:count] = source_gripper[:count]
        env.env.sim.data.qpos[gripper_idx] = values
        env.env.sim.forward()
    except Exception as exc:
        print(f"[warn] gripper alignment failed: {exc}")


def action_dict_from_vector(robot, action: np.ndarray) -> dict:
    controller = getattr(robot, "composite_controller", None)
    if controller is not None and hasattr(
        controller, "create_action_dict_from_action_vector"
    ):
        action_dict = controller.create_action_dict_from_action_vector(action)
        if len(action) > max(
            end_idx for _, end_idx in robot._action_split_indexes.values()
        ):
            action_dict["base_mode"] = action[-1:]
        return action_dict

    action_dict = {}
    for part_name, (start_idx, end_idx) in robot._action_split_indexes.items():
        action_dict[part_name] = action[start_idx:end_idx]
    if len(action) > max(
        end_idx for _, end_idx in robot._action_split_indexes.values()
    ):
        action_dict["base_mode"] = action[-1:]
    return action_dict


def adapt_action(source_action: np.ndarray, source_robot, target_robot) -> np.ndarray:
    action_dict = action_dict_from_vector(
        source_robot, np.asarray(source_action, dtype=np.float64)
    )
    return target_robot.create_action_vector(action_dict)


def render_frame(env, camera_name: str, height: int, width: int):
    return env.render(
        mode="rgb_array",
        height=height,
        width=width,
        camera_name=camera_name,
    )


def write_episode_video(path: Path, frames: list[np.ndarray], fps: int = 20):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def write_raw_hdf5(raw_path: Path, episodes: list[dict], env_meta: dict):
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "w") as f:
        data_grp = f.create_group("data")
        now = dt.datetime.now()
        data_grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        data_grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        data_grp.attrs["robocasa_version"] = robocasa.__version__
        data_grp.attrs["robosuite_version"] = robosuite.__version__
        data_grp.attrs["mujoco_version"] = mujoco.__version__
        data_grp.attrs["env"] = env_meta["env_name"]
        data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
        data_grp.attrs["env_info"] = json.dumps(env_meta["env_kwargs"], indent=4)
        data_grp.attrs["total"] = int(sum(ep["actions"].shape[0] for ep in episodes))

        for idx, episode in enumerate(episodes):
            demo_grp = data_grp.create_group(f"demo_{idx}")
            demo_grp.attrs["model_file"] = episode["model_file"]
            demo_grp.attrs["ep_meta"] = json.dumps(episode["ep_meta"], indent=4)
            demo_grp.attrs["source_episode"] = int(episode["source_episode"])
            demo_grp.attrs["num_samples"] = int(episode["actions"].shape[0])
            demo_grp.attrs["success"] = bool(episode["success"])
            demo_grp.attrs["initial_eef_pos_error"] = float(
                episode["initial_eef_pos_error"]
            )
            demo_grp.attrs["initial_eef_rot_error"] = float(
                episode["initial_eef_rot_error"]
            )
            demo_grp.create_dataset(
                "states", data=np.asarray(episode["states"], dtype=np.float64)
            )
            demo_grp.create_dataset(
                "actions", data=np.asarray(episode["actions"], dtype=np.float64)
            )


def reproduce_dataset(args):
    init_obs_utils()
    source_lerobot_dir = args.source_lerobot_dir
    with open(source_lerobot_dir / "extras" / "dataset_meta.json", "r") as f:
        dataset_meta = json.load(f)
    modality = load_modality(source_lerobot_dir)

    source_env_meta = dataset_meta["env_args"]
    target_env_meta = build_target_env_meta(
        source_env_meta,
        target_robot=args.target_robot,
        target_gripper=args.target_gripper,
    )

    task_name = source_env_meta["env_name"]
    date_tag = dt.datetime.now().strftime("%Y%m%d")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("/data1/local/robocasa/datasets/reproduced")
            / args.target_robot
            / task_name
            / date_tag
        )
    raw_path = output_dir / "raw.hdf5"
    video_dir = output_dir / "videos"

    print(f"[source] {source_lerobot_dir}")
    print(f"[target] {args.target_robot}/{args.target_gripper}")
    print(f"[output] {output_dir}")

    source_env = create_env(source_env_meta, seed=args.seed)
    target_env = create_env(target_env_meta, seed=args.seed)
    reproduced = []
    failures = []

    try:
        paths = episode_paths(source_lerobot_dir, args.start_episode, args.max_episodes)
        for parquet_path in tqdm(paths, desc="reproduce"):
            episode_num = int(parquet_path.stem.split("_")[-1])
            try:
                episode_df = read_episode_df(parquet_path)
                source_states, ep_meta, model_xml = load_episode_payload(
                    source_lerobot_dir, episode_num
                )
                source_actions = reorder_lerobot_actions_to_hdf5(
                    np.stack(episode_df["action"].to_numpy()),
                    modality=modality,
                )

                install_ep_meta(source_env, ep_meta)
                reset_source_to_episode(
                    source_env,
                    state=np.asarray(source_states[0]).reshape(-1),
                    ep_meta=ep_meta,
                    model_xml=model_xml,
                )
                source_eef_pos, source_eef_xmat = robot_eef_pose(source_env)

                reset_target_to_episode(target_env, ep_meta)
                source_lowdim0 = np.asarray(
                    episode_df.iloc[0]["observation.state"], dtype=np.float64
                ).reshape(-1)
                if not args.no_align_base:
                    align_base_to_source_lowdim(target_env, source_lowdim0)
                solve_ik_to_eef_pose(
                    target_env,
                    target_pos=source_eef_pos,
                    target_xmat=source_eef_xmat,
                    n_iters=args.ik_iters,
                )
                restore_gripper_from_lowdim(
                    target_env, source_lowdim0, source_env=source_env
                )

                target_eef_pos, target_eef_xmat = robot_eef_pose(target_env)
                eef_pos_error = np.linalg.norm(source_eef_pos - target_eef_pos)
                eef_rot_error = np.linalg.norm(source_eef_xmat - target_eef_xmat)
                model_file = target_env.get_state()["model"]

                target_states = []
                target_actions = []
                frames = []
                for step_idx, source_action in enumerate(source_actions):
                    target_action = adapt_action(
                        source_action,
                        source_robot=source_env.env.robots[0],
                        target_robot=target_env.env.robots[0],
                    )
                    target_actions.append(target_action)
                    target_env.step(target_action)
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

                reproduced.append(
                    {
                        "source_episode": episode_num,
                        "states": np.asarray(target_states, dtype=np.float64),
                        "actions": np.asarray(target_actions, dtype=np.float64),
                        "model_file": model_file,
                        "ep_meta": ep_meta,
                        "success": success,
                        "initial_eef_pos_error": eef_pos_error,
                        "initial_eef_rot_error": eef_rot_error,
                    }
                )
                print(
                    f"[episode {episode_num:06d}] success={int(success)} "
                    f"eef_pos_error={eef_pos_error:.5f} eef_rot_error={eef_rot_error:.5f}"
                )
            except Exception as exc:
                failures.append((episode_num, f"{type(exc).__name__}: {exc}"))
                print(f"[fail] episode {episode_num:06d}: {exc}")
                traceback.print_exc()

    finally:
        source_env.env.close()
        target_env.env.close()

    if not reproduced:
        raise RuntimeError(f"No episodes reproduced. Failures: {failures[:5]}")

    write_raw_hdf5(raw_path, reproduced, target_env_meta)
    print(f"[raw] {raw_path}")
    print(f"[summary] reproduced={len(reproduced)} failures={len(failures)}")
    if failures:
        print(f"[summary] first failures={failures[:5]}")
    return raw_path


def main():
    args = parse_args()
    reproduce_dataset(args)


if __name__ == "__main__":
    main()
