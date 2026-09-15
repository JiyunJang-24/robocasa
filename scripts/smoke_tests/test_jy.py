import argparse
import copy
import os

import imageio
import mujoco
import numpy as np
import robocasa  # noqa: F401 - registers RoboCasa environments with robosuite
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
from termcolor import colored
from tqdm import trange

# Smoke-tested in the `robocasa` conda env with PickPlaceCounterToCabinet,
# target scene 0, 1 random step, and offscreen render.
#
# Robot/default-gripper pairs that worked in this checkout:
#   GR1FloatingBody/default, IIWAOmron/default, PandaOmron/default,
#   Tiago/default, UR5eOmron/default.
#
# For GR1FloatingBody, PandaOmron, Tiago, and UR5eOmron, these explicit
# gripper_types also completed reset/step/render:
#   PandaGripper, Robotiq85Gripper, Robotiq140Gripper, RethinkGripper,
#   WipingGripper, JacoThreeFingerGripper, JacoThreeFingerDexterousGripper,
#   RobotiqThreeFingerGripper, RobotiqThreeFingerDexterousGripper,
#   InspireRightHand, FourierRightHand, XArm7Gripper, SuctionGripper, BDGripper.
#
# Robot/default-gripper pairs that failed in this checkout:
#   Baxter/default: missing robot0_agentview_center render camera at reset.
#   GR1/default: missing mobilebase joint.
#   GR1FixedLowerBody/default: WHOLE_BODY_MINK_IK controller dependency missing.
#   IIWA/default, Kinova3/default, Panda/default, PandaDexLH/default,
#   PandaDexRH/default, Sawyer/default, SpotWithArm/default, UR5e/default:
#   missing robot0_agentview_center at reset or missing mobilebase joint.
#
# For Franka-style comparisons, use PandaOmron instead of plain Panda.
# For KUKA iiwa-style comparisons, use IIWAOmron instead of plain IIWA.
# For UR5e-style comparisons, use UR5eOmron instead of plain UR5e.
# Common camera names differ by robot. Use --camera auto unless you need a
# specific view; auto tries agentview_center, robotview, then eye_in_hand.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run random-action RoboCasa rollouts for one or more robots."
    )
    parser.add_argument(
        "--environment",
        "--env",
        dest="environment",
        type=str,
        default="PickPlaceCounterToCabinet",
        help="RoboCasa task / robosuite env name.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["PandaOmron"],
        help="Robot names to test. A separate video is written for each robot.",
    )
    parser.add_argument(
        "--gripper_types",
        type=str,
        default="default",
        help="Gripper type passed to robosuite.make.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="target",
        choices=["pretrain", "target", "all"],
        help="Object split to use. Default uses target scenes for scene ids 0, 1, 2.",
    )
    parser.add_argument(
        "--scene_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="0-based scene ids to test in order. Scene N maps to layout/style id N+1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--video_dir", type=str, default="/data1/local/robocasa/videos")
    parser.add_argument(
        "--camera",
        type=str,
        default="auto",
        help="Camera name for video rendering. Use 'auto' for robot-compatible fallback.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if any(scene_id < 0 for scene_id in args.scene_ids):
        raise ValueError("--scene_ids must be 0-based non-negative integers")
    return args


def split_config(split):
    if split == "target":
        return {"obj_instance_split": "target", "clutter_mode": 1}
    if split == "pretrain":
        return {"obj_instance_split": "pretrain"}
    return {"obj_instance_split": None}


def make_env(args, robot, scene_id):
    controller_config = load_composite_controller_config(robot=robot)
    layout_style_id = scene_id + 1
    config = {
        "env_name": args.environment,
        "robots": robot,
        "gripper_types": args.gripper_types,
        "controller_configs": controller_config,
        "translucent_robot": False,
        "layout_and_style_ids": [(layout_style_id, layout_style_id)],
        **split_config(args.split),
    }

    return robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        control_freq=args.fps,
        seed=args.seed,
    )


def object_joint_qpos(env):
    return {
        name: env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        for name, obj in env.objects.items()
    }


def copy_scene_objects(source_env, target_env):
    source_qpos = object_joint_qpos(source_env)
    for name, qpos in source_qpos.items():
        if name not in target_env.objects:
            continue
        target_env.sim.data.set_joint_qpos(target_env.objects[name].joints[0], qpos)
    target_env.sim.forward()


def copy_base_qpos(source_env, target_env):
    source_robot = source_env.robots[0]
    target_robot = target_env.robots[0]
    if len(source_robot.robot_model.base_joints) != len(
        target_robot.robot_model.base_joints
    ):
        raise ValueError(
            "Cannot copy base qpos: source and target base joint counts differ "
            f"({len(source_robot.robot_model.base_joints)} vs "
            f"{len(target_robot.robot_model.base_joints)})"
        )

    for source_joint, target_joint in zip(
        source_robot.robot_model.base_joints,
        target_robot.robot_model.base_joints,
    ):
        target_env.sim.data.set_joint_qpos(
            target_joint,
            source_env.sim.data.get_joint_qpos(source_joint).copy(),
        )
    target_env.sim.forward()


def obs_base_yaw(obs):
    return T.mat2euler(
        T.quat2mat(np.asarray(obs["robot0_base_quat"], dtype=np.float64))
    )[2]


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def current_obs(env):
    env.sim.forward()
    return env._get_observations(force_update=True)


def align_base_to_source_obs(source_obs, target_env, n_iters=5):
    """Match target mobile-base observation after qpos copy.

    Omron mobile bases expose qpos as forward / side / yaw, while RoboCasa's
    robot0_base_pos is measured from the mobilebase0_center site. Copying qpos
    is the main alignment step; the small iterative correction below removes
    any remaining x/y/yaw observation error from robot-specific site offsets.
    """
    target_robot = target_env.robots[0]
    base_joints = list(target_robot.robot_model.base_joints)
    if len(base_joints) < 3:
        return current_obs(target_env)

    forward_joint, side_joint, yaw_joint = base_joints[:3]
    for _ in range(n_iters):
        obs = current_obs(target_env)
        pos_err = np.asarray(source_obs["robot0_base_pos"][:2]) - np.asarray(
            obs["robot0_base_pos"][:2]
        )
        yaw_err = wrap_to_pi(obs_base_yaw(source_obs) - obs_base_yaw(obs))
        if np.linalg.norm(pos_err) < 1e-5 and abs(yaw_err) < 1e-5:
            break

        # For robosuite's mobile base XML, x is controlled by -side and y by +forward.
        target_env.sim.data.set_joint_qpos(
            forward_joint,
            target_env.sim.data.get_joint_qpos(forward_joint) + pos_err[1],
        )
        target_env.sim.data.set_joint_qpos(
            side_joint,
            target_env.sim.data.get_joint_qpos(side_joint) - pos_err[0],
        )
        target_env.sim.data.set_joint_qpos(
            yaw_joint,
            target_env.sim.data.get_joint_qpos(yaw_joint) + yaw_err,
        )
        target_env.sim.forward()
    return current_obs(target_env)


def flatten_index_container(value):
    if isinstance(value, dict):
        parts = [np.asarray(v, dtype=np.int64).reshape(-1) for v in value.values()]
        return (
            np.concatenate(parts).astype(np.int64)
            if parts
            else np.zeros(0, dtype=np.int64)
        )
    return np.asarray(value, dtype=np.int64).reshape(-1)


def eef_pose(env):
    robot = env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    env.sim.forward()
    return (
        env.sim.data.site_xpos[int(site_id)].copy(),
        env.sim.data.site_xmat[int(site_id)].reshape(3, 3).copy(),
    )


def sync_controller_goals(env):
    for robot in env.robots:
        robot.composite_controller.update_state()
        robot.composite_controller.reset()


def align_quat_sign(quat, reference_quat):
    quat = np.asarray(quat).copy()
    return -quat if np.dot(quat, reference_quat) < 0 else quat


def solve_ik_to_eef_pose(env, target_pos, target_xmat, n_iters=250):
    sim = env.sim
    robot = env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]

    qpos_idx = flatten_index_container(robot._ref_joint_pos_indexes)
    qvel_idx = flatten_index_container(robot._ref_joint_vel_indexes)
    if qpos_idx.size == 0 or qvel_idx.size == 0:
        raise ValueError("Target robot has no controllable joint indexes for IK")

    raw_model = sim.model._model if hasattr(sim.model, "_model") else sim.model
    raw_data = sim.data._data if hasattr(sim.data, "_data") else sim.data
    joint_ids = flatten_index_container(robot._ref_joint_indexes)
    limited = np.asarray(sim.model.jnt_limited[joint_ids], dtype=bool)
    joint_ranges = np.asarray(sim.model.jnt_range[joint_ids], dtype=np.float64)

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
        mujoco.mj_jacSite(raw_model, raw_data, jacp, jacr, int(site_id))
        jac = np.vstack((jacp[:, qvel_idx], jacr[:, qvel_idx]))
        err = np.concatenate((pos_err, rot_err))
        damping = 1e-2
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(6), err)
        sim.data.qpos[qpos_idx] = sim.data.qpos[qpos_idx] + 0.5 * dq
        if joint_ids.size == qpos_idx.size and np.any(limited):
            sim.data.qpos[qpos_idx[limited]] = np.clip(
                sim.data.qpos[qpos_idx[limited]],
                joint_ranges[limited, 0],
                joint_ranges[limited, 1],
            )

    sim.data.qvel[:] = 0.0
    if hasattr(sim.data, "qacc"):
        sim.data.qacc[:] = 0.0
    if hasattr(sim.data, "ctrl"):
        sim.data.ctrl[:] = 0.0
    sim.forward()
    sync_controller_goals(env)


def align_target_to_source_state(source_env, target_env, source_obs):
    """RoboCasa version of the LIBERO-style cross-embodiment init conversion.

    The order matters:
      1. copy non-robot scene object qpos from the source scene,
      2. copy and observation-correct the mobile base,
      3. solve target arm IK to the source EEF pose after the base is fixed.
    """
    copy_scene_objects(source_env, target_env)
    copy_base_qpos(source_env, target_env)
    align_base_to_source_obs(source_obs, target_env)

    source_eef_pos, source_eef_xmat = eef_pose(source_env)
    solve_ik_to_eef_pose(target_env, source_eef_pos, source_eef_xmat)
    return current_obs(target_env)


def make_env_another_robot(args, robot, scene_id, source_env=None):
    owns_source_env = source_env is None
    Panda_env = (
        source_env if source_env is not None else make_env(args, "PandaOmron", scene_id)
    )
    Panda_env_reset = None if source_env is not None else Panda_env.reset()
    controller_config = load_composite_controller_config(robot=robot)
    layout_style_id = scene_id + 1
    config = {
        "env_name": args.environment,
        "robots": robot,
        "gripper_types": args.gripper_types,
        "controller_configs": controller_config,
        "translucent_robot": False,
        "layout_and_style_ids": [(layout_style_id, layout_style_id)],
        **split_config(args.split),
    }

    target_env = robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        control_freq=args.fps,
        seed=args.seed,
    )

    original_reset = target_env.reset
    original_close = target_env.close

    def aligned_reset(*reset_args, **reset_kwargs):
        nonlocal Panda_env_reset
        if owns_source_env:
            Panda_env_reset = Panda_env.reset()
        else:
            Panda_env_reset = current_obs(Panda_env)
        target_env.set_ep_meta(copy.deepcopy(Panda_env.get_ep_meta()))
        original_reset(*reset_args, **reset_kwargs)

        obs = align_target_to_source_state(Panda_env, target_env, Panda_env_reset)
        target_env._source_panda_reset = Panda_env_reset
        for quat_key in ("robot0_eef_quat", "robot0_base_quat"):
            if quat_key in obs and quat_key in Panda_env_reset:
                obs[quat_key] = align_quat_sign(
                    obs[quat_key], Panda_env_reset[quat_key]
                )
        return obs

    def aligned_close():
        try:
            if owns_source_env:
                Panda_env.close()
        finally:
            original_close()

    target_env.reset = aligned_reset
    target_env.close = aligned_close
    target_env._source_panda_env = Panda_env
    target_env._source_panda_reset = Panda_env_reset
    return target_env


def random_action(env, rng):
    low, high = env.action_spec
    return rng.uniform(low=low, high=high)


def choose_camera(env, requested_camera):
    camera_names = set(env.sim.model.camera_names)
    if requested_camera != "auto":
        if requested_camera not in camera_names:
            raise ValueError(
                f'No camera named "{requested_camera}". Available: {sorted(camera_names)}'
            )
        return requested_camera

    for camera_name in (
        "robot0_agentview_center",
        "robot0_robotview",
        "robot0_eye_in_hand",
    ):
        if camera_name in camera_names:
            return camera_name

    if not camera_names:
        raise ValueError("No cameras are available in this environment")
    return sorted(camera_names)[0]


def save_random_rollout_video(env, video_path, args, rng, scene_id):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=args.fps)
    successes = 0
    camera_name = None

    try:
        for rollout_i in range(args.num_rollouts):
            env.reset()
            if camera_name is None:
                camera_name = choose_camera(env, args.camera)
            for step_i in trange(
                args.num_steps,
                desc=f"scene {scene_id} rollout {rollout_i + 1}/{args.num_rollouts}",
                leave=False,
            ):
                env.step(random_action(env, rng))
                frame = env.sim.render(
                    height=args.height,
                    width=args.width,
                    camera_name=camera_name,
                )[::-1]
                writer.append_data(frame)

                if env._check_success():
                    successes += 1
                    break
    finally:
        writer.close()

    return successes


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    for robot in args.robots:
        safe_robot_name = robot.replace("/", "_")
        for scene_id in args.scene_ids:
            video_path = os.path.join(
                args.video_dir,
                f"debug_{args.environment}_{safe_robot_name}_scene{scene_id}_random.mp4",
            )

            print(
                colored(
                    f"Initializing {args.environment} with {robot}, scene {scene_id}",
                    "yellow",
                )
            )
            env = make_env(args, robot, scene_id)
            ur5e_args = copy.deepcopy(args)
            ur5e_args.robots = ["UR5eOmron"]
            ur5e_env = make_env_another_robot(
                ur5e_args, "UR5eOmron", scene_id, source_env=env
            )
            franka_reset = env.reset()
            ur5e_reset = ur5e_env.reset()
            import pdb

            pdb.set_trace()
            try:
                successes = save_random_rollout_video(
                    env, video_path, args, rng, scene_id
                )
                successes_ur5e = save_random_rollout_video(
                    ur5e_env,
                    video_path.replace(safe_robot_name, "UR5eOmron"),
                    ur5e_args,
                    rng,
                    scene_id,
                )

            finally:
                env.close()
                ur5e_env.close()

            print(
                colored(
                    f"Saved {robot} scene {scene_id} video to {video_path} "
                    f"({successes}/{args.num_rollouts} successful rollouts)",
                    "green",
                )
            )


if __name__ == "__main__":
    main()
