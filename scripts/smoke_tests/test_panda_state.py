import argparse
import copy
import os

import imageio
import mujoco
import numpy as np
import robocasa  # noqa: F401 - registers RoboCasa environments with robosuite
import robosuite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", "--env", default="PickPlaceCounterToCabinet")
    parser.add_argument("--source-robot", default="PandaOmron")
    parser.add_argument("--target-robot", default="UR5eOmron")
    parser.add_argument("--gripper-types", default="default")
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument(
        "--split", choices=["pretrain", "target", "all"], default="target"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--video-path",
        default="/data1/local/robocasa/videos/panda_vs_ur5e_same_init.mp4",
    )
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--target-action-mode",
        choices=["mapped", "eef-delta"],
        default="mapped",
        help=(
            "mapped reorders the source action by robot part. eef-delta replaces "
            "the target arm action with an OSC delta that points at the source "
            "robot's next EEF pose."
        ),
    )
    parser.add_argument(
        "--track-source-eef",
        action="store_true",
        help=(
            "After applying mapped same actions, IK-align the target EEF to the "
            "source EEF before rendering. This is useful for visual EEF-path "
            "comparison, but it is not a pure target-robot physics rollout."
        ),
    )
    parser.add_argument("--track-ik-iters", type=int, default=80)
    return parser.parse_args()


def split_config(split):
    if split == "target":
        return {"obj_instance_split": "target", "clutter_mode": 1}
    if split == "pretrain":
        return {"obj_instance_split": "pretrain"}
    return {"obj_instance_split": None}


def make_env(args, robot):
    layout_style_id = args.scene_id + 1
    return robosuite.make(
        env_name=args.environment,
        robots=robot,
        gripper_types=args.gripper_types,
        controller_configs=load_composite_controller_config(robot=robot),
        translucent_robot=False,
        layout_and_style_ids=[(layout_style_id, layout_style_id)],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        control_freq=20,
        seed=args.seed,
        **split_config(args.split),
    )


def object_joint_qpos(env):
    return {
        name: env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        for name, obj in env.objects.items()
    }


def absolute_object_state(env):
    values = []
    for name in sorted(env.obj_body_id):
        body_id = env.obj_body_id[name]
        values.extend(env.sim.data.body_xpos[body_id])
        values.extend(env.sim.data.body_xquat[body_id])
    return np.asarray(values, dtype=np.float64)


def canonicalize_object_state_quats(object_state):
    """
    RoboCasa object-state is [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
    per object. Quaternions q and -q represent the same rotation, so canonicalize
    signs before using a raw vector norm for comparison.
    """
    state = np.asarray(object_state, dtype=np.float64).copy()
    per_obj_dim = 14
    if state.shape[0] % per_obj_dim != 0:
        return state
    for start in range(0, state.shape[0], per_obj_dim):
        for quat_start in (start + 3, start + 10):
            quat = state[quat_start : quat_start + 4]
            if quat[-1] < 0:
                state[quat_start : quat_start + 4] = -quat
    return state


def sign_invariant_object_state_diff(source_object_state, target_object_state):
    source = canonicalize_object_state_quats(source_object_state)
    target = canonicalize_object_state_quats(target_object_state)
    return np.linalg.norm(source - target)


def copy_scene_objects(source_env, target_env):
    """
    Make target_env use the same sampled scene/object placement as source_env.

    set_ep_meta() alone fixes layout/style/object configs, but object placements are
    sampled during reset. Copying object joint qpos after target reset makes the
    absolute object poses identical across robots.
    """
    target_env.set_ep_meta(copy.deepcopy(source_env.get_ep_meta()))
    target_env.reset()

    source_qpos = object_joint_qpos(source_env)
    missing = []
    for name, qpos in source_qpos.items():
        if name not in target_env.objects:
            missing.append(name)
            continue
        target_obj = target_env.objects[name]
        target_env.sim.data.set_joint_qpos(target_obj.joints[0], qpos)
    target_env.sim.forward()
    return missing


def base_qpos(env):
    robot = env.robots[0]
    values = []
    for joint_name in robot.robot_model.base_joints:
        values.append(np.asarray(env.sim.data.get_joint_qpos(joint_name)).reshape(-1))
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


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


def sync_controller_goals(env):
    """Reset controller targets after directly editing qpos / qvel."""
    for robot in env.robots:
        robot.composite_controller.update_state()
        robot.composite_controller.reset()


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
    sim = env.sim
    robot = env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    sim.forward()
    return (
        sim.data.site_xpos[int(site_id)].copy(),
        sim.data.site_xmat[int(site_id)].reshape(3, 3).copy(),
    )


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
        dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), err)
        sim.data.qpos[qpos_idx] = sim.data.qpos[qpos_idx] + 0.5 * dq

    sim.data.qvel[:] = 0.0
    if hasattr(sim.data, "qacc"):
        sim.data.qacc[:] = 0.0
    if hasattr(sim.data, "ctrl"):
        sim.data.ctrl[:] = 0.0
    sim.forward()
    sync_controller_goals(env)


def action_dict_from_vector(robot, action):
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


def adapt_action(source_env, target_env, source_action):
    action_dict = action_dict_from_vector(source_env.robots[0], source_action)
    return target_env.robots[0].create_action_vector(action_dict)


def inverse_scale_action(controller, scaled_delta):
    if controller.action_scale is None:
        controller.scale_action(np.zeros(controller.control_dim))
    return (
        scaled_delta - controller.action_output_transform
    ) / controller.action_scale + controller.action_input_transform


def xmat_in_controller_origin(env, xmat):
    robot = env.robots[0]
    controller = robot.part_controllers[robot.arms[0]]
    robot.composite_controller.update_state()
    if controller.input_ref_frame == "base":
        return controller.origin_ori.T @ xmat
    if controller.input_ref_frame == "world":
        return xmat
    raise ValueError(f"Unsupported input_ref_frame: {controller.input_ref_frame}")


def pos_in_controller_origin(env, pos):
    robot = env.robots[0]
    controller = robot.part_controllers[robot.arms[0]]
    robot.composite_controller.update_state()
    if controller.input_ref_frame == "base":
        return controller.world_to_origin_frame(pos)
    if controller.input_ref_frame == "world":
        return pos
    raise ValueError(f"Unsupported input_ref_frame: {controller.input_ref_frame}")


def arm_action_to_eef_pose(target_env, desired_pos, desired_xmat):
    robot = target_env.robots[0]
    arm = robot.arms[0]
    controller = robot.part_controllers[arm]
    target_env.sim.forward()
    controller.update(force=True)

    current_pos, current_xmat = eef_pose(target_env)
    desired_pos_origin = pos_in_controller_origin(target_env, desired_pos)
    current_pos_origin = pos_in_controller_origin(target_env, current_pos)
    pos_delta = desired_pos_origin - current_pos_origin

    desired_xmat_origin = xmat_in_controller_origin(target_env, desired_xmat)
    current_xmat_origin = xmat_in_controller_origin(target_env, current_xmat)
    rot_delta = desired_xmat_origin @ current_xmat_origin.T
    ori_delta = Rotation.from_matrix(rot_delta).as_rotvec()

    scaled_delta = np.concatenate([pos_delta, ori_delta])
    return np.clip(inverse_scale_action(controller, scaled_delta), -1.0, 1.0)


def retarget_action_to_source_eef(
    source_env, target_env, source_action, desired_pos, desired_xmat
):
    action_dict = action_dict_from_vector(source_env.robots[0], source_action)
    target_arm = target_env.robots[0].arms[0]
    action_dict[target_arm] = arm_action_to_eef_pose(
        target_env, desired_pos, desired_xmat
    )
    return target_env.robots[0].create_action_vector(action_dict)


def choose_camera(env, requested_camera):
    camera_names = set(env.sim.model.camera_names)
    if requested_camera != "auto":
        if requested_camera not in camera_names:
            raise ValueError(
                f"No camera named {requested_camera}. Available: {sorted(camera_names)}"
            )
        return requested_camera

    for camera_name in (
        "robot0_agentview_center",
        "robot0_robotview",
        "robot0_eye_in_hand",
    ):
        if camera_name in camera_names:
            return camera_name
    return sorted(camera_names)[0]


def render(env, camera_name, height, width):
    return env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]


def rollout_same_actions_to_video(source_env, target_env, args):
    rng = np.random.default_rng(args.seed)
    source_camera = choose_camera(source_env, args.camera)
    target_camera = choose_camera(target_env, args.camera)
    os.makedirs(os.path.dirname(args.video_path), exist_ok=True)

    writer = imageio.get_writer(args.video_path, fps=args.fps)
    eef_pos_diffs = []
    eef_rot_diffs = []
    try:
        for _ in range(args.num_steps):
            low, high = source_env.action_spec
            source_action = rng.uniform(low=low, high=high)

            source_env.step(source_action)
            source_eef_pos, source_eef_xmat = eef_pose(source_env)
            if args.target_action_mode == "eef-delta":
                target_action = retarget_action_to_source_eef(
                    source_env,
                    target_env,
                    source_action,
                    source_eef_pos,
                    source_eef_xmat,
                )
            else:
                target_action = adapt_action(source_env, target_env, source_action)
            target_env.step(target_action)

            if args.track_source_eef:
                solve_ik_to_eef_pose(
                    target_env,
                    source_eef_pos,
                    source_eef_xmat,
                    n_iters=args.track_ik_iters,
                )
            target_eef_pos, target_eef_xmat = eef_pose(target_env)
            eef_pos_diffs.append(np.linalg.norm(source_eef_pos - target_eef_pos))
            eef_rot_diffs.append(np.linalg.norm(source_eef_xmat - target_eef_xmat))

            source_frame = render(source_env, source_camera, args.height, args.width)
            target_frame = render(target_env, target_camera, args.height, args.width)
            separator = np.zeros((args.height, 16, 3), dtype=np.uint8)
            writer.append_data(
                np.concatenate([source_frame, separator, target_frame], axis=1)
            )
    finally:
        writer.close()

    return {
        "final_eef_pos_diff": eef_pos_diffs[-1] if eef_pos_diffs else 0.0,
        "max_eef_pos_diff": max(eef_pos_diffs) if eef_pos_diffs else 0.0,
        "final_eef_rot_diff": eef_rot_diffs[-1] if eef_rot_diffs else 0.0,
        "max_eef_rot_diff": max(eef_rot_diffs) if eef_rot_diffs else 0.0,
    }


def main():
    args = parse_args()
    source_env = make_env(args, args.source_robot)
    target_env = make_env(args, args.target_robot)

    try:
        source_obs = source_env.reset()
        target_obs_plain = target_env.reset()

        plain_abs_diff = np.linalg.norm(
            absolute_object_state(source_env) - absolute_object_state(target_env)
        )
        plain_object_state_diff = np.linalg.norm(
            source_obs["object-state"] - target_obs_plain["object-state"]
        )

        missing = copy_scene_objects(source_env, target_env)
        base_diff_after_object_copy = np.linalg.norm(
            base_qpos(source_env) - base_qpos(target_env)
        )
        copy_base_qpos(source_env, target_env)
        base_diff_after_base_copy = np.linalg.norm(
            base_qpos(source_env) - base_qpos(target_env)
        )
        target_obs_same_scene = target_env._get_observations(force_update=True)

        same_scene_abs_diff = np.linalg.norm(
            absolute_object_state(source_env) - absolute_object_state(target_env)
        )
        same_scene_object_state_diff = np.linalg.norm(
            source_obs["object-state"] - target_obs_same_scene["object-state"]
        )
        same_scene_object_state_canon_diff = sign_invariant_object_state_diff(
            source_obs["object-state"], target_obs_same_scene["object-state"]
        )

        source_eef_pos, source_eef_xmat = eef_pose(source_env)
        solve_ik_to_eef_pose(target_env, source_eef_pos, source_eef_xmat)
        target_obs_same_scene_eef = target_env._get_observations(force_update=True)
        target_eef_pos, target_eef_xmat = eef_pose(target_env)
        base_diff_after_ik = np.linalg.norm(
            base_qpos(source_env) - base_qpos(target_env)
        )
        eef_pos_diff = np.linalg.norm(source_eef_pos - target_eef_pos)
        eef_rot_diff = np.linalg.norm(source_eef_xmat - target_eef_xmat)
        same_scene_eef_object_state_diff = np.linalg.norm(
            source_obs["object-state"] - target_obs_same_scene_eef["object-state"]
        )
        same_scene_eef_object_state_canon_diff = sign_invariant_object_state_diff(
            source_obs["object-state"], target_obs_same_scene_eef["object-state"]
        )

        print(f"source robot: {args.source_robot}")
        print(f"target robot: {args.target_robot}")
        print(f"scene id: {args.scene_id} -> layout/style {args.scene_id + 1}")
        print(f"object names source: {sorted(source_env.objects.keys())}")
        print(f"object names target: {sorted(target_env.objects.keys())}")
        if missing:
            print(f"missing target objects: {missing}")

        print("\nBefore copying source object qpos:")
        print(f"  absolute object pose diff: {plain_abs_diff:.8f}")
        print(f"  full object-state diff:    {plain_object_state_diff:.8f}")

        print("\nAfter copying source object qpos:")
        print(f"  absolute object pose diff: {same_scene_abs_diff:.8f}")
        print(f"  base qpos diff before base copy: {base_diff_after_object_copy:.8f}")
        print(f"  base qpos diff after base copy:  {base_diff_after_base_copy:.8f}")
        print(f"  full object-state diff:    {same_scene_object_state_diff:.8f}")
        print(f"  sign-invariant diff:       {same_scene_object_state_canon_diff:.8f}")

        print("\nAfter also matching target eef with IK:")
        print(f"  base qpos diff after IK:   {base_diff_after_ik:.8f}")
        print(f"  target eef position diff:  {eef_pos_diff:.8f}")
        print(f"  target eef rotation diff:  {eef_rot_diff:.8f}")
        print(f"  full object-state diff:    {same_scene_eef_object_state_diff:.8f}")
        print(
            f"  sign-invariant diff:       {same_scene_eef_object_state_canon_diff:.8f}"
        )
        print(
            "\nNote: full object-state includes obj_to_eef_pos/quat, so it can "
            "differ across robots even when absolute object poses are identical. "
            "IK can reduce that relative-state difference, but gripper geometry "
            "and robot-specific observables can still make it nonzero."
        )

        rollout_stats = rollout_same_actions_to_video(source_env, target_env, args)
        print(f"\nSaved side-by-side rollout video: {args.video_path}")
        print(
            f"Rollout EEF differences after {args.target_action_mode} target actions:"
        )
        print(f"  final position diff: {rollout_stats['final_eef_pos_diff']:.8f}")
        print(f"  max position diff:   {rollout_stats['max_eef_pos_diff']:.8f}")
        print(f"  final rotation diff: {rollout_stats['final_eef_rot_diff']:.8f}")
        print(f"  max rotation diff:   {rollout_stats['max_eef_rot_diff']:.8f}")
    finally:
        source_env.close()
        target_env.close()


if __name__ == "__main__":
    main()
