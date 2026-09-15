import argparse
import os

import imageio
import numpy as np
import robocasa  # noqa: F401 - registers RoboCasa environments with robosuite
import robosuite
from robosuite.controllers import load_composite_controller_config
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
    parser.add_argument("--num_steps", type=int, default=100)
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
    import pdb

    pdb.set_trace()
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
                f"{args.environment}_{safe_robot_name}_scene{scene_id}_random.mp4",
            )

            print(
                colored(
                    f"Initializing {args.environment} with {robot}, scene {scene_id}",
                    "yellow",
                )
            )
            env = make_env(args, robot, scene_id)
            try:
                successes = save_random_rollout_video(
                    env, video_path, args, rng, scene_id
                )
            finally:
                env.close()

            print(
                colored(
                    f"Saved {robot} scene {scene_id} video to {video_path} "
                    f"({successes}/{args.num_rollouts} successful rollouts)",
                    "green",
                )
            )


if __name__ == "__main__":
    main()
