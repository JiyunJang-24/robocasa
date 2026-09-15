#!/usr/bin/env python3
"""
Step 1/2 of the cross-embodiment atomic-action replay demo (see
atomic_target_rollout.py for step 2 and the full write-up).

Runs under the CONDA robocasa environment (the one with the v1.0 "atomic"-
named task classes, e.g. PickPlaceCounterToSink). Faithfully replays one
recorded PandaOmron episode's *stored MuJoCo states* (kinematic ground truth,
not action-driven) to produce the reference video, and writes a small JSON
"bridge" file (raw 12-D actions, episode metadata, and the manipulated
object's initial world pose) for atomic_target_rollout.py -- which runs in
BARX's separate uv virtualenv (its own "robocasa" package, incompatible with
this one; hence the two-process split) -- to pick up.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v2 as imageio
import numpy as np
import cv2
import robosuite.utils.transform_utils as T

ROOT = Path("/data1/local/robocasa")
SOURCE_TASK = "PickPlaceCounterToSink"
CAMERA = "robot0_agentview_center"
IMAGE_SIZE = 480


def load_gen_module():
    spec = importlib.util.spec_from_file_location(
        "mg_ur5e_gen", ROOT / "generate_mimicgen_ur5e_dataset.py"
    )
    mg_gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg_gen)
    mg_gen.patch_python_paths()
    mg_gen.patch_huggingface_hub_compat()
    mg_gen.patch_robomimic_algo_imports()
    return mg_gen


def flattened_base_pose(base_pos: np.ndarray, base_ori: np.ndarray) -> np.ndarray:
    """A 4x4 pose anchored at the base's (x, y) with z=0 and pure yaw rotation
    (no roll/pitch, which the flat-floored Omron mobile base never has
    anyway). Used instead of the full 3-D base pose so that the transplanted
    object's WORLD HEIGHT passes through unchanged: different embodiments'
    calibrated base/torso heights differ (that's the whole point of BARX's
    per-embodiment offsets), but the kitchen counter's height does not, so
    height should not be re-expressed relative to a moving base frame."""
    yaw = np.arctan2(base_ori[1, 0], base_ori[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    r_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T.make_pose(np.array([base_pos[0], base_pos[1], 0.0]), r_yaw)


def live_base_pose(env):
    """robot.base_pos/base_ori are cached at robot construction time (reading
    the "robot0_base" body, which sits at the [10, 10, 0] staging position
    before the mobile base's own joints move it into the scene) and are never
    refreshed by reset_to()'s low-level state restore. mobilebase0_support is
    the live, post-forward-kinematics body that actually reflects the current
    base pose in world coordinates."""
    sim = env.env.sim
    body_id = sim.model.body_name2id("mobilebase0_support")
    return (
        sim.data.body_xpos[body_id].copy(),
        sim.data.body_xmat[body_id].reshape(3, 3).copy(),
    )


def label_frame(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    bar_h = 22 * len(lines) + 10
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 20 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--bridge-dir",
        type=Path,
        default=Path(
            "/data1/local/BARX/datasets/cross_embodiment/atomic_action_replay/_bridge"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "datasets" / "mg_runs" / "atomic_demo_work",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.bridge_dir.mkdir(parents=True, exist_ok=True)

    mg_gen = load_gen_module()
    source_dir = mg_gen.latest_lerobot_dir(
        ROOT / "datasets" / "v1.0", "pretrain", SOURCE_TASK
    )
    build_args = SimpleNamespace(
        task=SOURCE_TASK,
        split="pretrain",
        source_root=ROOT / "datasets" / "v1.0",
        source_lerobot_dir=None,
        work_dir=args.work_dir,
        source_n=args.episode + 1,
        force_source_raw=True,
    )
    raw_path = mg_gen.build_source_raw_hdf5(build_args, source_dir)

    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )

    with h5py.File(raw_path, "r") as f:
        demo = f[f"data/demo_{args.episode}"]
        env_meta = json.loads(f["data"].attrs["env_args"])
        states = np.asarray(demo["states"])
        actions = np.asarray(demo["actions"], dtype=np.float64)
        ep_meta = json.loads(demo.attrs["ep_meta"])
        model_file = demo.attrs["model_file"]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
    )
    n = min(len(states), len(actions))
    frames = []
    obj_pos0 = obj_quat0 = None
    try:
        for t in range(n):
            reset_state = {"states": states[t]}
            if t == 0:
                reset_state["model"] = model_file
                reset_state["ep_meta"] = json.dumps(ep_meta)
            env.reset_to(reset_state)
            if t == 0:
                body_id = env.env.obj_body_id["obj"]
                obj_pos0 = env.env.sim.data.body_xpos[body_id].copy()
                obj_quat0 = env.env.sim.data.body_xquat[body_id].copy()
                base_pos0, base_ori0 = live_base_pose(env)
            frame = env.render(
                mode="rgb_array",
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                camera_name=CAMERA,
            )
            frames.append(
                label_frame(
                    frame,
                    [
                        "PandaOmron (RoboCasa v1.0 atomic demo)",
                        f"step {t:03d}/{n - 1}  \"{ep_meta['lang']}\"",
                    ],
                )
            )
        success = bool(env.env._check_success())
    finally:
        env.env.close()

    video_path = args.bridge_dir / "source_frames.mp4"
    imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)

    # Object pose expressed relative to the ROBOT BASE frame, not world/fixture
    # coordinates. Two reasons this frame was chosen over the sink fixture's:
    #  1. BARX's frozen asset library only ships 10 layouts x 12 styles (an
    #     older, much smaller RoboCasa scene registry), while this v1.0
    #     dataset was collected on a newer RoboCasa with dozens of
    #     layouts/styles (this episode uses layout 36 / style 30, both
    #     entirely absent from BARX's asset directory) -- so matching
    #     layout/style world coordinates is impossible, and a sink-relative
    #     transplant is fragile anyway: individual kitchen layouts can mirror
    #     which side of the sink the counter continues on, which silently
    #     flips a sink-frame offset onto the wrong (unreachable, often
    #     off-camera) side.
    #  2. BARX's whole cross-embodiment calibration (BARX_ROBOT_INITIAL_QPOS /
    #     BARX_ROBOT_POSITION_OFFSETS) is defined precisely to make different
    #     robots' BASE frames face the counter/sink the same way, which is
    #     also the action space's own reference frame (OSC input_ref_frame=
    #     "base"). Reusing that same frame for the object transplant is the
    #     one choice that is consistent, by construction, with how BARX
    #     already equalizes embodiments.
    T_base_flat = flattened_base_pose(base_pos0, base_ori0)
    T_obj_world = T.make_pose(
        obj_pos0, T.quat2mat(T.convert_quat(obj_quat0, to="xyzw"))
    )
    T_obj_in_base = T.pose_in_A_to_pose_in_B(
        pose_A=T_obj_world, pose_A_in_B=T.pose_inv(T_base_flat)
    )

    bridge = {
        "source_task": SOURCE_TASK,
        "episode": args.episode,
        "layout_id": ep_meta["layout_id"],
        "style_id": ep_meta["style_id"],
        "lang": ep_meta["lang"],
        "actions": actions[:n].tolist(),
        "obj_pos0": obj_pos0.tolist(),
        "obj_quat0": obj_quat0.tolist(),
        "base_pos0": base_pos0.tolist(),
        "base_ori0": base_ori0.tolist(),
        "obj_pose_in_base_frame": T_obj_in_base.tolist(),
        "source_success": success,
        "source_frames_video": str(video_path),
        "num_frames": n,
    }
    bridge_path = args.bridge_dir / "bridge.json"
    with open(bridge_path, "w") as f:
        json.dump(bridge, f, indent=2)

    print(
        f"[source] episode {args.episode}: layout={ep_meta['layout_id']} style={ep_meta['style_id']}"
    )
    print(f"[source] lang: {ep_meta['lang']}")
    print(f"[source] frames={n} success={success}")
    print(f"[source] wrote {video_path}")
    print(f"[bridge] wrote {bridge_path}")


if __name__ == "__main__":
    main()
