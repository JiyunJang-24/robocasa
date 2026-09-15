#!/usr/bin/env python3
"""
Check whether a recorded MimicGen attempt's ACTIONS are actually driving the
robot the way they're supposed to: for each step, compare the commanded
position/rotation delta (from the recorded action) against the ACTUAL
resulting end-effector displacement (from the recorded states, which reflect
what genuinely happened when that action was applied during generation), and
report arm joint values against their limits at the point of closest
approach. This distinguishes "the controller is being told to go somewhere
sensible but physically can't get there" (saturation/limits/tracking lag)
from "the commanded target itself is wrong" (actions don't point toward the
object at all).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import robosuite.utils.transform_utils as T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--obj-name", default="obj")
    args = parser.parse_args()

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )

    with h5py.File(args.hdf5, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        demo = f[f"data/{args.demo}"]
        states = np.asarray(demo["states"])
        actions = np.asarray(demo["actions"], dtype=np.float64)
        model_file = demo.attrs["model_file"]
        ep_meta = demo.attrs.get("ep_meta", None)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False
    )
    n = min(len(states), len(actions)) - 1

    eef_positions = []
    base_oris = []
    obj_positions = []
    joint_positions = []
    robot = None
    for t in range(n + 1):
        reset_state = {"states": states[t]}
        if t == 0:
            reset_state["model"] = model_file
            if ep_meta:
                reset_state["ep_meta"] = ep_meta
        env.reset_to(reset_state)
        if robot is None:
            robot = env.env.robots[0]
        sim = env.env.sim
        site_id = robot.eef_site_id
        if isinstance(site_id, dict):
            site_id = site_id[robot.arms[0]]
        eef_positions.append(sim.data.site_xpos[int(site_id)].copy())
        base_body_id = sim.model.body_name2id("mobilebase0_support")
        base_oris.append(sim.data.body_xmat[base_body_id].reshape(3, 3).copy())
        obj_positions.append(
            sim.data.body_xpos[env.env.obj_body_id[args.obj_name]].copy()
        )
        joint_positions.append(
            np.array([sim.data.qpos[i] for i in robot._ref_arm_joint_pos_indexes])
        )

    eef_positions = np.array(eef_positions)
    obj_positions = np.array(obj_positions)
    joint_positions = np.array(joint_positions)
    dists = np.linalg.norm(eef_positions - obj_positions, axis=1)
    min_step = int(np.argmin(dists))

    # OSC "default" controller: output_max for arm is +-0.05 m/step, +-0.5 rad/step
    OUTPUT_MAX_POS = 0.05
    print(f"episode length: {n + 1}")
    print(
        f"min eef<->object distance: {dists[min_step]*100:.2f} cm at step {min_step}\n"
    )

    print(
        "Per-step: commanded action pos-delta (world frame, scaled by output_max) "
        "vs ACTUAL eef displacement, and error-reducing-ness (dot product with "
        "direction-to-object > 0 means the action pointed toward the object):"
    )
    window = range(max(0, min_step - 15), min(n, min_step + 5))
    for t in window:
        action_pos_local = (
            actions[t, 0:3] * OUTPUT_MAX_POS
        )  # normalized -> meters, base frame
        action_pos_world = base_oris[t] @ action_pos_local
        actual_delta = eef_positions[t + 1] - eef_positions[t]
        to_obj = obj_positions[t] - eef_positions[t]
        to_obj_dir = to_obj / (np.linalg.norm(to_obj) + 1e-9)
        commanded_alignment = float(
            np.dot(
                action_pos_world / (np.linalg.norm(action_pos_world) + 1e-9), to_obj_dir
            )
        )
        actual_alignment = float(
            np.dot(actual_delta / (np.linalg.norm(actual_delta) + 1e-9), to_obj_dir)
        )
        tracking_ratio = float(
            np.linalg.norm(actual_delta) / (np.linalg.norm(action_pos_world) + 1e-9)
        )
        print(
            f"  t={t:3d} dist={dists[t]*100:6.2f}cm "
            f"cmd|={np.linalg.norm(action_pos_world)*100:5.2f}cm "
            f"actual|={np.linalg.norm(actual_delta)*100:5.2f}cm "
            f"tracking_ratio={tracking_ratio:4.2f} "
            f"cmd_aligned_to_obj={commanded_alignment:+.2f} "
            f"actual_aligned_to_obj={actual_alignment:+.2f}"
        )

    print("\nArm joint positions vs limits at closest-approach step:")
    joint_names = robot.robot_arm_joints
    sim = env.env.sim
    for i, jn in enumerate(joint_names):
        jid = sim.model.joint_name2id(jn)
        lo, hi = sim.model.jnt_range[jid]
        val = joint_positions[min_step, i]
        near_limit = ""
        if hi > lo:
            frac = (val - lo) / (hi - lo)
            if frac < 0.05 or frac > 0.95:
                near_limit = "  <-- NEAR LIMIT"
        print(f"  {jn}: {val:.3f} rad  (range [{lo:.3f}, {hi:.3f}]){near_limit}")

    env.env.close()


if __name__ == "__main__":
    main()
