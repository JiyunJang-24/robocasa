#!/usr/bin/env python3
"""
Instrument one recorded (failed) MimicGen attempt frame-by-frame: replay its
states, and at every timestep measure the actual distance between the
gripper's pad midpoint ("grasp center") and the manipulated object's center,
alongside the recorded gripper action (open/close). This answers directly
whether the object ever ends up centered between the pads at the moment the
gripper closes -- i.e. whether the approach motion itself is landing where it
needs to, independent of any theorizing about grasp geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def gripper_pad_geom_ids(sim):
    ids = []
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ""
        if (
            name.startswith("gripper0")
            and "pad" in name.lower()
            and "visual" not in name
        ):
            ids.append((gid, name))
    return ids


def grasp_center_world(sim):
    pads = [sim.data.geom_xpos[gid].copy() for gid, _ in gripper_pad_geom_ids(sim)]
    if not pads:
        return None
    return np.mean(pads, axis=0)


def pad_side_positions(sim):
    groups = {}
    for gid, name in gripper_pad_geom_ids(sim):
        side = "l" if ("left" in name or "finger2" in name) else "r"
        groups.setdefault(side, []).append(sim.data.geom_xpos[gid].copy())
    return {k: np.mean(v, axis=0) for k, v in groups.items() if v}


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

    n = min(len(states), len(actions))
    rows = []
    for t in range(n):
        reset_state = {"states": states[t]}
        if t == 0:
            reset_state["model"] = model_file
            if ep_meta:
                reset_state["ep_meta"] = ep_meta
        env.reset_to(reset_state)
        sim = env.env.sim
        obj_id = env.env.obj_body_id[args.obj_name]
        obj_pos = sim.data.body_xpos[obj_id].copy()
        gcenter = grasp_center_world(sim)
        pads = pad_side_positions(sim)
        dist = (
            float(np.linalg.norm(gcenter - obj_pos))
            if gcenter is not None
            else float("nan")
        )
        pad_gap = (
            float(np.linalg.norm(pads["l"] - pads["r"]))
            if "l" in pads and "r" in pads
            else float("nan")
        )
        gripper_action = float(actions[t, 6])

        robot = env.env.robots[0]
        site_id = robot.eef_site_id
        if isinstance(site_id, dict):
            site_id = site_id[robot.arms[0]]
        eef_pos = sim.data.site_xpos[int(site_id)].copy()
        eef_dist = float(np.linalg.norm(eef_pos - obj_pos))

        rows.append((t, dist, pad_gap, gripper_action, eef_dist))

    env.env.close()

    dists = np.array([r[1] for r in rows])
    eef_dists = np.array([r[4] for r in rows])
    grip_actions = np.array([r[3] for r in rows])
    closing_steps = np.where(np.diff((grip_actions > 0).astype(int)) == 1)[0] + 1
    min_dist_step = int(np.argmin(dists))
    min_eef_step = int(np.argmin(eef_dists))

    print(f"episode length: {n}")
    print(
        f"min grasp-center<->object distance: {dists[min_dist_step]*100:.2f} cm at step {min_dist_step}"
    )
    print(
        f"min EEF-SITE<->object distance: {eef_dists[min_eef_step]*100:.2f} cm at step {min_eef_step}"
    )
    print(f"gripper action range: [{grip_actions.min():.2f}, {grip_actions.max():.2f}]")
    print(
        f"gripper close-command transitions (open->close) at steps: {closing_steps.tolist()}"
    )
    for step in closing_steps:
        window = slice(max(0, step - 3), min(n, step + 4))
        print(
            f"  around close@{step}: dist(cm)={np.round(dists[window] * 100, 2)}, "
            f"eef_dist(cm)={np.round(eef_dists[window] * 100, 2)}, "
            f"pad_gap(cm)={np.round([rows[i][2] * 100 for i in range(window.start, window.stop)], 2)}"
        )
    print(
        "\ndecimated trace every 5 steps (t, dist_cm, eef_dist_cm, pad_gap_cm, gripper_action):"
    )
    interesting = (
        set(range(0, n, 5))
        | set(closing_steps.tolist())
        | {min_dist_step, min_eef_step}
    )
    for t, dist, gap, ga, eefd in rows:
        if t not in interesting:
            continue
        marker = (
            " <-- min dist"
            if t == min_dist_step
            else (" <-- close cmd" if t in closing_steps else "")
        )
        print(
            f"  t={t:3d}  dist={dist*100:6.2f}cm  eef_dist={eefd*100:6.2f}cm  pad_gap={gap*100:6.2f}cm  grip_action={ga:5.2f}{marker}"
        )


if __name__ == "__main__":
    main()
