#!/usr/bin/env python3
"""
Render MimicGen-generated RoboCasa demos into the normalized BARX HDF5 layout
consumed by BARX/dataset/rlds/robocasa_x_dataset_builder.py:

    data/<demo>/actions                              (T, 12) float64
    data/<demo>/obs/agentview_rgb                    (T, 180, 320, 3) uint8
    data/<demo>/obs/ee_states                        (T, 6)  eef pos + euler(xyz)
    data/<demo>/obs/gripper_states                   (T, 2)  first two gripper qpos
    data/<demo>/aux_info/eef_normalized_image_pts    (T, 2)  eef pixel / (W, H)
    data/<demo>/aux_info/bboxes_2d/<obj>             (T, 4)  pixel [x0, y0, x1, y1]
    attrs: ep_meta (with "lang"), model_file, num_samples

Images are re-rendered by replaying the stored MuJoCo states with the
robot0_agentview_center camera at 320x180 (the BARX release resolution).

Usage:
    python render_barx_normalized_hdf5.py --output out.hdf5 \
        --inputs run1/demo.hdf5 run2/demo.hdf5 [--obj-names obj container]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/data1/local/robocasa")
for path in (str(ROOT), str(ROOT.parent / "robosuite"), str(ROOT / "robomimic")):
    if path not in sys.path:
        sys.path.insert(0, path)

CAMERA = "robot0_agentview_center"
WIDTH, HEIGHT = 320, 180


def euler_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    return T.mat2euler(T.quat2mat(quat))


def make_env(env_meta: dict):
    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
    )


def camera_transform(sim):
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    return get_camera_transform_matrix(sim, CAMERA, HEIGHT, WIDTH)


def project(points: np.ndarray, world_to_pixel: np.ndarray) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) pixel (x, y) coordinates."""
    from robosuite.utils.camera_utils import project_points_from_world_to_camera

    pts = project_points_from_world_to_camera(
        points=points,
        world_to_camera_transform=world_to_pixel,
        camera_height=HEIGHT,
        camera_width=WIDTH,
    )
    # robosuite returns (row, col) = (y, x)
    return pts[..., ::-1].astype(np.float64)


def eef_world_pos(env) -> np.ndarray:
    robot = env.env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    return env.env.sim.data.site_xpos[int(site_id)].copy()


def object_bbox_pixels(env, body_id: int, world_to_pixel: np.ndarray) -> np.ndarray:
    """Conservative 2-D bbox from per-geom bounding spheres of one body."""
    sim = env.env.sim
    model = sim.model
    corners = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        center = sim.data.geom_xpos[geom_id]
        radius = model.geom_rbound[geom_id]
        for dx in (-radius, radius):
            for dy in (-radius, radius):
                for dz in (-radius, radius):
                    corners.append(center + np.array([dx, dy, dz]))
    if not corners:
        return np.array([0.0, 0.0, 0.0, 0.0])
    pix = project(np.asarray(corners), world_to_pixel)
    x0, y0 = pix.min(axis=0)
    x1, y1 = pix.max(axis=0)
    return np.array(
        [
            np.clip(x0, 0, WIDTH - 1),
            np.clip(y0, 0, HEIGHT - 1),
            np.clip(x1, 0, WIDTH - 1),
            np.clip(y1, 0, HEIGHT - 1),
        ]
    )


def gripper_qpos_indexes(env) -> list[int]:
    sim = env.env.sim
    idxs = []
    for joint_id in range(sim.model.njnt):
        name = sim.model.joint_id2name(joint_id) or ""
        if name.startswith("gripper0"):
            idxs.append(int(sim.model.jnt_qposadr[joint_id]))
    return idxs[:2] if idxs else []


def eef_site_pose(env):
    import robosuite.utils.transform_utils as T

    robot = env.env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    sim = env.env.sim
    pos = sim.data.site_xpos[int(site_id)].copy()
    mat = sim.data.site_xmat[int(site_id)].reshape(3, 3).copy()
    return pos, T.mat2euler(mat)


def render_demo(env, demo, obj_names: list[str]):
    states = np.asarray(demo["states"])
    actions = np.asarray(demo["actions"], dtype=np.float64)
    # MimicGen outputs store low-dim obs; source-converted human raw files do
    # not, so fall back to reading the poses live from the replayed sim.
    has_lowdim = "obs" in demo and "robot0_eef_pos" in demo["obs"]
    if has_lowdim:
        eef_pos = np.asarray(demo["obs/robot0_eef_pos"], dtype=np.float64)
        quat_key = (
            "obs/robot0_eef_quat_site"
            if "robot0_eef_quat_site" in demo["obs"]
            else "obs/robot0_eef_quat"
        )
        eef_quat = np.asarray(demo[quat_key], dtype=np.float64)
        gripper_qpos = np.asarray(demo["obs/robot0_gripper_qpos"], dtype=np.float64)
    else:
        eef_pos = eef_quat = gripper_qpos = None

    n = min(len(states), len(actions))
    model_file = demo.attrs.get("model_file", "")
    ep_meta = demo.attrs.get("ep_meta", "")

    reset_state = {"states": states[0]}
    if model_file:
        reset_state["model"] = model_file
        if ep_meta:
            reset_state["ep_meta"] = ep_meta
    env.reset_to(reset_state)

    world_to_pixel = camera_transform(env.env.sim)
    body_ids = {}
    for name in obj_names:
        body_id = getattr(env.env, "obj_body_id", {}).get(name)
        if body_id is not None:
            body_ids[name] = body_id

    images = np.empty((n, HEIGHT, WIDTH, 3), dtype=np.uint8)
    ee_states = np.empty((n, 6), dtype=np.float64)
    gripper_states = np.empty((n, 2), dtype=np.float64)
    ee_pts_norm = np.empty((n, 2), dtype=np.float64)
    bboxes = {name: np.empty((n, 4), dtype=np.float64) for name in body_ids}
    gripper_idxs = None if has_lowdim else gripper_qpos_indexes(env)

    for t in range(n):
        env.reset_to({"states": states[t]})
        images[t] = env.render(
            mode="rgb_array", height=HEIGHT, width=WIDTH, camera_name=CAMERA
        )
        if has_lowdim:
            ee_states[t, :3] = eef_pos[t]
            ee_states[t, 3:] = euler_from_quat_xyzw(eef_quat[t])
            gripper_states[t] = gripper_qpos[t, :2]
        else:
            pos, euler = eef_site_pose(env)
            ee_states[t, :3] = pos
            ee_states[t, 3:] = euler
            qpos = [env.env.sim.data.qpos[i] for i in gripper_idxs]
            gripper_states[t] = (qpos + [0.0, 0.0])[:2]
        pix = project(eef_world_pos(env)[None], world_to_pixel)[0]
        ee_pts_norm[t] = np.clip(pix / np.array([WIDTH, HEIGHT]), 0.0, 1.0)
        for name, body_id in body_ids.items():
            bboxes[name][t] = object_bbox_pixels(env, body_id, world_to_pixel)

    return {
        "actions": actions[:n],
        "agentview_rgb": images,
        "ee_states": ee_states,
        "gripper_states": gripper_states,
        "eef_normalized_image_pts": ee_pts_norm,
        "bboxes_2d": bboxes,
        "model_file": model_file,
        "ep_meta": ep_meta,
        "num_samples": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obj-names", nargs="*", default=["obj"])
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Stop after writing this many demos across all inputs",
    )
    args = parser.parse_args()

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    # Kinova3Omron only exists via the generator wrapper's registration.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "barx_gen", ROOT / "generate_barx_cross_embodiment_dataset.py"
    )
    barx_gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(barx_gen)
    barx_gen.register_kinova3_omron()

    env = None
    env_args_json = None
    demo_index = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        data_grp = out.create_group("data")
        for input_path in args.inputs:
            with h5py.File(input_path, "r") as source:
                if env is None:
                    env_args_json = source["data"].attrs["env_args"]
                    env_meta = json.loads(env_args_json)
                    env = make_env(env_meta)
                    data_grp.attrs["env_args"] = env_args_json
                    data_grp.attrs["env"] = env_meta["env_name"]
                for demo_key in sorted(
                    source["data"].keys(), key=lambda k: int(k.split("_")[-1])
                ):
                    if args.max_demos is not None and demo_index >= args.max_demos:
                        break
                    result = render_demo(
                        env, source[f"data/{demo_key}"], args.obj_names
                    )
                    grp = data_grp.create_group(f"demo_{demo_index}")
                    grp.attrs["ep_meta"] = result["ep_meta"]
                    grp.attrs["model_file"] = result["model_file"]
                    grp.attrs["num_samples"] = result["num_samples"]
                    grp.attrs["source_file"] = str(input_path)
                    grp.attrs["source_demo"] = demo_key
                    grp.create_dataset("actions", data=result["actions"])
                    obs = grp.create_group("obs")
                    obs.create_dataset(
                        "agentview_rgb",
                        data=result["agentview_rgb"],
                        compression="gzip",
                        compression_opts=4,
                    )
                    obs.create_dataset("ee_states", data=result["ee_states"])
                    obs.create_dataset("gripper_states", data=result["gripper_states"])
                    aux = grp.create_group("aux_info")
                    aux.create_dataset(
                        "eef_normalized_image_pts",
                        data=result["eef_normalized_image_pts"],
                    )
                    bbox_grp = aux.create_group("bboxes_2d")
                    for name, arr in result["bboxes_2d"].items():
                        bbox_grp.create_dataset(name, data=arr)
                    print(
                        f"[demo_{demo_index}] {input_path.name}:{demo_key} "
                        f"T={result['num_samples']} bboxes={list(result['bboxes_2d'])}"
                    )
                    demo_index += 1
        data_grp.attrs["total"] = demo_index
    if env is not None:
        env.env.close()
    print(f"wrote {args.output} ({demo_index} demos)")


if __name__ == "__main__":
    main()
