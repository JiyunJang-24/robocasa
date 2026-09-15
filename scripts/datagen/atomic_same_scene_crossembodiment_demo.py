#!/usr/bin/env python3
"""
Same-scene, same-object, robot-only-differs comparison: replay a v1.0 atomic
PickPlaceCounterToSink episode's exact raw actions on UR5eOmron, in the SAME
kitchen room (same layout_id/style_id -> same walls/counter/sink/style/
textures) with the manipulated object transplanted to the SAME absolute world
pose the source episode recorded. Only the robot embodiment differs.

This supersedes the earlier two-process atomic_source_rollout.py /
BARX/atomic_target_rollout.py pair, which used BARX's own frozen environment
for the target side. BARX's asset library only ships 10 layouts x 12 styles,
entirely disjoint from the layout/style ids v1.0 episodes actually use, so
that approach could never put the target in the SAME room as the source --
at best a same-relative-pose object in a *different*, unrelated-looking
kitchen. Building both sides directly in this conda environment's RoboCasa
(the same one the v1.0 dataset was collected with) sidesteps that entirely:
both sides can use the literal same layout_id/style_id, so the kitchen scene
is identical and only the robot (and therefore the controller/kinematics)
differs -- which is what actually needs to be held constant for "does the
same action sequence work on a different embodiment" to be a fair test.

The UR5e kinematic-workspace calibration (init_qpos, base backoff) reuses the
values already validated in generate_mimicgen_ur5e_dataset.py earlier this
session (UR5e's OSC-tracked kitchen-ready arm pose and the 0.10 m counter
backoff that avoids upper-arm/counter collisions for the 6-DOF arm). This is
the workspace-fit part of BARX's calibration; there is no grasp-pose/IK
retargeting here (no MimicGen), because the whole point is to test the RAW,
unmodified recorded action sequence open-loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path("/data1/local/robocasa")
SOURCE_TASK = "PickPlaceCounterToSink"
CAMERA = "robot0_agentview_center"
IMAGE_SIZE = 480
OUTPUT_ROOT = Path("/data1/local/BARX/datasets/cross_embodiment/atomic_action_replay")


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


def build_source(mg_gen, episode: int, work_dir: Path):
    source_dir = mg_gen.latest_lerobot_dir(
        ROOT / "datasets" / "v1.0", "pretrain", SOURCE_TASK
    )
    build_args = SimpleNamespace(
        task=SOURCE_TASK,
        split="pretrain",
        source_root=ROOT / "datasets" / "v1.0",
        source_lerobot_dir=None,
        work_dir=work_dir,
        source_n=episode + 1,
        force_source_raw=True,
    )
    raw_path = mg_gen.build_source_raw_hdf5(build_args, source_dir)

    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )

    with h5py.File(raw_path, "r") as f:
        demo = f[f"data/demo_{episode}"]
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
        frame = env.render(
            mode="rgb_array", height=IMAGE_SIZE, width=IMAGE_SIZE, camera_name=CAMERA
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
    env.env.close()

    return {
        "env_meta": env_meta,
        "actions": actions[:n],
        "ep_meta": ep_meta,
        "frames": frames,
        "obj_pos0": obj_pos0,
        "obj_quat0": obj_quat0,
        "success": success,
        "num_frames": n,
    }


def build_target_env_meta(
    source_env_meta: dict, layout_id: int, style_id: int, obj_group: str
) -> dict:
    import copy
    from robosuite.controllers import load_composite_controller_config

    target_meta = copy.deepcopy(source_env_meta)
    kwargs = target_meta["env_kwargs"]
    kwargs["robots"] = "UR5eOmron"
    kwargs["layout_ids"] = None
    kwargs["style_ids"] = None
    kwargs["layout_and_style_ids"] = [[layout_id, style_id]]
    kwargs["obj_groups"] = obj_group
    # NOT None: robosuite's gripper_types=None means "no gripper" (NullGripper,
    # action_dim short by 1) -- "default" resolves to UR5eOmron's own gripper.
    kwargs["gripper_types"] = "default"
    # Passing no controller_configs at all makes robosuite fall back to a
    # bare-bones composite config with no gripper sub-controller (NullGripper,
    # action_dim short by 1) -- load UR5e's own default explicitly instead,
    # same as generate_mimicgen_ur5e_dataset.py's target_controller_config().
    kwargs["controller_configs"] = load_composite_controller_config(robot="UR5eOmron")
    return target_meta


def free_joint_qpos_addr(sim, body_id: int) -> int:
    model = sim.model
    joint_id = model.body_jntadr[body_id]
    if joint_id < 0 or model.jnt_type[joint_id] != 0:
        raise RuntimeError("Object body has no free joint; cannot transplant its pose.")
    return int(model.jnt_qposadr[joint_id])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--base-backoff", type=float, default=0.10)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "datasets" / "mg_runs" / "atomic_demo_work",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    mg_gen = load_gen_module()

    print(f"[source] building episode {args.episode}...")
    source = build_source(mg_gen, args.episode, args.work_dir)
    ep_meta = source["ep_meta"]
    layout_id, style_id = ep_meta["layout_id"], ep_meta["style_id"]
    obj_group = ep_meta["object_cfgs"][0]["info"]["cat"]
    print(
        f"[source] layout={layout_id} style={style_id} obj_group={obj_group!r} "
        f"lang={ep_meta['lang']!r} success={source['success']}"
    )

    ur5e_args = SimpleNamespace(
        target_robot="UR5eOmron",
        base_backoff=args.base_backoff,
        no_init_qpos_patch=False,
    )
    mg_gen.install_base_backoff(ur5e_args)
    mg_gen.install_init_qpos_patch(ur5e_args)

    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils

    target_env_meta = build_target_env_meta(
        source["env_meta"], layout_id, style_id, obj_group
    )
    target_env = EnvUtils.create_env_from_metadata(
        env_meta=target_env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    target_env.reset()

    body_id = target_env.env.obj_body_id["obj"]
    addr = free_joint_qpos_addr(target_env.env.sim, body_id)
    target_env.env.sim.data.qpos[addr : addr + 3] = source["obj_pos0"]
    target_env.env.sim.data.qpos[addr + 3 : addr + 7] = source["obj_quat0"]
    target_env.env.sim.forward()
    obj_start = target_env.env.sim.data.body_xpos[body_id].copy()
    print(
        f"[target] obj transplanted at {np.round(obj_start, 3)} "
        f"(source: {np.round(source['obj_pos0'], 3)})"
    )

    target_frames = [
        label_frame(
            target_env.render(
                mode="rgb_array",
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                camera_name=CAMERA,
            ),
            [
                "UR5eOmron (same room/object, BARX kinematic calibration)",
                f"open-loop replay of Panda's raw actions  step 000/{len(source['actions'])}",
            ],
        )
    ]
    for t, action in enumerate(source["actions"]):
        target_env.step(action)
        target_frames.append(
            label_frame(
                target_env.render(
                    mode="rgb_array",
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    camera_name=CAMERA,
                ),
                [
                    "UR5eOmron (same room/object, BARX kinematic calibration)",
                    f"open-loop replay of Panda's raw actions  step {t + 1:03d}/{len(source['actions'])}",
                ],
            )
        )
    target_success = bool(target_env.env._check_success())
    obj_final = target_env.env.sim.data.body_xpos[body_id].copy()
    target_env.env.close()

    displacement = float(np.linalg.norm(obj_final - obj_start))
    print(f"[target] obj displacement over rollout: {displacement:.3f} m")
    print(f"[target] UR5eOmron open-loop replay success: {target_success}")

    output = (
        args.output
        or OUTPUT_ROOT / f"{SOURCE_TASK}_same_scene_panda_vs_ur5e_ep{args.episode}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(source["frames"]), len(target_frames))
    combined = [
        np.concatenate([source["frames"][i], target_frames[i]], axis=1)
        for i in range(n)
    ]
    imageio.mimsave(output, combined, fps=20, macro_block_size=1)
    print(f"[video] wrote {len(combined)} frames to {output}")

    report = {
        "source_task": SOURCE_TASK,
        "episode": args.episode,
        "layout_id": layout_id,
        "style_id": style_id,
        "obj_group": obj_group,
        "lang": ep_meta["lang"],
        "num_actions": int(len(source["actions"])),
        "source_success_recheck": source["success"],
        "target_ur5e_openloop_success": target_success,
        "obj_pos0": source["obj_pos0"].tolist(),
        "obj_transplanted_pos": obj_start.tolist(),
        "obj_final_pos": obj_final.tolist(),
        "obj_displacement_m": displacement,
        "video": str(output),
    }
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] wrote {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
