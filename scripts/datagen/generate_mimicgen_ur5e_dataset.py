#!/usr/bin/env python3
"""
Generate a RoboCasa atomic dataset with MimicGen on UR5eOmron.

The script is task-driven: pass a RoboCasa atomic task name and it will
1. find the source LeRobot dataset,
2. convert source states/actions to a raw robomimic HDF5 if needed,
3. prepare MimicGen datagen_info,
4. build a UR5eOmron MimicGen config from the task template,
5. run MimicGen generation and save generated dataset videos.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import pkgutil
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v2 as imageio
import numpy as np


ROOT = Path("/data1/local/robocasa")
MIMICGEN_ROOT = ROOT / "mimicgen"
ROBOMIMIC_ROOT = ROOT / "robomimic"
ROBOSUITE_ROOT = ROOT.parent / "robosuite"
DEFAULT_SOURCE_ROOT = ROOT / "datasets" / "v1.0"
DEFAULT_OUTPUT_ROOT = ROOT / "datasets" / "mimicgen_ur5e"

# --- source (PandaOmron) gripper constants, measured live via
# scratchpad/mg_measure_gripper_constants.py -------------------------------
# Axis (eef-site frame) along which the Panda finger pads separate. Robotiq85
# pads separate along site-y instead -> commanding raw Panda orientations makes
# the Robotiq straddle objects 90 deg off; we fold a Rz(twist) into every
# commanded pose so the pads land where the Panda's did.
PANDA_OPENING_AXIS = np.array([1.0, 0.0, 0.0])
# Fingertip-pad midpoint in the eef-site frame (z = approach axis). The
# Robotiq's midpoint sits 1.1 cm shallower, enough for knob grasps to pinch
# the rim and slip.
PANDA_GRASP_CENTER = np.array([0.0, 0.0, -0.0036])
# Panda pad friction (sliding / torsional / rolling); the stock Robotiq85 pads
# ship far slicker (0.8 / 0.005) and cannot beat knob-joint frictionloss.
PANDA_PAD_FRICTION = (2.0, 0.05, 1.0e-4)

# Kitchen-ready arm configuration for UR5eOmron, solved via DLS IK to match
# PandaOmron's initial base-relative EEF pose (scratchpad/mg_ur5e_init_qpos.py,
# residual 0.15 mm, zero arm-scene contacts). The stock robosuite UR5e
# init_qpos is a folded tabletop pose from which the kitchen OSC controller
# stalls with a persistent ~5 cm tracking lag.
UR5E_KITCHEN_INIT_QPOS = np.array([-0.9593, -2.1292, 1.2628, -0.7801, -1.3080, 2.1020])


def patch_python_paths():
    for path in (
        str(ROOT),
        str(ROBOSUITE_ROOT),
        str(MIMICGEN_ROOT),
        str(ROBOMIMIC_ROOT),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


def patch_huggingface_hub_compat():
    """
    The local robomimic imports diffusers, and this env currently has a newer
    huggingface_hub without legacy HfFolder / cached_download symbols.
    MimicGen data generation does not use them, so a small compatibility shim is
    enough to let robomimic import.
    """
    try:
        import huggingface_hub
    except Exception:
        return

    if not hasattr(huggingface_hub, "HfFolder"):

        class HfFolder:
            @staticmethod
            def get_token():
                return None

            @staticmethod
            def save_token(token):
                return None

            @staticmethod
            def delete_token():
                return None

        huggingface_hub.HfFolder = HfFolder

    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download


def patch_robomimic_algo_imports():
    """
    robomimic.utils.file_utils imports robomimic.algo for checkpoint loading.
    MimicGen dataset preparation / generation only needs dataset metadata, so
    avoid importing training stacks such as torchvision / diffusers in this env.
    """
    if "robomimic.algo" in sys.modules:
        return

    algo_mod = types.ModuleType("robomimic.algo")

    def algo_factory(*_args, **_kwargs):
        raise RuntimeError(
            "robomimic.algo_factory is not available in this data-generation wrapper"
        )

    class RolloutPolicy:
        pass

    algo_mod.algo_factory = algo_factory
    algo_mod.RolloutPolicy = RolloutPolicy
    sys.modules["robomimic.algo"] = algo_mod


def import_mimicgen_robocasa_configs():
    import mimicgen.configs.robocasa as robocasa_configs

    for module_info in pkgutil.walk_packages(
        robocasa_configs.__path__,
        prefix=f"{robocasa_configs.__name__}.",
    ):
        importlib.import_module(module_info.name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="AdjustToasterOvenTemperature")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-lerobot-dir", type=Path, default=None)
    parser.add_argument(
        "--work-dir", type=Path, default=ROOT / "datasets" / "mimicgen_work"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-robot", default="UR5eOmron")
    parser.add_argument("--target-gripper", default="default")
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--source-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument(
        "--video-camera", nargs="+", default=["robot0_agentview_center"]
    )
    parser.add_argument("--render-attempt-video", action="store_true")
    parser.add_argument(
        "--attempts-only",
        action="store_true",
        help=(
            "Run exactly --num-demos attempts instead of retrying until that many successes. "
            "Useful for smoke tests and for inspecting failed generated videos."
        ),
    )
    parser.add_argument("--force-source-raw", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # --- UR5e embodiment adaptation (all no-ops when the target is Panda) ---
    parser.add_argument(
        "--base-backoff",
        type=float,
        default=0.10,
        help="Spawn the target robot base this many meters further from the "
        "counter. The 6-DOF UR5e needs elbow room the 7-DOF Panda does "
        "not; 0.10 m removed all arm-counter collisions in prior tests.",
    )
    parser.add_argument(
        "--grasp-depth-extra",
        type=float,
        default=0.01,
        help="Extra approach depth (m) on top of the measured grasp-center "
        "alignment, compensating the Robotiq linkage's forward swing "
        "when closing.",
    )
    parser.add_argument(
        "--rot-gain",
        type=float,
        default=2.5,
        help="Gain on the normalized delta-rotation action (clipped to [-1,1] "
        "after scaling). The UR5e wrist tracks source rotations with "
        "2.6-7.7 deg of lag at gain 1.0; 2.5 reduced this to under 1.1 "
        "deg in prior eef-tracking tests with no position downside.",
    )
    parser.add_argument(
        "--osc-kp",
        type=float,
        default=300.0,
        help="Position/orientation kp for the target arm's OSC controller "
        "(kp_limits allow up to 300). The UR5e trails the per-step "
        "MimicGen waypoints by 3-8 cm at the source kp of 150, leaving "
        "doors/racks/knobs just short of their success thresholds.",
    )
    parser.add_argument(
        "--no-twist-align",
        action="store_true",
        help="Disable gripper opening-axis (90 deg) alignment",
    )
    parser.add_argument(
        "--no-pad-friction-boost",
        action="store_true",
        help="Do not raise target pad friction to Panda level",
    )
    parser.add_argument(
        "--no-init-qpos-patch",
        action="store_true",
        help="Keep the stock robosuite UR5e init_qpos",
    )
    parser.add_argument(
        "--fallback-video-n",
        type=int,
        default=10,
        help="Max demos per hdf5 to render in fallback videos",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# UR5e embodiment adaptation layer
# ---------------------------------------------------------------------------


def _gripper_pad_geom_ids(sim):
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


def _eef_site_pose(sim, robot):
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    sim.forward()
    return (
        sim.data.site_xpos[int(site_id)].copy(),
        sim.data.site_xmat[int(site_id)].reshape(3, 3).copy(),
    )


def _opening_axis_in_site_frame(sim, robot):
    _, sR = _eef_site_pose(sim, robot)
    groups = {}
    for gid, name in _gripper_pad_geom_ids(sim):
        side = "l" if ("left" in name or "finger2" in name) else "r"
        groups.setdefault(side, []).append(sim.data.geom_xpos[gid].copy())
    if len(groups) < 2:
        return np.array([0.0, 1.0, 0.0])
    d_site = sR.T @ (np.mean(groups["r"], axis=0) - np.mean(groups["l"], axis=0))
    d_site[2] = 0.0
    n = np.linalg.norm(d_site)
    return d_site / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def _grasp_center_in_site_frame(sim, robot):
    sp, sR = _eef_site_pose(sim, robot)
    pads = [sim.data.geom_xpos[gid].copy() for gid, _ in _gripper_pad_geom_ids(sim)]
    if not pads:
        return np.zeros(3)
    return sR.T @ (np.mean(pads, axis=0) - sp)


def _twist_matrix_to_panda_axis(sim, robot):
    """Rz aligning the live gripper's pad-opening axis with the Panda's,
    folded into [-90, 90] deg (parallel grippers are 180-deg symmetric)."""
    ax_tgt = _opening_axis_in_site_frame(sim, robot)
    ang = np.arctan2(ax_tgt[1], ax_tgt[0]) - np.arctan2(
        PANDA_OPENING_AXIS[1], PANDA_OPENING_AXIS[0]
    )
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def install_base_backoff(args):
    """Spawn the target robot base further from the counter via robocasa's
    per-robot-class offset table (offset[1] is along the fixture-facing axis,
    negative = away from the fixture; robocasa multiplies it by face_dir)."""
    if args.base_backoff == 0.0 or args.target_robot == "PandaOmron":
        return
    import robocasa.utils.env_utils as renv_utils

    offset = list(renv_utils._ROBOT_POS_OFFSETS.get(args.target_robot, [0.0, 0.0, 0.0]))
    offset[1] -= args.base_backoff
    renv_utils._ROBOT_POS_OFFSETS[args.target_robot] = offset
    print(f"[patch] base backoff {args.base_backoff} m for {args.target_robot}")


def install_init_qpos_patch(args):
    if args.no_init_qpos_patch or args.target_robot != "UR5eOmron":
        return
    from robosuite.models.robots.compositional import UR5eOmron

    UR5eOmron.init_qpos = property(lambda self: UR5E_KITCHEN_INIT_QPOS.copy())
    print(f"[patch] UR5eOmron kitchen init_qpos {UR5E_KITCHEN_INIT_QPOS}")


def install_pad_friction_boost(args):
    """Boost target gripper pad friction to Panda level after every reset
    (no-op when the pads already are Panda pads)."""
    if args.no_pad_friction_boost:
        return
    from robomimic.envs.env_robosuite import EnvRobosuite

    def _boost(env_wrapper):
        sim = env_wrapper.env.sim
        for gid, _ in _gripper_pad_geom_ids(sim):
            sim.model.geom_friction[gid] = np.asarray(PANDA_PAD_FRICTION)

    orig_reset = EnvRobosuite.reset
    orig_reset_to = EnvRobosuite.reset_to

    def reset(self, *a, **k):
        out = orig_reset(self, *a, **k)
        _boost(self)
        return out

    def reset_to(self, *a, **k):
        out = orig_reset_to(self, *a, **k)
        _boost(self)
        return out

    EnvRobosuite.reset = reset
    EnvRobosuite.reset_to = reset_to
    print("[patch] gripper pad friction boost installed")


def install_grasp_pose_adaptation(args):
    """Transform every commanded EEF pose so the target gripper's pads land
    where the source Panda's pads did: opening-axis twist alignment plus
    grasp-center depth alignment. Both are measured live from the target env,
    so this is an exact no-op for a Panda target."""
    from mimicgen.env_interfaces.robosuite import RobosuiteInterface

    cache = {}

    def _adaptation(interface):
        key = id(interface.env)
        if key not in cache:
            sim = interface.env.sim
            robot = interface.env.robots[0]
            if args.no_twist_align:
                twist = np.eye(3)
            else:
                twist = _twist_matrix_to_panda_axis(sim, robot)
            offset = PANDA_GRASP_CENTER - _grasp_center_in_site_frame(sim, robot)
            if np.linalg.norm(offset) > 2e-3:
                # different gripper model: also compensate the closing swing
                offset = offset + np.array([0.0, 0.0, args.grasp_depth_extra])
            else:
                # same (Panda) gripper: exact no-op
                offset = np.zeros(3)
            cache[key] = (twist, offset)
            print(
                f"[patch] grasp adaptation: twist=\n{np.round(twist, 3)} offset={np.round(offset, 4)}"
            )
        return cache[key]

    orig = RobosuiteInterface.target_pose_to_action

    def target_pose_to_action(self, target_pose, relative=True):
        twist, offset = _adaptation(self)
        pose = np.array(target_pose)
        rot = pose[:3, :3]
        pose[:3, 3] = pose[:3, 3] + rot @ offset
        pose[:3, :3] = rot @ twist
        action = orig(self, pose, relative=relative)
        if relative and args.rot_gain != 1.0:
            action = np.array(action)
            action[3:6] = np.clip(action[3:6] * args.rot_gain, -1.0, 1.0)
        return action

    RobosuiteInterface.target_pose_to_action = target_pose_to_action
    print("[patch] grasp pose adaptation installed")


def install_embodiment_patches(args):
    install_base_backoff(args)
    install_init_qpos_patch(args)
    install_pad_friction_boost(args)
    install_grasp_pose_adaptation(args)


def latest_lerobot_dir(source_root: Path, split: str, task: str) -> Path:
    candidates = sorted(
        (source_root / split / "atomic" / task).glob("*/lerobot"), reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"No LeRobot dataset found for task={task} under {source_root / split / 'atomic' / task}"
        )
    return candidates[0]


def find_template(task: str) -> Path:
    candidates = sorted(
        (MIMICGEN_ROOT / "mimicgen" / "exps" / "templates" / "robocasa").glob(
            f"**/{task}.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No MimicGen template JSON found for task={task}")
    return candidates[0]


def episode_paths(source_lerobot_dir: Path, source_n: int | None):
    paths = sorted(source_lerobot_dir.glob("data/chunk-*/episode_*.parquet"))
    if source_n is not None:
        paths = paths[:source_n]
    if not paths:
        raise FileNotFoundError(f"No parquet episodes found in {source_lerobot_dir}")
    return paths


def load_modality(source_lerobot_dir: Path) -> dict:
    with open(source_lerobot_dir / "meta" / "modality.json", "r") as f:
        return json.load(f)


ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


def reorder_lerobot_actions_to_hdf5(actions: np.ndarray, modality: dict) -> np.ndarray:
    raw_dim = max(end for _, end in ACTION_KEY_ORDERING_HDF5.values())
    raw_actions = np.zeros((actions.shape[0], raw_dim), dtype=np.float64)
    for key, (dst_start, dst_end) in ACTION_KEY_ORDERING_HDF5.items():
        if key not in modality["action"]:
            continue
        src_start = modality["action"][key]["start"]
        src_end = modality["action"][key]["end"]
        raw_actions[:, dst_start:dst_end] = actions[:, src_start:src_end]
    return raw_actions


def load_episode_payload(source_lerobot_dir: Path, episode_num: int):
    episode_dir = source_lerobot_dir / "extras" / f"episode_{episode_num:06d}"
    states = np.load(episode_dir / "states.npz", allow_pickle=True)["states"]
    with open(episode_dir / "ep_meta.json", "r") as f:
        ep_meta = json.load(f)
    model_file = ""
    model_path = episode_dir / "model.xml.gz"
    if model_path.exists():
        with gzip.open(model_path, "rt") as f:
            model_file = f.read()
    return states, ep_meta, model_file


def source_raw_path(args, source_lerobot_dir: Path) -> Path:
    source_tag = source_lerobot_dir.parent.name
    n_tag = "all" if args.source_n is None else str(args.source_n)
    return (
        args.work_dir
        / "source_raw"
        / args.split
        / args.task
        / source_tag
        / f"source_n{n_tag}.hdf5"
    )


def build_source_raw_hdf5(args, source_lerobot_dir: Path) -> Path:
    import pandas as pd
    import robocasa
    import robosuite
    import mujoco

    raw_path = source_raw_path(args, source_lerobot_dir)
    if raw_path.exists() and not args.force_source_raw:
        print(f"[source-raw] reuse {raw_path}")
        return raw_path

    with open(source_lerobot_dir / "extras" / "dataset_meta.json", "r") as f:
        source_env_meta = json.load(f)["env_args"]
    modality = load_modality(source_lerobot_dir)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["date"] = "generated-from-lerobot"
        data_grp.attrs["time"] = "generated-from-lerobot"
        data_grp.attrs["robocasa_version"] = robocasa.__version__
        data_grp.attrs["robosuite_version"] = robosuite.__version__
        data_grp.attrs["mujoco_version"] = mujoco.__version__
        data_grp.attrs["env"] = source_env_meta["env_name"]
        data_grp.attrs["env_args"] = json.dumps(source_env_meta, indent=4)
        data_grp.attrs["env_info"] = json.dumps(source_env_meta["env_kwargs"], indent=4)

        total = 0
        for demo_idx, parquet_path in enumerate(
            episode_paths(source_lerobot_dir, args.source_n)
        ):
            episode_num = int(parquet_path.stem.split("_")[-1])
            episode_df = pd.read_parquet(parquet_path)
            states, ep_meta, model_file = load_episode_payload(
                source_lerobot_dir, episode_num
            )
            actions = reorder_lerobot_actions_to_hdf5(
                np.stack(episode_df["action"].to_numpy()),
                modality=modality,
            )
            n = min(len(states), len(actions))
            demo_grp = data_grp.create_group(f"demo_{demo_idx}")
            demo_grp.attrs["model_file"] = model_file
            demo_grp.attrs["ep_meta"] = json.dumps(ep_meta, indent=4)
            demo_grp.attrs["num_samples"] = int(n)
            demo_grp.attrs["source_episode"] = int(episode_num)
            demo_grp.create_dataset(
                "states", data=np.asarray(states[:n], dtype=np.float64)
            )
            demo_grp.create_dataset(
                "actions", data=np.asarray(actions[:n], dtype=np.float64)
            )
            total += n
        data_grp.attrs["total"] = int(total)

    print(f"[source-raw] wrote {raw_path}")
    return raw_path


def has_datagen_info(raw_path: Path) -> bool:
    with h5py.File(raw_path, "r") as f:
        demos = sorted(f["data"].keys())
        return bool(demos) and ("datagen_info" in f[f"data/{demos[0]}"])


def prepare_source_for_mimicgen(args, raw_path: Path):
    if has_datagen_info(raw_path) and not args.force_prepare:
        print(f"[prepare] datagen_info already exists in {raw_path}")
        return

    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    print(f"[prepare] extracting datagen_info for {raw_path}")
    prepare_src_dataset(
        dataset_path=str(raw_path),
        env_interface_name=f"MG_{args.task}",
        env_interface_type="robosuite",
        filter_key=None,
        n=args.source_n,
        output_path=None,
    )


def target_controller_config(target_robot: str, osc_kp: float | None = None):
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(robot=target_robot)
    if osc_kp is not None and target_robot != "PandaOmron":
        right = cfg.get("body_parts", {}).get("right", {})
        if "kp" in right:
            right["kp"] = osc_kp
            kp_limits = right.get("kp_limits")
            if kp_limits is not None:
                right["kp"] = float(np.clip(osc_kp, kp_limits[0], kp_limits[1]))
    return cfg


def build_mimicgen_config(args, source_raw: Path, template_path: Path) -> Path:
    cfg = json.load(open(template_path, "r"))
    cfg["name"] = args.task
    cfg["type"] = cfg.get("type", "robosuite")

    experiment = cfg.setdefault("experiment", {})
    experiment["name"] = f"{args.task}_{args.target_robot}"
    experiment.setdefault("source", {})
    experiment["source"]["dataset_path"] = str(source_raw)
    experiment["source"]["filter_key"] = None
    experiment["source"]["n"] = args.source_n
    experiment["source"]["start"] = None

    experiment.setdefault("generation", {})
    experiment["generation"]["path"] = str(
        args.output_root / args.target_robot / args.split / args.task
    )
    experiment["generation"]["num_trials"] = args.num_demos
    experiment["generation"]["guarantee"] = not args.attempts_only
    experiment["generation"]["keep_failed"] = True

    experiment.setdefault("task", {})
    experiment["task"]["name"] = args.task
    # Leave task.robot / task.gripper as None because MimicGen's helper only
    # accepts fixed-base robosuite robot names through those fields.
    experiment["task"]["robot"] = None
    experiment["task"]["gripper"] = None
    experiment["task"]["interface"] = f"MG_{args.task}"
    experiment["task"]["interface_type"] = "robosuite"
    task_kwargs = deepcopy(experiment["task"].get("kwargs", {}))
    task_kwargs.update(
        {
            "robots": args.target_robot,
            "gripper_types": args.target_gripper,
            "controller_configs": target_controller_config(
                args.target_robot, osc_kp=args.osc_kp
            ),
            "translucent_robot": False,
            # source datasets predate the robocasa split rename
            # ("train"/"test" -> "pretrain"/"target"); the stale value makes
            # object sampling raise ValueError on tasks that place objects.
            "obj_instance_split": "pretrain" if args.split == "pretrain" else "target",
        }
    )
    experiment["task"]["kwargs"] = task_kwargs

    experiment["render_video"] = True
    experiment["num_demo_to_render"] = min(args.num_demos, 10)
    experiment["num_fail_demo_to_render"] = 2
    experiment["seed"] = args.seed

    cfg.setdefault("obs", {})
    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = []
    cfg["obs"]["camera_height"] = 84
    cfg["obs"]["camera_width"] = 84

    config_path = (
        args.work_dir / "configs" / args.split / args.task / f"{args.target_robot}.json"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"[config] wrote {config_path}")
    return config_path


def run_mimicgen(args, config_path: Path):
    import mimicgen.scripts.generate_dataset_multicore as mg_generate

    gen_args = SimpleNamespace(
        config=str(config_path),
        debug=args.debug,
        auto_remove_exp=True,
        render=False,
        video_path=(
            str(
                args.output_root
                / args.target_robot
                / args.split
                / args.task
                / "attempts.mp4"
            )
            if args.render_attempt_video
            else None
        ),
        video_skip=args.video_skip,
        render_image_names=args.video_camera,
        pause_subtask=False,
        source=None,
        task_name=None,
        folder=None,
        num_demos=None,
        seed=None,
        num_procs=args.num_procs,
    )
    # generate_dataset_multiprocess references the module-level args.
    mg_generate.args = gen_args
    mg_generate.main(gen_args)


def latest_generation_dir(args) -> Path | None:
    base = (
        args.output_root
        / args.target_robot
        / args.split
        / args.task
        / f"{args.task}_{args.target_robot}"
    )
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None


def hdf5_has_demos(dataset_path: Path) -> bool:
    if not dataset_path.exists():
        return False
    with h5py.File(dataset_path, "r") as f:
        return bool(f["data"].keys()) and int(f["data"].attrs.get("total", 0)) > 0


def render_hdf5_states_videos(
    dataset_path: Path,
    video_dir: Path,
    tag: str,
    camera_names,
    video_skip: int,
    max_demos: int,
):
    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}}
    )

    with h5py.File(dataset_path, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for demo_key in demo_keys[:max_demos]:
            with h5py.File(dataset_path, "r") as f:
                demo = f[f"data/{demo_key}"]
                states = np.asarray(demo["states"])
                model_file = demo.attrs.get("model_file", None)
                ep_meta = demo.attrs.get("ep_meta", None)
            video_path = video_dir / f"{tag}_{demo_key}.mp4"
            writer = imageio.get_writer(video_path, fps=20)
            try:
                for i, state in enumerate(states):
                    if i % video_skip != 0 and i != len(states) - 1:
                        continue
                    reset_state = {"states": state}
                    if i == 0 and model_file:
                        reset_state["model"] = model_file
                        if ep_meta:
                            reset_state["ep_meta"] = ep_meta
                    env.reset_to(reset_state)
                    frames = []
                    for camera_name in camera_names:
                        frames.append(
                            env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=camera_name,
                            )
                        )
                    writer.append_data(
                        frames[0]
                        if len(frames) == 1
                        else np.concatenate(frames, axis=1)
                    )
            finally:
                writer.close()
            written.append(video_path)
            print(f"[video] wrote {video_path}")
    finally:
        env.env.close()
    return written


def render_fallback_videos(args):
    gen_dir = latest_generation_dir(args)
    if gen_dir is None:
        print("[video] no generation directory found")
        return
    for name, tag in (("demo", "success"), ("demo_failed", "failed")):
        dataset_path = gen_dir / f"{name}.hdf5"
        if not hdf5_has_demos(dataset_path):
            continue
        try:
            render_hdf5_states_videos(
                dataset_path=dataset_path,
                video_dir=gen_dir / "videos",
                tag=tag,
                camera_names=args.video_camera,
                video_skip=args.video_skip,
                max_demos=args.fallback_video_n,
            )
        except Exception as exc:
            print(f"[video] failed for {dataset_path}: {type(exc).__name__}: {exc}")


def main():
    patch_python_paths()
    patch_huggingface_hub_compat()
    patch_robomimic_algo_imports()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args()

    import robocasa  # noqa: F401 - registers RoboCasa envs
    import mimicgen  # noqa: F401 - registers RoboCasa MimicGen interfaces

    import_mimicgen_robocasa_configs()
    install_embodiment_patches(args)

    source_lerobot_dir = args.source_lerobot_dir or latest_lerobot_dir(
        args.source_root,
        args.split,
        args.task,
    )
    template_path = find_template(args.task)

    print(f"[task] {args.task}")
    print(f"[source] {source_lerobot_dir}")
    print(f"[template] {template_path}")
    print(f"[target] {args.target_robot}/{args.target_gripper}")

    source_raw = build_source_raw_hdf5(args, source_lerobot_dir)
    prepare_source_for_mimicgen(args, source_raw)
    config_path = build_mimicgen_config(args, source_raw, template_path)
    if args.prepare_only:
        print("[prepare-only] skipped MimicGen generation")
        return
    run_mimicgen(args, config_path)
    render_fallback_videos(args)


if __name__ == "__main__":
    main()
