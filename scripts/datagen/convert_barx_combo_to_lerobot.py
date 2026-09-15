#!/usr/bin/env python3
"""
Thin wrapper around robocasa's own convert_hdf5_lerobot.py that additionally
labels the output dataset with the real robot+gripper combo instead of the
hardcoded "PandaOmron" the upstream script writes.

The upstream script otherwise does the full, official HDF5(robomimic-style,
successes only) -> LeRobot v2.1 dataset conversion: replays states through a
freshly rendered env to get images/state/reward/done, writes video-backed
parquet episodes, and populates meta/{info,modality,embodiment,tasks,stats,
episodes,episodes_stats}.json(l) and extras/{dataset_meta.json,episode_*/}.
Nothing about that logic is embodiment-specific (the unified BARX/RoboCasa-X
action-state layout is embodiment-agnostic), so it works unmodified for
non-Panda robots -- only the human-readable robot_type/embodiment_tag labels
need correcting afterward.

Usage:
    python convert_barx_combo_to_lerobot.py \
        --raw-hdf5 merged.hdf5 \
        --output-dir /data1/local/BARX/datasets/lerobot_datasets/UR5eOmron_Robotiq85Gripper/PickPlaceCounterToStove/2026-07-23 \
        --robot-type UR5eOmron_Robotiq85Gripper
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/data1/local/robocasa")


def load_converter_module():
    spec = importlib.util.spec_from_file_location(
        "convert_hdf5_lerobot",
        ROOT / "robocasa" / "scripts" / "dataset_scripts" / "convert_hdf5_lerobot.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_gripper_qpos_width(converter):
    """robocasa's reorder_hdf5_state assumes robot0_gripper_qpos has exactly 2
    columns (matching PandaGripper's 2-DOF parallel fingers). Other grippers
    expose more raw joint values -- e.g. Robotiq85Gripper's 4-bar linkage has
    8 (driver/coupler/spring_link/follower per side). Downselect to the N
    driver joints (evenly spaced across the raw array, one per side) instead
    of crashing on a shape mismatch; the passive/coupled joints move in lockstep
    with the driver joint anyway, so this loses no information the target
    modality.json's 2-wide gripper_qpos field could represent regardless."""
    import numpy as np

    original = converter.reorder_hdf5_state

    def reorder_hdf5_state(state_hdf5, modality_dict):
        state_info = modality_dict["state"]
        gripper_key = "gripper_qpos"
        if gripper_key in state_info:
            target_width = (
                state_info[gripper_key]["end"] - state_info[gripper_key]["start"]
            )
            raw = state_hdf5["robot0_gripper_qpos"]
            raw_width = raw.shape[1]
            if raw_width != target_width:
                idxs = [int(i * raw_width / target_width) for i in range(target_width)]
                state_hdf5 = dict(state_hdf5)
                state_hdf5["robot0_gripper_qpos"] = np.asarray(raw)[:, idxs]
                print(
                    f"[patch] robot0_gripper_qpos: took columns {idxs} of {raw_width} "
                    f"to match modality.json's {target_width}-wide gripper_qpos"
                )
        return original(state_hdf5, modality_dict)

    converter.reorder_hdf5_state = reorder_hdf5_state


def patch_torch_numpy_interop():
    """This conda env has torch 2.0.1 (built against the numpy 1.x C ABI) but
    robocasa hard-asserts numpy==2.2.5 at import time -- so torch.Tensor.numpy()
    raises "RuntimeError: Numpy is not available" for the ENTIRE process
    lifetime (lerobot's LeRobotDataset.save_episode() calls this internally).
    Downgrading numpy breaks robocasa outright (assert in robocasa/__init__.py)
    and would silently corrupt the live MimicGen generation jobs sharing this
    env, so neither side's declared version can move. Route the conversion
    through Tensor.tolist() -> np.array() instead: tolist() is pure torch (no
    numpy C-extension call), so it works under any installed numpy version;
    only the final np.array() call touches numpy, safely, on plain Python
    floats/ints."""
    import numpy as np
    import torch

    def safe_numpy(self, *args, **kwargs):
        return np.array(self.detach().cpu().tolist())

    torch.Tensor.numpy = safe_numpy


def patch_robot_type(converter, robot_type: str):
    """LerobotDatasetWrapper.create(robot_type="PandaOmron", ...) is a
    positional-ish kwarg baked into convert_hdf5_lerobot.main(); rebind the
    module's LerobotDatasetWrapper.create classmethod so our robot_type flows
    through without touching the upstream file."""
    original_create = converter.LerobotDatasetWrapper.create

    @classmethod
    def create_with_robot_type(cls, *args, **kwargs):
        kwargs["robot_type"] = robot_type
        return original_create.__func__(cls, *args, **kwargs)

    converter.LerobotDatasetWrapper.create = create_with_robot_type


def fix_embodiment_metadata(lerobot_dir: Path, robot_type: str, embodiment_tag: str):
    embodiment_path = lerobot_dir / "meta" / "embodiment.json"
    embodiment = json.loads(embodiment_path.read_text())
    embodiment["robot_name"] = robot_type
    embodiment["robot_type"] = robot_type
    embodiment["embodiment_tag"] = embodiment_tag
    embodiment_path.write_text(json.dumps(embodiment, indent=2))

    info_path = lerobot_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = robot_type
    info_path.write_text(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-hdf5", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain <output-dir>/lerobot/ (mirrors <Task>/<date>/ in v1.0)",
    )
    parser.add_argument("--robot-type", type=str, required=True)
    parser.add_argument("--embodiment-tag", type=str, default=None)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=[
            "robot0_eye_in_hand",
            "robot0_agentview_left",
            "robot0_agentview_right",
        ],
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staged_hdf5 = args.output_dir / "raw.hdf5"
    if staged_hdf5.resolve() != args.raw_hdf5.resolve():
        shutil.copy2(args.raw_hdf5, staged_hdf5)

    patch_torch_numpy_interop()
    converter = load_converter_module()
    embodiment_tag = args.embodiment_tag or f"robocasa_{args.robot_type.lower()}"
    patch_robot_type(converter, args.robot_type)
    patch_gripper_qpos_width(converter)

    converter.main(
        SimpleNamespace(
            raw_dataset_path=str(staged_hdf5),
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )
    )

    lerobot_dir = args.output_dir / "lerobot"
    fix_embodiment_metadata(lerobot_dir, args.robot_type, embodiment_tag)
    print(f"[done] {lerobot_dir}")


if __name__ == "__main__":
    main()
