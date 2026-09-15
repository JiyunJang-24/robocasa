#!/usr/bin/env python3
"""
Generate a BARX-style cross-embodiment RoboCasa dataset with MimicGen.

Wraps the proven UR5e pipeline (generate_mimicgen_ur5e_dataset.py) and extends
it to the BARX source embodiments (IIWA, Kinova3, UR5e) plus the PandaOmron
baseline, on the RoboCasa atomic tasks that correspond to the BARX benchmark
tasks:

    BARX task                RoboCasa atomic task (source demos, v1.0)
    ----------------------   -----------------------------------------
    pnp_counter_to_sink      PickPlaceCounterToSink
    pnp_sink_to_counter      PickPlaceSinkToCounter
    turn_on_sink_faucet      TurnOnSinkFaucet

Per-embodiment constants (initial arm pose) come from the BARX release
(BARX/robocasa_x/robocasa/environments/kitchen/barx_tasks.py). The gripper is
each arm's robosuite default, matching BARX's embodiment table (IIWA ->
Robotiq140, Kinova3/UR5e -> Robotiq85, Panda baseline -> PandaGripper).

One task x one embodiment per invocation (patches are process-global).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/local/robocasa")
GEN_PATH = ROOT / "generate_mimicgen_ur5e_dataset.py"
DEFAULT_OUTPUT_ROOT = Path("/data1/local/BARX/datasets/cross_embodiment")

# Initial arm poses from the BARX release (barx_tasks.py, BARX_ROBOT_INITIAL_QPOS).
# UR5eOmron intentionally keeps the pose already validated in this MimicGen
# pipeline (gen.UR5E_KITCHEN_INIT_QPOS) instead of the BARX value, because the
# BARX pose was calibrated against BARX's own base offsets / zero torso height.
BARX_ROBOT_INITIAL_QPOS = {
    "IIWAOmron": [
        0.01115337,
        -0.08399538,
        0.01529279,
        -1.22063772,
        0.00151392,
        2.00516437,
        0.02812827,
    ],
    "Kinova3Omron": [
        -0.05554795,
        -0.17629911,
        -0.0708673,
        1.4654714,
        -0.01258805,
        1.8501173,
        -1.69937106,
    ],
    "JacoOmron": [
        3.1776479,
        2.95208179,
        -0.02913301,
        1.20341971,
        -0.0055724,
        4.53421693,
        3.14988761,
    ],
}

# Base backoff (m) away from the counter, per embodiment. UR5e's 0.10 m was
# validated in earlier sessions (6-DOF upper-arm/counter collision). The BARX
# release places IIWA ~4 cm further back than Panda and Kinova3 ~6 cm closer,
# so the 7-DOF arms start from small values and can be overridden with
# --base-backoff.
DEFAULT_BASE_BACKOFF = {
    "PandaOmron": 0.0,
    "UR5eOmron": 0.10,
    "IIWAOmron": 0.05,
    "Kinova3Omron": 0.0,
    "JacoOmron": 0.0,
}

# RoboCasa v1.0 dataset task name -> MimicGen (legacy) task name used by its
# templates, env interfaces, and config classes.
MIMICGEN_TASK_ALIASES = {
    "PickPlaceCounterToSink": "PnPCounterToSink",
    "PickPlaceSinkToCounter": "PnPSinkToCounter",
    "PickPlaceCounterToCabinet": "PnPCounterToCabinet",
    "PickPlaceCounterToStove": "PnPCounterToStove",
    "PickPlaceDrawerToCounter": "PnPDrawerToCounter",
    "PickPlaceCounterToMicrowave": "PnPCounterToMicrowave",
    "PickPlaceMicrowaveToCounter": "PnPMicrowaveToCounter",
}


def load_gen_module():
    spec = importlib.util.spec_from_file_location("mg_ur5e_gen", GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mg_ur5e_gen"] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="TurnOnSinkFaucet")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--source-root", type=Path, default=ROOT / "datasets" / "v1.0")
    parser.add_argument("--source-lerobot-dir", type=Path, default=None)
    parser.add_argument(
        "--work-dir", type=Path, default=ROOT / "datasets" / "mimicgen_work"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-robot", default="UR5eOmron")
    parser.add_argument("--target-gripper", default="default")
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--source-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument(
        "--video-camera", nargs="+", default=["robot0_agentview_center"]
    )
    parser.add_argument("--render-attempt-video", action="store_true")
    parser.add_argument("--attempts-only", action="store_true")
    parser.add_argument("--force-source-raw", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--base-backoff",
        type=float,
        default=None,
        help="Override the per-embodiment default backoff (m)",
    )
    parser.add_argument("--grasp-depth-extra", type=float, default=0.01)
    parser.add_argument("--rot-gain", type=float, default=2.5)
    parser.add_argument("--osc-kp", type=float, default=300.0)
    parser.add_argument("--no-twist-align", action="store_true")
    parser.add_argument("--no-pad-friction-boost", action="store_true")
    parser.add_argument("--no-init-qpos-patch", action="store_true")
    parser.add_argument(
        "--init-qpos",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override the target robot's init_qpos directly (task-specific IK "
            "match; see solve_task_init_qpos.py). Takes precedence over both "
            "the UR5e kitchen-init and BARX_ROBOT_INITIAL_QPOS defaults, which "
            "were calibrated for BARX's sink-adjacent tasks specifically and "
            "do not generalize to unrelated fixtures (stove/drawer/etc)."
        ),
    )
    parser.add_argument("--fallback-video-n", type=int, default=10)
    args = parser.parse_args()

    if args.base_backoff is None:
        args.base_backoff = DEFAULT_BASE_BACKOFF.get(args.target_robot, 0.0)
    return args


_EXTRA_OMRON_CLASSES: dict[str, type] = {}


def register_extra_omron_robots():
    """The conda robosuite defines PandaOmron/UR5eOmron/IIWAOmron but not
    Kinova3Omron or JacoOmron; register them the same way (BARX gets them from
    robosuite_models / robocasa_x.compositional_robots). JacoOmron follows the
    BARX class: Robotiq85Gripper instead of the Jaco three-finger hand."""
    import robosuite.robots as rs_robots
    from robosuite.robots import register_robot_class

    if "Kinova3Omron" not in rs_robots.ROBOT_CLASS_MAPPING:
        from robosuite.models.robots import Kinova3

        @register_robot_class("WheeledRobot")
        class Kinova3Omron(Kinova3):
            @property
            def default_base(self):
                return "OmronMobileBase"

            @property
            def default_arms(self):
                return {"right": "Kinova3"}

            @property
            def init_torso_qpos(self):
                return np.array([0.2])

            @property
            def base_xpos_offset(self):
                return {
                    "bins": (-0.6, -0.1, 0),
                    "empty": (-0.6, 0, 0),
                    "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                }

        _EXTRA_OMRON_CLASSES["Kinova3Omron"] = Kinova3Omron
        print("[patch] registered Kinova3Omron (WheeledRobot)")

    if "JacoOmron" not in rs_robots.ROBOT_CLASS_MAPPING:
        from robosuite.models.robots import Jaco

        @register_robot_class("WheeledRobot")
        class JacoOmron(Jaco):
            @property
            def default_base(self):
                return "OmronMobileBase"

            @property
            def default_arms(self):
                return {"right": "Jaco"}

            @property
            def default_gripper(self):
                return {"right": "Robotiq85Gripper"}

            @property
            def init_torso_qpos(self):
                return np.array([0.2])

            @property
            def base_xpos_offset(self):
                return {
                    "bins": (-0.6, -0.1, 0),
                    "empty": (-0.6, 0, 0),
                    "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                }

        _EXTRA_OMRON_CLASSES["JacoOmron"] = JacoOmron
        print("[patch] registered JacoOmron (WheeledRobot, Robotiq85)")


def register_kinova3_omron():
    """Backward-compatible alias."""
    register_extra_omron_robots()


def register_task_aliases(task: str):
    """RoboCasa v1.0 renamed PnP* -> PickPlace*; MimicGen still registers the
    old names. Register thin alias subclasses under the new names."""
    mg_name = MIMICGEN_TASK_ALIASES.get(task)
    if mg_name is None:
        return
    import mimicgen.env_interfaces.robocasa.single_stage.mg_pnp as mg_pnp
    import mimicgen.configs.robocasa.single_stage.config_pnp as config_pnp

    interface_base = getattr(mg_pnp, f"MG_{mg_name}")
    type(f"MG_{task}", (interface_base,), {"__module__": __name__})

    config_base = None
    for attr in vars(config_pnp).values():
        if isinstance(attr, type) and getattr(attr, "NAME", None) == mg_name:
            config_base = attr
            break
    if config_base is None:
        raise RuntimeError(f"No MimicGen config class registered for {mg_name}")
    type(f"Config_{task}", (config_base,), {"NAME": task, "__module__": __name__})
    print(f"[patch] aliased {task} -> {mg_name} (interface + config)")


def patch_find_template(gen):
    orig = gen.find_template

    def find_template(task: str) -> Path:
        return orig(MIMICGEN_TASK_ALIASES.get(task, task))

    gen.find_template = find_template


def patch_target_controller_config(gen):
    """Kinova3Omron has no default_<robot>.json in robosuite; the Omron
    HYBRID_MOBILE_BASE composite config is arm-agnostic, so fall back to the
    IIWAOmron config instead of robosuite's silent basic.json fallback."""
    import robosuite

    orig = gen.target_controller_config

    def target_controller_config(target_robot: str, osc_kp: float | None = None):
        from robosuite.controllers import load_composite_controller_config

        config_dir = (
            Path(robosuite.__file__).parent / "controllers" / "config" / "robots"
        )
        if (config_dir / f"default_{target_robot.lower()}.json").exists():
            return orig(target_robot, osc_kp)
        # Load through robosuite's loader so its "arms" flattening applies.
        cfg = load_composite_controller_config(
            controller=str(config_dir / "default_iiwaomron.json")
        )
        if osc_kp is not None:
            right = cfg.get("body_parts", {}).get("right", {})
            if "kp" in right:
                kp_limits = right.get("kp_limits")
                right["kp"] = (
                    float(np.clip(osc_kp, kp_limits[0], kp_limits[1]))
                    if kp_limits is not None
                    else osc_kp
                )
        print(f"[patch] {target_robot}: using arm-agnostic Omron controller config")
        return cfg

    gen.target_controller_config = target_controller_config


def install_generic_init_qpos(args, gen):
    if args.no_init_qpos_patch or args.target_robot == "PandaOmron":
        return
    if args.init_qpos is not None:
        if args.target_robot in _EXTRA_OMRON_CLASSES:
            cls = _EXTRA_OMRON_CLASSES[args.target_robot]
        else:
            import robosuite.models.robots.compositional as compositional

            cls = getattr(compositional, args.target_robot)
        qpos_arr = np.asarray(args.init_qpos, dtype=np.float64)
        cls.init_qpos = property(lambda self, q=qpos_arr: q.copy())
        print(
            f"[patch] {args.target_robot} task-specific init_qpos (--init-qpos) {np.round(qpos_arr, 4)}"
        )
        return
    if args.target_robot == "UR5eOmron":
        gen.install_init_qpos_patch(args)
        return
    qpos = BARX_ROBOT_INITIAL_QPOS.get(args.target_robot)
    if qpos is None:
        print(
            f"[patch] no kitchen init_qpos known for {args.target_robot}; keeping stock pose"
        )
        return
    if args.target_robot in _EXTRA_OMRON_CLASSES:
        cls = _EXTRA_OMRON_CLASSES[args.target_robot]
    else:
        import robosuite.models.robots.compositional as compositional

        cls = getattr(compositional, args.target_robot)
    qpos_arr = np.asarray(qpos, dtype=np.float64)
    cls.init_qpos = property(lambda self, q=qpos_arr: q.copy())
    print(f"[patch] {args.target_robot} BARX kitchen init_qpos {np.round(qpos_arr, 4)}")


def main():
    gen = load_gen_module()
    gen.patch_python_paths()
    gen.patch_huggingface_hub_compat()
    gen.patch_robomimic_algo_imports()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args()

    import robocasa  # noqa: F401 - registers RoboCasa envs
    import mimicgen  # noqa: F401 - registers MimicGen interfaces

    gen.import_mimicgen_robocasa_configs()
    register_kinova3_omron()
    register_task_aliases(args.task)
    patch_find_template(gen)
    patch_target_controller_config(gen)

    gen.install_base_backoff(args)
    install_generic_init_qpos(args, gen)
    gen.install_pad_friction_boost(args)
    gen.install_grasp_pose_adaptation(args)

    source_lerobot_dir = args.source_lerobot_dir or gen.latest_lerobot_dir(
        args.source_root, args.split, args.task
    )
    template_path = gen.find_template(args.task)

    print(f"[task] {args.task}")
    print(f"[source] {source_lerobot_dir}")
    print(f"[template] {template_path}")
    print(
        f"[target] {args.target_robot}/{args.target_gripper} backoff={args.base_backoff}"
    )

    source_raw = gen.build_source_raw_hdf5(args, source_lerobot_dir)
    gen.prepare_source_for_mimicgen(args, source_raw)
    config_path = gen.build_mimicgen_config(args, source_raw, template_path)
    if args.prepare_only:
        print("[prepare-only] skipped MimicGen generation")
        return
    gen.run_mimicgen(args, config_path)
    gen.render_fallback_videos(args)


if __name__ == "__main__":
    main()
