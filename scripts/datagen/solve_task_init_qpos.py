#!/usr/bin/env python3
"""
Solve a task-specific initial arm qpos for a target embodiment by matching
PandaOmron's actual initial end-effector pose *relative to its own base* for
one specific task, via damped-least-squares IK on the target's own (possibly
base-offset-patched) base pose.

Rationale: the UR5e/IIWA smoke tests for PickPlaceCounterToStove and
PickPlaceDrawerToCounter both got 0/160 successes across 8 tuning
configurations (base backoff 0/0.05/0.10/0.15, init_qpos patch on/off,
grasp-depth/rot-gain variants) -- ruling out those as the cause. The init_qpos
values that WERE tried (UR5E_KITCHEN_INIT_QPOS, BARX_ROBOT_INITIAL_QPOS) were
both derived by matching Panda's eef pose for BARX's *sink-adjacent* tasks
specifically; reusing them for a stove/drawer task is exactly the kind of
mismatch that produced the earlier "IIWA arm curled up in a nonsensical
posture" failure mode. This script re-derives a fresh init_qpos matched to
THIS task's own initial scene, the same methodology that produced
UR5E_KITCHEN_INIT_QPOS in the first place (per-task IK match, not reuse).

Usage:
    python solve_task_init_qpos.py --task PickPlaceCounterToStove --target-robot UR5eOmron
    python solve_task_init_qpos.py --task PickPlaceDrawerToCounter --target-robot IIWAOmron \
        --target-gripper Robotiq85Gripper

Multi-seed mode (--num-seeds > 1): the target end-effector pose generally has
more than one valid arm configuration (different elbow-up/down, different
wrist wrap). A single DLS IK run converges to whichever configuration is
nearest its starting seed, which can land in a poorly-conditioned
(near-singular) posture -- e.g. UR5eOmron/PickPlaceCounterToStove's solved
qpos put wrist_2 within ~5 deg of -90 deg, and instrumented action-tracking
showed the OSC controller only executing 5-30% of its commanded per-step
motion there (and reversing direction at the closest approach) -- a classic
near-singularity symptom, not a wrong-target problem. This mode tries several
randomized starting seeds and reports each solution's manipulability
(sqrt(det(J J^T)); higher = better-conditioned, farther from a singularity),
so a better-conditioned alternative configuration reaching the SAME target
pose can be picked instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import robosuite.utils.transform_utils as T

ROOT = Path("/data1/local/robocasa")


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


def eef_pose(env, arm="right"):
    robot = env.env.robots[0]
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[arm]
    sim = env.env.sim
    sim.forward()
    return (
        sim.data.site_xpos[int(site_id)].copy(),
        sim.data.site_xmat[int(site_id)].reshape(3, 3).copy(),
    )


def base_pose(env):
    sim = env.env.sim
    body_id = sim.model.body_name2id("mobilebase0_support")
    return (
        sim.data.body_xpos[body_id].copy(),
        sim.data.body_xmat[body_id].reshape(3, 3).copy(),
    )


def relative_pose(env):
    eef_p, eef_r = eef_pose(env)
    base_p, base_r = base_pose(env)
    pos_rel = base_r.T @ (eef_p - base_p)
    mat_rel = base_r.T @ eef_r
    return pos_rel, mat_rel


def manipulability(sim, robot, arm="right") -> float:
    """Yoshikawa manipulability index sqrt(det(J J^T)) of the 6xN arm Jacobian.
    Near 0 at a singularity; higher is better-conditioned."""
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[arm]
    site_name = sim.model.site_id2name(int(site_id))
    jacp = sim.data.get_site_jacp(site_name)
    jacr = sim.data.get_site_jacr(site_name)
    dof_idxs = [
        sim.model.jnt_dofadr[sim.model.joint_name2id(jn)]
        for jn in robot.robot_arm_joints
    ]
    J = np.vstack([jacp, jacr])[:, dof_idxs]
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def solve_dls_ik(
    env,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    arm="right",
    max_iters=400,
    damping=0.05,
    step_scale=1.0,
    seed_qpos: np.ndarray | None = None,
):
    robot = env.env.robots[0]
    sim = env.env.sim
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[arm]
    site_id = int(site_id)
    joint_idxs = robot._ref_arm_joint_pos_indexes
    n = len(joint_idxs)

    if seed_qpos is not None:
        for idx, val in zip(joint_idxs, seed_qpos):
            sim.data.qpos[idx] = val
        sim.forward()

    for it in range(max_iters):
        sim.forward()
        cur_pos = sim.data.site_xpos[site_id].copy()
        cur_mat = sim.data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = target_pos - cur_pos
        rot_err_mat = target_mat @ cur_mat.T
        rot_err = T.quat2axisangle(T.mat2quat(rot_err_mat))
        err = np.concatenate([pos_err, rot_err])

        if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
            break

        site_name = sim.model.site_id2name(site_id)
        jacp = sim.data.get_site_jacp(site_name)
        jacr = sim.data.get_site_jacr(site_name)
        J_full = np.vstack([jacp, jacr])

        dof_idxs = [
            sim.model.jnt_dofadr[sim.model.joint_name2id(jn)]
            for jn in robot.robot_arm_joints
        ]
        J = J_full[:, dof_idxs]

        lam = damping * np.eye(6)
        dq = J.T @ np.linalg.solve(J @ J.T + lam, err)
        dq = np.clip(dq * step_scale, -0.2, 0.2)

        qpos = np.array([sim.data.qpos[i] for i in joint_idxs])
        qpos = qpos + dq
        for idx, val in zip(joint_idxs, qpos):
            sim.data.qpos[idx] = val

    sim.forward()
    final_pos = sim.data.site_xpos[site_id].copy()
    final_mat = sim.data.site_xmat[site_id].reshape(3, 3).copy()
    residual_m = float(np.linalg.norm(target_pos - final_pos))
    residual_deg = float(
        np.degrees(
            np.linalg.norm(T.quat2axisangle(T.mat2quat(target_mat @ final_mat.T)))
        )
    )
    solved_qpos = np.array([sim.data.qpos[i] for i in joint_idxs])
    manip = manipulability(sim, robot, arm)
    return solved_qpos, residual_m, residual_deg, manip


def roll_variants(
    target_mat: np.ndarray, degrees_list: list[float]
) -> list[tuple[float, np.ndarray]]:
    """Rotate the target orientation about its own approach axis (local Z).
    A parallel gripper's grasp is usually insensitive to a range of roll
    about this axis (the live twist-alignment adaptation re-measures the
    gripper's opening axis at every reset and corrects for it independent of
    this choice), so this explores the redundant DOF a fixed-target IK search
    cannot reach."""
    variants = []
    for deg in degrees_list:
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        r_local = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        variants.append((deg, target_mat @ r_local))
    return variants


def check_arm_scene_collisions(env) -> list[str]:
    """Return names of non-gripper, non-object contact pairs involving the arm."""
    sim = env.env.sim
    contacts = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1 = sim.model.geom_id2name(c.geom1) or ""
        n2 = sim.model.geom_id2name(c.geom2) or ""
        names = (n1, n2)
        if not any(n.startswith("robot0") for n in names):
            continue
        if any("gripper0" in n for n in names):
            continue
        contacts.append(f"{n1} <-> {n2}")
    return contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-robot", required=True)
    parser.add_argument("--target-gripper", default="default")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--base-backoff", type=float, default=0.0)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "datasets" / "mg_runs" / "init_qpos_work",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Try this many randomized IK starting seeds and rank by manipulability",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for random IK starting seeds"
    )
    parser.add_argument(
        "--roll-search",
        type=float,
        nargs="+",
        default=None,
        help="Sweep these roll offsets (deg, about the gripper approach axis) instead of "
        "full-joint random seeding, and rank by manipulability at each.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    mg_gen = load_gen_module()

    source_dir = mg_gen.latest_lerobot_dir(
        ROOT / "datasets" / "v1.0", "pretrain", args.task
    )
    build_args = SimpleNamespace(
        task=args.task,
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
        model_file = demo.attrs["model_file"]
        ep_meta = json.loads(demo.attrs["ep_meta"])

    source_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
    )
    source_env.reset_to(
        {"states": states[0], "model": model_file, "ep_meta": json.dumps(ep_meta)}
    )
    pos_rel, mat_rel = relative_pose(source_env)
    source_env.env.close()
    print(f"[source] Panda base-relative eef pose: pos={np.round(pos_rel, 4)}")

    if args.base_backoff != 0.0:
        base_args = SimpleNamespace(
            target_robot=args.target_robot, base_backoff=args.base_backoff
        )
        mg_gen.install_base_backoff(base_args)

    import copy
    from robosuite.controllers import load_composite_controller_config

    target_meta = copy.deepcopy(env_meta)
    kwargs = target_meta["env_kwargs"]
    kwargs["robots"] = args.target_robot
    kwargs["gripper_types"] = args.target_gripper
    kwargs["layout_ids"] = (
        kwargs.get("layout_ids") if kwargs.get("layout_ids") not in (-2, None) else None
    )
    kwargs["style_ids"] = (
        kwargs.get("style_ids") if kwargs.get("style_ids") not in (-2, None) else None
    )
    if kwargs["layout_ids"] is None and kwargs["style_ids"] is None:
        kwargs["layout_and_style_ids"] = [[ep_meta["layout_id"], ep_meta["style_id"]]]
    kwargs["controller_configs"] = load_composite_controller_config(
        robot=args.target_robot
    )

    target_env = EnvUtils.create_env_from_metadata(
        env_meta=target_meta, render=False, render_offscreen=True, use_image_obs=False
    )
    target_env.reset()

    base_p, base_r = base_pose(target_env)
    desired_pos = base_p + base_r @ pos_rel
    desired_mat = base_r @ mat_rel

    robot = target_env.env.robots[0]
    joint_idxs = robot._ref_arm_joint_pos_indexes
    n_joints = len(joint_idxs)
    jnt_ranges = np.array(
        [
            target_env.env.sim.model.jnt_range[
                target_env.env.sim.model.joint_name2id(jn)
            ]
            for jn in robot.robot_arm_joints
        ]
    )
    stock_qpos = np.array([target_env.env.sim.data.qpos[i] for i in joint_idxs])

    rng = np.random.default_rng(args.seed)

    def try_seeds(target_pos, target_mat, label):
        candidates = []
        for seed_idx in range(args.num_seeds):
            if seed_idx == 0:
                seed_qpos = stock_qpos
            else:
                lo = np.where(np.isfinite(jnt_ranges[:, 0]), jnt_ranges[:, 0], -np.pi)
                hi = np.where(np.isfinite(jnt_ranges[:, 1]), jnt_ranges[:, 1], np.pi)
                lo = np.maximum(lo, -np.pi)
                hi = np.minimum(hi, np.pi)
                seed_qpos = rng.uniform(lo, hi)
            target_env.reset()
            solved_qpos, residual_m, residual_deg, manip = solve_dls_ik(
                target_env, target_pos, target_mat, seed_qpos=seed_qpos
            )
            contacts = check_arm_scene_collisions(target_env)
            candidates.append(
                (solved_qpos, residual_m, residual_deg, manip, contacts, seed_idx)
            )
        valid = [c for c in candidates if c[1] < 0.005 and c[2] < 2.0 and not c[4]]
        pool = valid if valid else candidates
        best = max(pool, key=lambda c: c[3])
        print(
            f"  {label}: best of {len(candidates)} seeds -> "
            f"residual={best[1]*1000:6.2f}mm/{best[2]:5.2f}deg manip={best[3]:.5f} "
            f"contacts={'none' if not best[4] else len(best[4])}"
            f"{'' if valid else '  [no seed met residual/collision bar]'}"
        )
        return best

    if args.roll_search is not None:
        print(
            f"Roll search over {args.roll_search} deg (each trying {args.num_seeds} IK seed(s)):"
        )
        results = []
        for deg, mat in roll_variants(desired_mat, args.roll_search):
            best = try_seeds(desired_pos, mat, f"roll={deg:+.0f}deg")
            results.append((deg, *best))
        best = max(results, key=lambda r: r[4])
        deg, solved_qpos, residual_m, residual_deg, manip, contacts, seed_idx = best
        print(
            f"\n[best] roll={deg:+.0f}deg (seed {seed_idx}): {args.target_robot} solved init_qpos: "
            f"{np.round(solved_qpos, 4).tolist()}"
        )
    else:
        print(f"Trying {args.num_seeds} IK seed(s) at the source-matched orientation:")
        solved_qpos, residual_m, residual_deg, manip, contacts, seed_idx = try_seeds(
            desired_pos, desired_mat, "no roll offset"
        )
        print(
            f"\n[best] seed {seed_idx}: {args.target_robot} solved init_qpos: {np.round(solved_qpos, 4).tolist()}"
        )

    target_env.env.close()
    print(
        f"[best] residual: {residual_m * 1000:.2f} mm, {residual_deg:.2f} deg, manipulability={manip:.5f}"
    )
    print(
        f"[best] arm-scene contacts at solved pose: {contacts if contacts else 'none'}"
    )


if __name__ == "__main__":
    main()
