#!/usr/bin/env python3
"""
RoboCasa port of shortcut-learning-in-grps's
experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_visual_robust_cross_embodiment.py

Takes a v1.0 atomic LeRobot dataset (PandaOmron source demos) and, for each
episode, replays the SAME raw action sequence across several embodiments
(default PandaOmron/UR5eOmron/IIWAOmron) with all scene objects flown far out
of camera view first, so the only thing that varies between embodiment tracks
is the robot itself.

Ported technique (matches the LIBERO script exactly, see
libero_ur5_utils.convert_franka_flat_to_ur5e_flat / solve_ik_ur5e_inplace):
  1. Reconstruct the episode's exact kitchen scene per embodiment straight
     from its own reset (RoboCasa env_kwargs only need `robots` swapped --
     unlike LIBERO, this already gives every embodiment the identical
     fixtures/layout for free, no qpos-mask copying required).
  2. Fly every object in `env.objects` far out of frame (world-frame
     translation of its free joint, large per-object spacing -- small
     spacing was found by the LIBERO version to overlap flown-away bodies
     and blow up the physics).
  3. Solve a small damped-least-squares Jacobian IK (position-only by
     default) on ONLY the target embodiment's arm joints so its eef site
     starts at the same world position the source Panda trajectory starts
     at -- this is the one-time "aligned start" step.
  4. From that aligned start, replay the exact same raw action sequence
     (independently, open-loop, through each embodiment's own controller)
     that produced the source demo. Because every RoboCasa Omron composite
     robot shares the same 12-dim raw action space (arm-agnostic OSC_POSE +
     gripper + base + torso + mode) and starts eef-aligned, the resulting
     eef/grip-site trajectory tracks closely step-by-step across embodiments
     without any further per-step correction -- exactly the LIBERO script's
     "same grip-site every step" property.

Run under the conda `robocasa` interpreter (has the up-to-date
robosuite/robocasa/mujoco stack this v1.0 data needs -- see claude.txt for
why barx/robocasa_x can't replay it):

    MUJOCO_GL=egl /home/fick17/anaconda3/envs/robocasa/bin/python \
        generate_visual_robust_cross_embodiment.py \
        --source-lerobot-dir /data1/local/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceSinkToCounter/20250819/lerobot \
        --episodes 0 1 2 --robots PandaOmron UR5eOmron IIWAOmron
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/data1/local/robocasa")
ROBOSUITE_ROOT = ROOT.parent / "robosuite"
ROBOMIMIC_ROOT = ROOT / "robomimic"

SETTLE_STEPS = 10
CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

# "Kitchen-ready" arm joint configurations (elbow-up, reasonable reach for a
# counter/sink-adjacent task), used to seed non-source embodiments' IK at
# t=0 instead of their raw (often folded/tabletop) default reset pose.
# Ported from /data1/local/barx/scripts/generate_atomic_pnp_cross_embodiment.py
# (UR5E_KITCHEN_INIT_QPOS: DLS-IK matched to PandaOmron's initial
# base-relative EEF pose, residual 0.15mm; IIWA_KITCHEN_INIT_QPOS: BARX
# release barx_tasks.py). A bad t=0 seed was found to be the actual root
# cause of the "IK converges to a valid but contorted/elbow-flipped pose"
# bug reported mid-episode: DLS IK is a *local* method warm-started from the
# previous solution, so a bad starting branch at t=0 (or a large joint jump
# introduced by physically stepping the env between per-step IK solves, see
# `align()` below) propagates through the rest of the episode.
KITCHEN_INIT_QPOS = {
    "UR5eOmron": np.array([-0.9593, -2.1292, 1.2628, -0.7801, -1.3080, 2.1020]),
    "IIWAOmron": np.array(
        [
            0.01115337,
            -0.08399538,
            0.01529279,
            -1.22063772,
            0.00151392,
            2.00516437,
            0.02812827,
        ]
    ),
}
FPS = 20

# Same raw<->LeRobot action reordering as convert_mg_to_atomic_lerobot.py /
# robocasa's own lerobot_utils.py (ACTION_KEY_ORDERING_HDF5 + modality.json).
ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}
MODALITY_ACTION = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}


def patch_python_paths():
    for path in (str(ROOT), str(ROBOSUITE_ROOT), str(ROBOMIMIC_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def robotiq85_mesh_swap_note():
    """This machine's /data1/local/robosuite has a local, undocumented patch:
    Robotiq85GripperBase loads a swapped-in MuJoCo-Menagerie 2f85 mesh
    (grippers/robotiq_2f85_menagerie/2f85_robosuite.xml, ONE tendon-coupled
    actuator) instead of robosuite's stock grippers/robotiq_gripper_85.xml
    (still on disk, unreferenced; barx's own pinned/unmodified robosuite in
    /data1/local/barx/.venv still loads it -- confirmed via `git diff` on
    /data1/local/robosuite earlier this session). The Menagerie mesh is
    mounted ~90 degrees off from the stock mesh's opening-axis convention,
    which is why UR5e's gripper visibly looks rotated relative to Panda's
    even with the eef *site* orientation IK-matched (same site-x vs site-y
    mismatch this session's earlier generate_atomic_pnp_cross_embodiment.py
    twist-corrected for the delta-action path).

    Reverting to the stock mesh (tried first) does NOT work here: the stock
    XML has TWO independent position actuators (control_dim=2,
    grippers/robotiq_gripper_85.xml:32-33) instead of the Menagerie mesh's
    one tendon-coupled actuator (control_dim=1) --
    robosuite/controllers/parts/gripper/simple_grip.py:87 derives
    control_dim straight from the actuator count, so it rejects RoboCasa's
    canonical 1-dim gripper_close action ("Expected 2, got 1"). The mesh
    swap was very likely made specifically so Robotiq85 fits RoboCasa's
    embodiment-agnostic 1-dim gripper action space, not by accident.

    So instead of changing the mesh, measure_opening_axis()/build_twist()
    below live-measure each robot's actual gripper opening axis in its own
    eef-site frame and fold a compensating Rz twist into the IK *target*
    orientation each step -- same fix, applied to the orientation target
    instead of the asset."""


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


def measure_opening_axis(base_env):
    """Unit vector (in the eef-site local frame) along which this robot's
    gripper pads separate. Ported from generate_atomic_pnp_cross_embodiment.py
    (_opening_axis_in_site_frame)."""
    sim = base_env.sim
    sid = eef_site_id(base_env.robots[0])
    sim.forward()
    site_pos = sim.data.site_xpos[sid].copy()
    site_r = sim.data.site_xmat[sid].reshape(3, 3).copy()
    groups = {}
    for gid, name in _gripper_pad_geom_ids(sim):
        side = "l" if ("left" in name or "finger2" in name) else "r"
        groups.setdefault(side, []).append(sim.data.geom_xpos[gid].copy())
    if len(groups) < 2:
        return np.array([0.0, 1.0, 0.0])
    d_site = site_r.T @ (np.mean(groups["r"], axis=0) - np.mean(groups["l"], axis=0))
    d_site[2] = 0.0
    n = np.linalg.norm(d_site)
    return d_site / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def build_twist_to_reference_axis(this_axis, reference_axis):
    """Rz(theta) rotating this_axis onto reference_axis, folded into
    [-90, 90] deg since parallel-jaw grippers are 180-deg symmetric."""
    ang = np.arctan2(reference_axis[1], reference_axis[0]) - np.arctan2(
        this_axis[1], this_axis[0]
    )
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    if ang > np.pi / 2:
        ang -= np.pi
    elif ang < -np.pi / 2:
        ang += np.pi
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def lerobot_action_to_raw(action_lerobot: np.ndarray) -> np.ndarray:
    """Inverse of robocasa's lerobot_utils.reorder_hdf5_action: undo the
    LeRobot column order back to the native env.step() 12-dim action."""
    raw = np.zeros_like(action_lerobot)
    for key, (raw_start, raw_end) in ACTION_KEY_ORDERING_HDF5.items():
        lerobot_start, lerobot_end = MODALITY_ACTION[key]
        raw[:, raw_start:raw_end] = action_lerobot[:, lerobot_start:lerobot_end]
    return raw


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-lerobot-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="episode indices to process (default: --num-episodes starting at 0)",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--robots", nargs="+", default=["PandaOmron", "UR5eOmron", "IIWAOmron"]
    )
    parser.add_argument(
        "--hide-scope",
        choices=["all_objects", "robot_only", "none"],
        default="robot_only",
        help="all_objects: fly task objects away, kitchen fixtures stay visible. "
        "robot_only: also hide the kitchen background (counter/sink/cabinets/walls/floor) "
        "so only the robot renders. none: no removal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "datasets" / "visual_robust_cross_embodiment",
    )
    parser.add_argument("--camera-names", nargs="+", default=CAMERA_NAMES)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    parser.add_argument(
        "--no-match-orientation",
        dest="match_orientation",
        action="store_false",
        help="IK-match position only, not orientation (default: match both -- validated to converge "
        "to <0.05deg error and is needed for the gripper-twist correction to be meaningful)",
    )
    parser.add_argument("--fly-base-offset", type=float, default=50.0)
    parser.add_argument("--fly-spacing", type=float, default=5.0)
    parser.add_argument(
        "--osc-kp",
        type=float,
        default=300.0,
        help="OSC controller kp for non-Panda arms (300 was validated for UR5e/IIWA earlier this session)",
    )
    parser.add_argument("--preview-frames", type=int, default=8)
    parser.add_argument(
        "--num-example-episodes",
        type=int,
        default=3,
        help="How many of the processed episodes also get individual per-robot MP4s/preview JPGs/HDF5 "
        "saved under <output-dir>/examples/ (the full LeRobot dataset is written for every "
        "episode regardless).",
    )
    parser.add_argument(
        "--no-lerobot",
        dest="write_lerobot",
        action="store_false",
        help="Skip writing the combined LeRobot dataset (data/videos/meta), examples only.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(match_orientation=True, write_lerobot=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Object removal ("fly away") -- RoboCasa equivalent of the LIBERO script's
# fly_objects_away/reposition_object.
# ---------------------------------------------------------------------------


def fly_objects_away(base_env, base_offset: float, spacing: float):
    for idx, (name, obj) in enumerate(base_env.objects.items()):
        delta = np.array([base_offset + spacing * idx, base_offset, base_offset])
        for joint_name in obj.joints:
            addr = base_env.sim.model.get_joint_qpos_addr(joint_name)
            if isinstance(addr, tuple):
                start, end = addr
                if end - start >= 3:
                    base_env.sim.data.qpos[start : start + 3] += delta
    base_env.sim.forward()


# Standard RoboCasa/robosuite naming prefixes for robot-owned geoms (arm
# links, gripper fingers, Omron mobile base). Everything else -- objects,
# distractors, counters/sinks/cabinets/walls/floor -- is kitchen scenery.
_ROBOT_GEOM_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")


def hide_kitchen_background(base_env):
    """Make every non-robot geom fully transparent so only the robot renders
    against the empty skybox/background -- stronger than fly_objects_away,
    which only moves the movable task objects and leaves fixtures (counter,
    sink, cabinets, walls, floor) visible. Alpha-hiding (rather than flying
    fixtures away like objects) avoids re-deriving per-fixture joint/anchor
    handling for dozens of mostly-static bodies, and can't perturb physics
    since collision geometry is untouched -- only the rgba alpha channel."""
    model = base_env.sim.model
    for gid in range(model.ngeom):
        name = model.geom_id2name(gid) or ""
        if not name.startswith(_ROBOT_GEOM_PREFIXES):
            model.geom_rgba[gid, 3] = 0.0
    base_env.sim.forward()


# ---------------------------------------------------------------------------
# Position-only DLS Jacobian IK to align one embodiment's starting eef site
# to a target world position. Ported from libero_ur5_utils.solve_ik_ur5e_inplace.
# ---------------------------------------------------------------------------


def _jac_site_compat(sim, site_id: int):
    import mujoco

    jacp = np.zeros((3, sim.model.nv), dtype=np.float64)
    jacr = np.zeros((3, sim.model.nv), dtype=np.float64)
    m, d = sim.model, sim.data
    for attr in ("_model", "model", "_mjmodel", "_m"):
        if hasattr(m, attr):
            m = getattr(m, attr)
            break
    for attr in ("_data", "data", "_mjdata", "_d"):
        if hasattr(d, attr):
            d = getattr(d, attr)
            break
    mujoco.mj_jacSite(m, d, jacp, jacr, int(site_id))
    return jacp, jacr


def _rot_to_omega(r_err: np.ndarray) -> np.ndarray:
    tr = np.trace(r_err)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    if ang < 1e-8:
        return np.zeros(3)
    w = np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ]
    )
    return w / (2.0 * np.sin(ang)) * ang


def eef_site_id(robot):
    site_id = robot.eef_site_id
    if isinstance(site_id, dict):
        site_id = site_id[robot.arms[0]]
    return int(site_id)


def eef_world_pose(base_env):
    sim = base_env.sim
    sid = eef_site_id(base_env.robots[0])
    sim.forward()
    pos = sim.data.site_xpos[sid].copy()
    xmat = sim.data.site_xmat[sid].reshape(3, 3).copy()
    return pos, xmat


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def measure_fingertip_offset_local(base_env):
    """Vector from the eef *site* to the actual fingertip/pad midpoint,
    expressed in the site's own local frame. Rigid attachment -> constant
    regardless of arm pose, exactly like measure_opening_axis(). Site-to-
    fingertip gaps were found to differ a lot per robot (Panda 3.6mm, UR5e
    14.3mm, IIWA 22.2mm), so IK-matching sites alone leaves the actual
    grip-site (fingertip midpoint, what the user asked to match) off by up
    to ~1.9cm cross-embodiment even though the sites match to <0.1mm."""
    sim = base_env.sim
    sid = eef_site_id(base_env.robots[0])
    sim.forward()
    site_pos = sim.data.site_xpos[sid].copy()
    site_r = sim.data.site_xmat[sid].reshape(3, 3).copy()
    pad_ids = _gripper_pad_geom_ids(sim)
    fingertip_world = np.mean([sim.data.geom_xpos[gid] for gid, _ in pad_ids], axis=0)
    return site_r.T @ (fingertip_world - site_pos)


def fingertip_world_pose(base_env, offset_local):
    sim = base_env.sim
    sid = eef_site_id(base_env.robots[0])
    sim.forward()
    site_pos = sim.data.site_xpos[sid].copy()
    site_r = sim.data.site_xmat[sid].reshape(3, 3).copy()
    return site_pos + site_r @ offset_local, site_r


def solve_ik_to_eef_pose(
    base_env,
    target_pos,
    target_xmat,
    match_orientation=False,
    fingertip_offset_local=None,
    n_iters=400,
    pos_tol=1e-4,
    ori_tol=1e-3,
    damping=2e-2,
    step=0.25,
):
    """IK to a target pose. If `fingertip_offset_local` is given, `target_pos`
    is matched at the fingertip/pad midpoint (Jacobian-transported from the
    site) instead of the site origin itself -- orientation is still matched
    at the site (that's what governs the visible gripper twist)."""
    sim = base_env.sim
    robot = base_env.robots[0]
    sid = eef_site_id(robot)
    jpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    jvel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)

    for _ in range(n_iters):
        sim.forward()
        site_pos = sim.data.site_xpos[sid].copy()
        cur_xmat = sim.data.site_xmat[sid].reshape(3, 3).copy()
        if fingertip_offset_local is not None:
            r_world = cur_xmat @ fingertip_offset_local
            cur_pos = site_pos + r_world
        else:
            r_world = None
            cur_pos = site_pos
        pos_err = target_pos - cur_pos
        ori_err = (
            _rot_to_omega(target_xmat @ cur_xmat.T)
            if match_orientation
            else np.zeros(3)
        )
        if np.linalg.norm(pos_err) < pos_tol and (
            not match_orientation or np.linalg.norm(ori_err) < ori_tol
        ):
            break

        jacp, jacr = _jac_site_compat(sim, sid)
        jp = jacp[:, jvel_idx]
        jr_full = jacr[:, jvel_idx]
        if r_world is not None:
            jp = jp - _skew(r_world) @ jr_full
        if match_orientation:
            j = np.vstack([jp, jr_full])
            err = np.concatenate([pos_err, ori_err])
        else:
            j = jp
            err = pos_err

        a = j @ j.T + (damping**2) * np.eye(j.shape[0])
        dq = j.T @ np.linalg.solve(a, err)
        q = sim.data.qpos[jpos_idx].copy()
        sim.data.qpos[jpos_idx] = q + step * dq
    sim.forward()


# ---------------------------------------------------------------------------
# Env creation / replay
# ---------------------------------------------------------------------------


def target_controller_config(target_robot: str, osc_kp: float):
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(robot=target_robot)
    right = cfg.get("body_parts", {}).get("right", {})
    if "kp" in right:
        kp_limits = right.get("kp_limits")
        right["kp"] = (
            float(np.clip(osc_kp, kp_limits[0], kp_limits[1])) if kp_limits else osc_kp
        )
    return cfg


def resolve_env_name(name: str) -> str:
    """Map a BARX robocasa_x env name onto this (non-x) robocasa's registry.

    atomic_new (built for robocasa_x, whose PnP tasks are the "X"-prefixed
    variants) writes env_name="XPnPCounterToSink"; this codebase registers
    the un-prefixed "PnPCounterToSink" instead. Both alias to the same
    "PickPlace..." name too, so cover that direction as well.
    """
    from robosuite.environments.base import REGISTERED_ENVS

    registered = {getattr(e, "__name__", str(e)) for e in REGISTERED_ENVS}
    if name in registered:
        return name
    candidates = [name[1:]] if name.startswith("X") else []
    for base in (name[1:] if name.startswith("X") else name,):
        if base.startswith("PnP"):
            candidates.append("PickPlace" + base[len("PnP") :])
        elif base.startswith("PickPlace"):
            candidates.append("PnP" + base[len("PickPlace") :])
    for alt in candidates:
        if alt in registered:
            return alt
    raise ValueError(
        f"env {name!r} not registered and no alias matched (tried {candidates})"
    )


def create_env(
    env_kwargs_base: dict, robot: str, camera_names, camera_h, camera_w, osc_kp: float
):
    from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={
            "obs": {
                "low_dim": ["robot0_eef_pos"],
                "rgb": [f"{c}_image" for c in camera_names],
            }
        }
    )
    env_meta = {
        "env_name": resolve_env_name(env_kwargs_base["env_name"]),
        "type": 1,
        "env_kwargs": dict(env_kwargs_base["env_kwargs"]),
    }
    env_meta["env_kwargs"]["robots"] = robot
    env_meta["env_kwargs"]["gripper_types"] = "default"
    # robocasa_x (which recorded atomic_new) uses "A"/"B" object splits; this
    # standalone robocasa only knows "pretrain"/"target"/None and otherwise
    # raises inside sample_kitchen_object_helper. The split only constrains
    # which instances a *freshly sampled* replacement object may be drawn
    # from (used by install_object_registry_fallback below when the exact
    # recorded instance isn't registered here) -- not something that needs to
    # match the original recording for a visual-robustness render.
    split = env_meta["env_kwargs"].get("obj_instance_split")
    if split not in (None, "pretrain", "target"):
        env_meta["env_kwargs"]["obj_instance_split"] = None
    if robot != "PandaOmron":
        env_meta["env_kwargs"]["controller_configs"] = target_controller_config(
            robot, osc_kp
        )
    # atomic_new's env_kwargs came from BARX robocasa_x, whose Kitchen accepts
    # a few kwargs (e.g. obj_init_range) this standalone robocasa's Kitchen
    # does not. They only affect how a *freshly sampled* scene would look;
    # every episode here is rebuilt from its recorded ep_meta/model XML
    # instead, so dropping them changes nothing about what gets replayed.
    import re as _re

    dropped = []
    while True:
        try:
            return EnvUtils.create_env_for_data_processing(
                env_meta=env_meta,
                camera_names=camera_names,
                camera_height=camera_h,
                camera_width=camera_w,
                reward_shaping=False,
            )
        except TypeError as exc:
            m = _re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if m is None or m.group(1) not in env_meta["env_kwargs"]:
                raise
            dropped.append(env_meta["env_kwargs"].pop(m.group(1)))
            print(f"[env] dropped unsupported kwarg {m.group(1)!r}")


def render_cameras(env_wrapper, base_env, camera_names, camera_h, camera_w):
    base_env.sim.forward()
    return {
        cam: env_wrapper.render(
            mode="rgb_array", height=camera_h, width=camera_w, camera_name=cam
        )
        for cam in camera_names
    }


def install_object_registry_fallback() -> None:
    """Fall back to a same-category instance when a recorded object is gone.

    atomic_new's ep_meta records the exact object mjcf_path the mg HDF5 used,
    resolved by BARX's robocasa_x install. This standalone robocasa checkout's
    object catalog is close but not identical -- some instances (e.g.
    objaverse/spoon/spoon_16) exist on disk but are not registered, so the
    deterministic reverse-lookup in sample_kitchen_object_helper raises
    ValueError and kills the whole episode. Since this render only needs the
    object's *category* to stay correct (the language description says
    "spoon", not "spoon_16"), retry with the recorded category instead of
    aborting -- deterministic seeding (per-episode rng) still keeps the choice
    reproducible.
    """
    import robocasa.models.objects.kitchen_object_utils as KOU

    if getattr(KOU, "_registry_fallback_installed", False):
        return
    original = KOU.sample_kitchen_object_helper

    def sample_kitchen_object_helper(groups, *args, **kwargs):
        try:
            return original(groups, *args, **kwargs)
        except ValueError:
            if not (isinstance(groups, str) and groups.endswith(".xml")):
                raise
            # groups_containing_sampled_obj / cat come from the caller's cfg
            # only via closure in create_obj, which we don't have here; the
            # category is the mjcf_path's parent-of-parent directory name
            # (.../objects/<registry-ish>/<category>/<instance>/model.xml).
            cat_guess = Path(groups).parent.parent.name
            print(
                f"[obj] {groups!r} not in this robocasa's registry, "
                f"resampling from category {cat_guess!r}"
            )
            return original([cat_guess], *args, **kwargs)

    KOU.sample_kitchen_object_helper = sample_kitchen_object_helper
    KOU._registry_fallback_installed = True


def install_recorded_camera_override() -> None:
    """Render from the cameras the episode recorded, not freshly derived ones.

    Kitchen._load_model calls CamUtils.set_cameras(self), which rebuilds
    `_cam_configs` from CamUtils.get_robot_cam_configs(robot_name) -- a
    per-robot canonical pose, not what ep_meta["cam_configs"] stored. Fine when
    the source used those canonical poses (the original v1.0 workflow), wrong
    for atomic_new, whose front camera is the per-episode randomized view the
    mg HDF5s rendered. Overlaying the recorded configs after CamUtils.set_cameras
    runs reproduces them.
    """
    import robocasa.utils.camera_utils as CamUtils

    if getattr(CamUtils, "_recorded_camera_hooked", False):
        return
    original = CamUtils.set_cameras

    def set_cameras(env):
        original(env)
        recorded = (getattr(env, "_ep_meta", None) or {}).get("cam_configs") or {}
        for name, cfg in recorded.items():
            if name in env._cam_configs:
                env._cam_configs[name] = deepcopy(cfg)

    CamUtils.set_cameras = set_cameras
    # kitchen.py imported `set_cameras` into its own module namespace via
    # `from . import camera_utils as CamUtils`, so patching the function
    # object on the CamUtils module (not a rebind of the name) is visible
    # through kitchen.py's `CamUtils.set_cameras(self)` call as well.
    CamUtils._recorded_camera_hooked = True


def replay_embodiment(
    env_wrapper,
    initial_state,
    ep_meta,
    is_source_robot,
    raw_actions,
    hide_scope,
    fly_base_offset,
    fly_spacing,
    settle_steps,
    ref_fingertip_pos_seq,
    ref_site_xmat_seq,
    reference_opening_axis,
    match_orientation,
    camera_h,
    camera_w,
    robot_name=None,
):
    """Replay one embodiment's trajectory.

    For the source (Panda) robot, this is a plain open-loop replay of the
    original raw actions -- it defines the reference grip-site path (its
    live-measured fingertip/pad-midpoint position becomes
    `ref_fingertip_pos_seq` and its gripper opening axis becomes
    `reference_opening_axis` for every other embodiment).

    For every other embodiment, an IK correction snaps the arm every step so
    its own fingertip/pad midpoint -- not just its eef *site*, which sits a
    different, robot-specific distance from the actual grip point (Panda
    3.6mm, UR5e 14.3mm, IIWA 22.2mm -- IK-matching sites alone left the real
    grip point off by up to ~1.9cm cross-embodiment) -- lands exactly on the
    reference's fingertip position (Jacobian-transported IK, see
    solve_ik_to_eef_pose's fingertip_offset_local). Per-step (not just at
    t=0) because open-loop replay of the same raw actions was found
    empirically to drift up to ~90cm from the reference by the end of a
    236-step RoboCasa Omron-base trajectory. The IK target *orientation* is
    additionally twisted by this robot's own gripper-opening-axis offset
    from the reference (see robotiq85_mesh_swap_note()), so the visible
    pad/finger orientation matches Panda's too.
    """
    base_env = env_wrapper.env
    if is_source_robot:
        obs = env_wrapper.reset_to(initial_state)
    else:
        # Every other embodiment must rebuild its OWN model (different robot
        # geometry/collision), reusing the SAME layout_id/style_id/objects
        # via ep_meta so the kitchen scene itself stays identical -- this is
        # the same reconstruct-from-ep_meta trick used for the mg->atomic
        # LeRobot conversion earlier this session (kitchen.py:591-608).
        base_env.set_ep_meta(dict(ep_meta))
        obs = env_wrapper.reset(unset_ep_meta=False)

    jpos_idx = np.asarray(base_env.robots[0]._ref_joint_pos_indexes, dtype=np.int64)

    if not is_source_robot and robot_name in KITCHEN_INIT_QPOS:
        # Seed the arm's starting joint configuration with a known
        # "kitchen-ready" pose instead of the robot's raw default reset
        # pose. DLS IK is a *local* redundancy-resolution method warm-started
        # from the current qpos -- an unlucky t=0 seed can converge to a
        # valid-but-contorted ("elbow-flipped") joint branch that then
        # persists/propagates through the rest of the per-step-aligned
        # episode. See jpos_idx freeze below for the other half of this fix.
        base_env.sim.data.qpos[jpos_idx] = KITCHEN_INIT_QPOS[robot_name]
        base_env.sim.data.qvel[:] = 0.0
        base_env.sim.forward()

    this_opening_axis = measure_opening_axis(base_env)
    this_fingertip_offset = measure_fingertip_offset_local(base_env)
    twist = (
        np.eye(3)
        if is_source_robot
        else build_twist_to_reference_axis(this_opening_axis, reference_opening_axis)
    )

    def target_xmat(t):
        return ref_site_xmat_seq[t] @ twist.T

    def align(t):
        solve_ik_to_eef_pose(
            base_env,
            ref_fingertip_pos_seq[t],
            target_xmat(t),
            match_orientation=match_orientation,
            fingertip_offset_local=this_fingertip_offset,
        )
        base_env.sim.forward()
        if match_orientation:
            # Orientation-matched IK can fail to reach the target position when
            # this embodiment's arm geometry makes the exact orientation
            # unreachable there (seen empirically with JacoOmron: up to ~100mm
            # position error while chasing an orientation that's just out of
            # reach, even though the position itself is reachable on its own).
            # Grip-site position tracking matters more than exact visual twist,
            # so fall back to position-only IK whenever the orientation-matched
            # solve leaves more than 5mm of position error.
            fp, _ = fingertip_world_pose(base_env, this_fingertip_offset)
            if np.linalg.norm(fp - ref_fingertip_pos_seq[t]) > 5e-3:
                solve_ik_to_eef_pose(
                    base_env,
                    ref_fingertip_pos_seq[t],
                    target_xmat(t),
                    match_orientation=False,
                    fingertip_offset_local=this_fingertip_offset,
                )
                base_env.sim.forward()
        base_env.sim.data.qvel[:] = 0.0
        base_env.sim.forward()

    if not is_source_robot:
        align(0)

    if hide_scope in ("all_objects", "robot_only"):
        fly_objects_away(base_env, fly_base_offset, fly_spacing)
    if hide_scope == "robot_only":
        hide_kitchen_background(base_env)

    # Zero arm/base motion but keep the demo's actual first gripper command
    # (the controller's "GRIP" gripper is an absolute state, not a delta, so
    # settling with gripper=0 would issue a spurious half-open/close command).
    settle_action = np.zeros(raw_actions.shape[1])
    settle_action[
        ACTION_KEY_ORDERING_HDF5["gripper_close"][0] : ACTION_KEY_ORDERING_HDF5[
            "gripper_close"
        ][1]
    ] = raw_actions[
        0,
        ACTION_KEY_ORDERING_HDF5["gripper_close"][0] : ACTION_KEY_ORDERING_HDF5[
            "gripper_close"
        ][1],
    ]
    for _ in range(settle_steps):
        obs, _, _, _ = env_wrapper.step(settle_action)
    if not is_source_robot:
        # Settling (gravity comp / controller warm-up) can itself perturb the
        # arm away from the aligned pose -- re-correct once more before t=0.
        align(0)

    buf = {
        "images": {cam: [] for cam in env_wrapper.camera_names},
        "eef_pos": [],
        "eef_quat": [],
        "fingertip_pos": [],
        "opening_axis": this_opening_axis,
        "fingertip_offset": this_fingertip_offset,
    }
    for t in range(raw_actions.shape[0]):
        if not is_source_robot:
            align(t)
        images = render_cameras(
            env_wrapper, base_env, env_wrapper.camera_names, camera_h, camera_w
        )
        for cam in env_wrapper.camera_names:
            buf["images"][cam].append(images[cam])
        pos, xmat = eef_world_pose(base_env)
        fingertip_pos, _ = fingertip_world_pose(base_env, this_fingertip_offset)
        buf["eef_pos"].append(pos)
        buf["eef_quat"].append(xmat)
        buf["fingertip_pos"].append(fingertip_pos)
        if is_source_robot:
            obs, _, _, _ = env_wrapper.step(raw_actions[t])
        else:
            # env_wrapper.step() runs real OSC-controller physics to advance
            # gripper/base state, which can physically perturb the arm's
            # joint qpos away from the clean IK-converged configuration
            # `align(t)` just produced. Left alone, that perturbed qpos
            # becomes the seed for next iteration's *local* DLS IK solve --
            # a bad seed can converge to a different (flipped-elbow) but
            # still-locally-valid solution branch. Freeze/restore the arm
            # qpos across the step so stepping only ever affects
            # gripper/base state, never the next align()'s IK seed.
            arm_qpos = base_env.sim.data.qpos[jpos_idx].copy()
            obs, _, _, _ = env_wrapper.step(raw_actions[t])
            base_env.sim.data.qpos[jpos_idx] = arm_qpos
            base_env.sim.data.qvel[:] = 0.0
            base_env.sim.forward()
    return buf


def load_episode(source_dir: Path, episode_index: int):
    import pandas as pd

    ep_dir = source_dir / "extras" / f"episode_{episode_index:06d}"
    states = np.load(ep_dir / "states.npz")["states"]
    ep_meta = json.loads((ep_dir / "ep_meta.json").read_text())
    with gzip.open(ep_dir / "model.xml.gz") as f:
        model_xml = f.read().decode("utf-8")
    env_args = json.loads((source_dir / "extras" / "dataset_meta.json").read_text())[
        "env_args"
    ]

    with open(source_dir / "meta" / "episodes.jsonl") as f:
        episode_lengths = {}
        for line in f:
            row = json.loads(line)
            episode_lengths[row["episode_index"]] = row["length"]
    length = episode_lengths[episode_index]

    chunk = episode_index // 1000
    parquet_path = (
        source_dir
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    df = pd.read_parquet(parquet_path)
    action_lerobot = np.stack(df["action"].to_numpy()).astype(np.float64)
    raw_actions = lerobot_action_to_raw(action_lerobot)

    assert len(states) == length == len(raw_actions), (
        len(states),
        length,
        len(raw_actions),
    )
    return states, raw_actions, ep_meta, model_xml, env_args


def save_preview(
    output_dir: Path, episode_index: int, robots, cam, results, max_frames=8
):
    import imageio.v2 as imageio

    num_steps = len(results[robots[0]]["images"][cam])
    frame_idx = np.linspace(0, num_steps - 1, min(max_frames, num_steps), dtype=int)
    rows = []
    for robot in robots:
        frames = [results[robot]["images"][cam][i] for i in frame_idx]
        rows.append(np.concatenate(frames, axis=1))
    grid = np.concatenate(rows, axis=0)
    out_path = output_dir / f"episode_{episode_index:06d}_{cam}_preview.jpg"
    imageio.imwrite(out_path, grid)
    print(f"[preview] wrote {out_path}")


def save_videos(output_dir: Path, episode_index: int, robots, cameras, results):
    import imageio_ffmpeg

    for robot in robots:
        for cam in cameras:
            frames = results[robot]["images"][cam]
            video_path = output_dir / f"episode_{episode_index:06d}_{robot}_{cam}.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(video_path),
                size=(frames[0].shape[1], frames[0].shape[0]),
                fps=FPS,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                # keep the exact render size (default pads to a 16px grid)
                macro_block_size=1,
                output_params=["-crf", "23"],
            )
            writer.send(None)
            try:
                for frame in frames:
                    writer.send(np.ascontiguousarray(frame))
            finally:
                writer.close()
            print(f"[video] wrote {video_path}")


def save_hdf5(output_dir: Path, episode_index: int, robots, raw_actions, results):
    path = output_dir / f"episode_{episode_index:06d}.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["robots"] = json.dumps(robots)
        f.create_dataset("actions", data=raw_actions)
        for robot in robots:
            grp = f.create_group(robot)
            grp.create_dataset("eef_pos", data=np.stack(results[robot]["eef_pos"]))
            grp.create_dataset("eef_xmat", data=np.stack(results[robot]["eef_quat"]))
            grp.create_dataset(
                "fingertip_pos", data=np.stack(results[robot]["fingertip_pos"])
            )
            grp.attrs["fingertip_offset_mm"] = float(
                np.linalg.norm(results[robot]["fingertip_offset"]) * 1000
            )
    print(f"[hdf5] wrote {path}")


def _mat2quat(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (x, y, z, w) quaternion."""
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


CHUNK_SIZE = 1000


class CrossEmbodimentLerobotWriter:
    """LeRobot v2.1 dataset writer for the visual-robust cross-embodiment
    replay, structurally modeled on shortcut-learning-in-grps's
    build_empty_scene_lerobot_dataset(): every embodiment gets its own
    tagged observation.images.<tag>/observation.wrist_image.<tag> video
    track within the SAME episode/frame, plus one shared action/state/
    reward/done track from the source (reference) embodiment (all
    embodiments replay literally the same raw action sequence by
    construction, so a per-embodiment action column would be redundant).
    Not a byte-identical schema to the LIBERO dataset (RoboCasa's state
    representation is fundamentally different) or to v1.0's own atomic
    schema (this is a derived, multi-embodiment product, not a single-robot
    demo) -- see claude.txt for the exact feature list."""

    def __init__(
        self,
        out_dir: Path,
        tags: list[str],
        camera_names: list[str],
        camera_h: int,
        camera_w: int,
    ):
        self.out_dir = out_dir
        self.tags = tags
        self.camera_names = camera_names
        self.camera_h = camera_h
        self.camera_w = camera_w
        self.data_dir = out_dir / "data"
        self.video_dir = out_dir / "videos"
        self.meta_dir = out_dir / "meta"
        for d in (self.data_dir, self.video_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.task_to_id: dict[str, int] = {}
        self.tasks_f = open(self.meta_dir / "tasks.jsonl", "a")
        self.episodes_f = open(self.meta_dir / "episodes.jsonl", "a")
        self.episode_index = 0
        self.total_frames = 0

    def _task_id(self, lang: str) -> int:
        if lang not in self.task_to_id:
            idx = len(self.task_to_id)
            self.task_to_id[lang] = idx
            self.tasks_f.write(json.dumps({"task_index": idx, "task": lang}) + "\n")
            self.tasks_f.flush()
        return self.task_to_id[lang]

    def add_episode(self, results: dict, raw_actions: np.ndarray, lang: str):
        import pandas as pd
        import imageio_ffmpeg

        ep_idx = self.episode_index
        ep_len = raw_actions.shape[0]
        task_idx = self._task_id(lang)
        chunk = ep_idx // CHUNK_SIZE

        ref_tag = self.tags[0]
        ref = results[ref_tag]
        state = np.concatenate(
            [
                np.stack(ref["fingertip_pos"]),
                np.stack([_mat2quat(r) for r in ref["eef_quat"]]),
                raw_actions[
                    :,
                    ACTION_KEY_ORDERING_HDF5["gripper_close"][
                        0
                    ] : ACTION_KEY_ORDERING_HDF5["gripper_close"][1],
                ],
            ],
            axis=1,
        )
        frame_index = np.arange(ep_len, dtype=np.int64)
        timestamp = (frame_index / FPS).astype(np.float32)
        columns = {
            "observation.state": list(state.astype(np.float32)),
            "action": list(raw_actions.astype(np.float32)),
            "next.reward": [np.array([0.0], dtype=np.float32)] * ep_len,
            "next.done": [
                np.array([t == ep_len - 1], dtype=bool) for t in range(ep_len)
            ],
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": np.full(ep_len, ep_idx, dtype=np.int64),
            "index": frame_index + self.total_frames,
            "task_index": np.full(ep_len, task_idx, dtype=np.int64),
        }
        df = pd.DataFrame(columns)
        data_chunk_dir = self.data_dir / f"chunk-{chunk:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_chunk_dir / f"episode_{ep_idx:06d}.parquet")

        for tag in self.tags:
            for cam in self.camera_names:
                frames = results[tag]["images"][cam]
                video_chunk_dir = self.video_dir / f"chunk-{chunk:03d}" / f"{tag}.{cam}"
                video_chunk_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_chunk_dir / f"episode_{ep_idx:06d}.mp4"
                writer = imageio_ffmpeg.write_frames(
                    str(video_path),
                    size=(frames[0].shape[1], frames[0].shape[0]),
                    fps=FPS,
                    codec="libx264",
                    pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p",
                    # keep the exact render size (default pads to a 16px grid)
                    macro_block_size=1,
                    output_params=["-crf", "23"],
                )
                writer.send(None)
                try:
                    for frame in frames:
                        writer.send(np.ascontiguousarray(frame))
                finally:
                    writer.close()

        self.episodes_f.write(
            json.dumps({"episode_index": ep_idx, "tasks": [lang], "length": ep_len})
            + "\n"
        )
        self.episodes_f.flush()
        self.episode_index += 1
        self.total_frames += ep_len

    def finalize(self, source_robot: str):
        self.tasks_f.close()
        self.episodes_f.close()

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["fingertip_xyz(3)+quat_xyzw(4)+gripper_close(1)"],
            },
            "action": {
                "dtype": "float32",
                "shape": [12],
                "names": ["arm_gripper_base_torso_mode_v1"],
            },
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }
        for tag in self.tags:
            for cam in self.camera_names:
                features[f"observation.images.{tag}.{cam}"] = {
                    "dtype": "video",
                    "shape": [self.camera_h, self.camera_w, 3],
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": FPS,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
        info = {
            "codebase_version": "v2.1",
            "robot_type": f"cross_embodiment[{','.join(self.tags)}]",
            "source_robot": source_robot,
            "total_episodes": self.episode_index,
            "total_frames": self.total_frames,
            "total_tasks": len(self.task_to_id),
            "total_videos": self.episode_index
            * len(self.tags)
            * len(self.camera_names),
            "total_chunks": (max(self.episode_index - 1, 0) // CHUNK_SIZE) + 1,
            "chunks_size": CHUNK_SIZE,
            "fps": FPS,
            "splits": {"train": f"0:{self.episode_index}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "embodiment_tags": self.tags,
        }
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)


class _EnvWithCameras:
    """Thin wrapper so replay_embodiment can read/track which cameras exist
    without re-deriving them from obs keys each call."""

    def __init__(self, env, camera_names):
        self._env = env
        self.camera_names = camera_names

    def __getattr__(self, item):
        return getattr(self._env, item)


def process_episode(
    args, source_env_meta, episode_index: int, writer=None, save_examples: bool = False
):
    states, raw_actions, ep_meta, model_xml, _env_args = load_episode(
        args.source_lerobot_dir, episode_index
    )
    initial_state = {
        "states": states[0],
        "model": model_xml,
        "ep_meta": json.dumps(ep_meta),
    }
    source_robot = source_env_meta["env_kwargs"]["robots"]
    if isinstance(source_robot, (list, tuple)):
        source_robot = source_robot[0]

    if source_robot not in args.robots:
        raise ValueError(
            f"--robots must include the source robot {source_robot!r} to define the reference trajectory"
        )

    results = {}
    ref_fingertip_pos_seq = None
    ref_site_xmat_seq = None
    reference_opening_axis = None
    # Source robot first: its per-step fingertip position (and site
    # orientation, gripper opening axis) is the target every other
    # embodiment gets IK-corrected to at each step.
    ordered_robots = [source_robot] + [r for r in args.robots if r != source_robot]
    for robot in ordered_robots:
        env = create_env(
            source_env_meta,
            robot,
            args.camera_names,
            args.camera_height,
            args.camera_width,
            args.osc_kp,
        )
        env = _EnvWithCameras(env, args.camera_names)
        try:
            buf = replay_embodiment(
                env,
                initial_state,
                ep_meta,
                robot == source_robot,
                raw_actions,
                args.hide_scope,
                args.fly_base_offset,
                args.fly_spacing,
                args.settle_steps,
                ref_fingertip_pos_seq,
                ref_site_xmat_seq,
                reference_opening_axis,
                args.match_orientation,
                args.camera_height,
                args.camera_width,
                robot_name=robot,
            )
        finally:
            env.env.close()
        results[robot] = buf
        if robot == source_robot:
            ref_fingertip_pos_seq = buf["fingertip_pos"]
            ref_site_xmat_seq = buf["eef_quat"]
            reference_opening_axis = buf["opening_axis"]
        print(
            f"[episode {episode_index}] {robot}: {raw_actions.shape[0]} steps replayed "
            f"(opening_axis={np.round(buf['opening_axis'], 3)}, "
            f"fingertip_offset={np.round(np.linalg.norm(buf['fingertip_offset'])*1000, 1)}mm)"
        )

    if writer is not None:
        writer.add_episode(results, raw_actions, ep_meta["lang"])

    if save_examples:
        examples_dir = args.output_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        for cam in args.camera_names:
            save_preview(
                examples_dir,
                episode_index,
                args.robots,
                cam,
                results,
                args.preview_frames,
            )
        save_videos(
            examples_dir, episode_index, args.robots, args.camera_names, results
        )
        save_hdf5(examples_dir, episode_index, args.robots, raw_actions, results)


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    patch_python_paths()
    args = parse_args()

    import robocasa  # noqa: F401 - registers RoboCasa envs

    install_recorded_camera_override()
    install_object_registry_fallback()

    env_args = json.loads(
        (args.source_lerobot_dir / "extras" / "dataset_meta.json").read_text()
    )["env_args"]
    source_robot = env_args["env_kwargs"]["robots"]
    # convert_mg_to_atomic_new.py records robots as a one-element list
    # (["IIWAOmron"]) instead of the bare string v1.0's env_args use.
    if isinstance(source_robot, (list, tuple)):
        source_robot = source_robot[0]
    if source_robot not in args.robots:
        raise ValueError(f"--robots must include the source robot {source_robot!r}")
    ordered_robots = [source_robot] + [r for r in args.robots if r != source_robot]

    writer = None
    if args.write_lerobot:
        writer = CrossEmbodimentLerobotWriter(
            args.output_dir / "lerobot",
            ordered_robots,
            args.camera_names,
            args.camera_height,
            args.camera_width,
        )

    episodes = (
        args.episodes if args.episodes is not None else list(range(args.num_episodes))
    )
    for i, episode_index in enumerate(episodes):
        process_episode(
            args,
            env_args,
            episode_index,
            writer=writer,
            save_examples=(i < args.num_example_episodes),
        )

    if writer is not None:
        writer.finalize(source_robot)
        print(
            f"[lerobot] wrote {writer.episode_index} episodes -> {args.output_dir / 'lerobot'}"
        )


if __name__ == "__main__":
    main()
