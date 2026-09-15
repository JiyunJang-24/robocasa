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
from collections import namedtuple
import contextlib
from copy import deepcopy
import gzip
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# robocasa_x variant: atomic_new (mg -> atomic_new conversion) was built with
# BARX's robocasa_x, whose Kitchen/fixture/object registries are forks of the
# standalone robocasa this script originally targeted -- close enough to look
# similar, different enough that fixture names (e.g. "counter_left_group" vs
# "counter_1_left_left_group") and object registries (spoon_16 missing, etc.)
# do not line up. Reconstructing an atomic_new episode's ep_meta needs the
# exact codebase that wrote it. Run under barx's own .venv (mujoco==3.1.1,
# numpy==1.23.5 -- the versions robocasa_x's import guards assert).
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("ROBOCASA_ROOT", REPO_ROOT))
ROBOCASA_X_ROOT = Path(
    os.environ.get("ROBOCASA_X_ROOT", ROOT / "third_party" / "barx" / "robocasa_x")
)
# Deliberately NOT putting /data1/local/robosuite on sys.path. That checkout
# carries an undocumented local patch that swaps Robotiq85Gripper's MJCF for
# a mujoco_menagerie mesh (grippers/robotiq_2f85_menagerie/2f85_robosuite.xml,
# one tendon-coupled actuator), which does NOT match what atomic_new's
# UR5eOmron/PandaOmron recordings actually used: the STOCK Robotiq85 (grippers/
# robotiq_gripper_85.xml, root body "robotiq_85_adapter_link", TWO
# independent position actuators -- confirmed directly off
# PandaOmron/TurnOnSinkFaucet's own stored model.xml.gz).
#
# Mixing the two seemed necessary at first only because /data1/local/robosuite
# also ships per-robot composite-controller config files (default_ur5eomron
# .json / default_iiwaomron.json) that barx's own pinned robosuite package
# lacks (it only has default_pandaomron.json) -- but pulling in that checkout
# for the config files ALSO pulls in its patched Robotiq85Gripper class,
# and the two are incompatible together: a scaffold built with the patched
# (1-actuator) class against a stored/replayed scene that expects the stock
# (2-actuator) one raises geom-name and control-dim errors that no amount of
# monkeypatching the gripper controller ever fully resolved (tried three
# different approaches; each fixed one robot and broke another). The correct
# fix turns out to be simpler: convert_mg_to_atomic_new.py -- the script that
# actually built atomic_new -- never touches /data1/local/robosuite either
# (it only adds ROBOCASA_X_ROOT), and successfully drove this exact stock
# gripper for all 2900 atomic_new episodes. Matching that exactly (barx's own
# robosuite only) reproduces the same success here: every one of Panda-source,
# UR5e-as-alternate, and IIWA-as-alternate opens/closes correctly (measured
# ~81mm -> ~36mm) with zero gripper-controller patches. The per-robot
# controller config gap is closed by target_controller_config() below reading
# each alternate robot's OWN atomic_new recording instead of a file lookup.
ATOMIC_NEW_ROOT = Path(
    os.environ.get(
        "ATOMIC_NEW_ROOT", ROOT / "third_party" / "barx" / "datasets" / "atomic_new"
    )
)
ATOMIC_NEW_DATASET = {
    "PandaOmron": ATOMIC_NEW_ROOT / "PandaOmron/pretrain/TurnOnSinkFaucet/lerobot",
    "UR5eOmron": ATOMIC_NEW_ROOT / "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
    "IIWAOmron": ATOMIC_NEW_ROOT / "IIWAOmron/pretrain/PnPCounterToSink/lerobot",
}

SETTLE_STEPS = 10
CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

# A frame counts as "reachable" for an embodiment when that arm got its
# fingertip to the reference position AND within this much of the reference
# gripper orientation. 5mm matches align()'s own position fallback threshold.
# 15deg is deliberately loose: a parallel-jaw gripper's useful pose is mostly
# defined by its approach direction and the pads' opening axis (the latter is
# already compensated per-embodiment by build_twist_to_reference_axis), so
# small residual twist does not change what the pose means. Large arms
# (UR5e/Panda/IIWA/Jaco) sit at ~0deg; the short arms (PiPER/YAM/VX300S) are
# the ones this actually discriminates.
# Every arm's IK stays inside its mechanical joint limits.
#
# Letting the wrist travel past its stop was tried, because on paper it buys
# orientation accuracy on the short arms and my own metrics said it was free:
# swept over all eight arms it reported 0.00mm of non-adjacent
# self-intersection and better on-pose rates (Sawyer 80.7% -> 100.0%,
# PiPER 6.8% -> 23.9%). Reviewing the rendered frames showed otherwise --
# out-of-limit poses visibly break the arm. The penetration check simply did
# not model the failure: it only looked at non-adjacent link pairs above a
# 2mm depth, so breakage from adjacent links overlapping, or from a joint
# wrapping several turns past its stop, passed it clean.
#
# Keep this as the standing lesson: a geometric proxy that says "safe" is not
# evidence the render looks right. Joint limits are also what the real
# hardware can do, so respecting them keeps the data transferable.
SELF_PENETRATION_TOL_M = 2e-3
REACH_POS_TOL_M = 5e-3
REACH_ROT_TOL_DEG = 15.0
# Tolerances the commanded eef pose must be hit within for a frame to count as
# produced by this embodiment. Tighter than the reporting tolerances above:
# these gate what goes in, those describe what came out.
POSE_POS_TOL_M = 5e-3
POSE_ROT_TOL_DEG = 5.0

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
    # Solved the same way as UR5e's: DLS-IK onto PandaOmron's initial
    # base-relative eef pose, scored on accuracy + self-collision, with joint
    # limits enforced (see solve_ik_to_eef_pose). Sawyer/VX300S match the
    # reference pose outright; PiPER and YAM are small arms that cannot also
    # satisfy the reference *orientation* there (51deg / 106deg residual at
    # best), so theirs come from the position-only solve -- which is exactly
    # what align() falls back to per-step for them anyway.
    "SawyerOmron": np.array(
        [-2.432545, -1.152041, -2.738516, 2.136089, -2.368845, -1.175039, 3.520376]
    ),
    "VX300SMobile": np.array(
        [0.027385, -0.718828, 0.767593, 3.14158, -0.032299, -1.640426]
    ),
    "PiPEROmron": np.array([-0.307048, 0.610669, -1.317907, 0.2755, 1.2028, 2.592105]),
    "YAMOmron": np.array(
        [0.522755, 0.870244, 1.815368, -1.455236, 0.455017, -2.019456]
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
    for path in (str(ROOT), str(ROBOCASA_X_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def install_mesh_inertia_compat():
    """Let this MuJoCo build load XMLs that set `inertia` on a <mesh>.

    `<mesh inertia="shell|legacy|exact|convex">` only exists from MuJoCo
    3.1.4; the venv this script runs under pins 3.1.1, which rejects it as a
    schema violation outright. Several robots/grippers we want to render ship
    it -- notably Sawyer's own robot.xml (so SawyerOmron could not be built at
    all) and robosuite_models' aloha/pr2/umi/yumi/schunk grippers.

    The attribute only selects how MuJoCo derives a mesh's inertia tensor, and
    every affected body here already declares an explicit <inertial>, so
    dropping it changes nothing about the rendered scene. Strip it in
    MujocoXML.__init__ rather than at compile time: robosuite merges robot,
    gripper, base and arena trees together and both get_model() (compile) and
    get_xml() (the string handed to the sim / stored per episode) read from
    the merged tree, so cleaning each file as it is parsed covers every path.
    A no-op on newer MuJoCo, which is probed for directly.
    """
    import mujoco
    from robosuite.models.base import MujocoXML

    if getattr(MujocoXML, "_mesh_inertia_compat", False):
        return
    try:
        mujoco.MjModel.from_xml_string(
            '<mujoco><asset><mesh name="probe" inertia="shell" '
            'vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset></mujoco>'
        )
        MujocoXML._mesh_inertia_compat = True
        return
    except ValueError:
        pass

    original_init = MujocoXML.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for mesh in self.root.iter("mesh"):
            mesh.attrib.pop("inertia", None)

    MujocoXML.__init__ = __init__
    MujocoXML._mesh_inertia_compat = True


def robotiq85_mesh_swap_note():
    """Why the IK target orientation gets twisted per embodiment.

    Panda's native gripper and IIWA's Robotiq140 don't share Robotiq85's
    opening-axis convention -- each gripper's fingers close along a
    different local axis relative to its own eef site, so IK-matching the
    site *orientation* alone still leaves the visible pad/finger orientation
    rotated relative to Panda's reference. measure_opening_axis()/
    build_twist_to_reference_axis() below live-measure each robot's actual
    gripper opening axis in its own eef-site frame and fold a compensating
    Rz twist into the IK *target* orientation each step, so the finger pads
    visually line up across embodiments regardless of which gripper model
    each one is holding.

    (Earlier in this script's history this docstring also covered a
    Robotiq85-mesh incompatibility -- /data1/local/robosuite's local patch
    swaps Robotiq85Gripper for a 1-actuator mujoco_menagerie mesh instead of
    the 2-actuator stock one atomic_new actually recorded. That's resolved
    now by simply never importing from /data1/local/robosuite at all -- see
    the ROOT-level comment above -- not by anything in this function.)"""


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
    if ids:
        return ids
    # Not every gripper names its contact patches "*pad*": AlohaGripper (the
    # VX300S/ViperX default) only has `*_left_finger_g0..g4` /
    # `*_right_finger_g0..g4`. With no match at all measure_opening_axis
    # silently returned its [0,1,0] fallback, which would then be fed to
    # build_twist_to_reference_axis as if it were a real measurement. Fall
    # back to the finger collision geoms instead, which carry the same
    # left/right split.
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ""
        low = name.lower()
        if (
            name.startswith("gripper0")
            and "finger" in low
            and "visual" not in low
            and not low.endswith("_vis")
        ):
            ids.append((gid, name))
    return ids


def _pad_side(name: str) -> str:
    """Which jaw ('l'/'r') a gripper pad/finger geom belongs to.

    The arm-side prefix has to come off first. Every geom of a right-arm
    gripper is called `gripper0_right_...`, so testing the raw name for
    right/left-ness makes *everything* look right-sided; and a bare `l_`/`r_`
    token (RethinkGripper's `l_fingerpad`/`r_fingerpad`, Sawyer's default
    gripper) doesn't contain the substring "left" at all. Both together meant
    both of Sawyer's pads landed in the same group, so measure_opening_axis
    hit its `len(groups) < 2` fallback and reported a fabricated axis.
    Tokenising the suffix handles `left`/`l`/`finger2` alike, and was
    verified to leave every already-rendered gripper's measured axis
    bit-identical (Robotiq85/Robotiq140/PandaGripper on UR5e/IIWA/Panda/Jaco).
    """
    body = name
    for prefix in ("gripper0_right_", "gripper0_left_", "gripper0_"):
        if body.startswith(prefix):
            body = body[len(prefix) :]
            break
    low = body.lower()
    if set(low.split("_")) & {"left", "l"} or "finger2" in low:
        return "l"
    return "r"


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
        groups.setdefault(_pad_side(name), []).append(sim.data.geom_xpos[gid].copy())
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
        "--robots",
        nargs="+",
        default=["PandaOmron", "UR5eOmron", "IIWAOmron"],
        help="each entry is either a bare robot class ('UR5eOmron', tag == class, "
        "gripper from ATOMIC_NEW_GRIPPER) or 'Tag:RobotClass:Gripper' to render "
        "the same robot class under a different gripper/tag (e.g. "
        "'PandaOmronPandaGripper:PandaOmron:default' alongside plain 'PandaOmron' "
        "to get both grippers in one episode).",
    )
    parser.add_argument(
        "--style-sweep",
        action="store_true",
        help="Override each episode's recorded kitchen style with "
        "episode_index %% 12, cycling all of RoboCasa's styles for "
        "background variety instead of only the handful the source "
        "recording happened to use. Style is very nearly "
        "appearance-only: measured against a same-style control "
        "(0.000mm), switching style leaves counter height identical "
        "and moves the sink by 14-23mm. That is ~6px at this render "
        "size and does not matter here because --hide-scope "
        "all_objects flies the task objects away, so the replay is a "
        "trajectory through a kitchen rather than a real manipulation. "
        "Layout is deliberately NOT swept: it rearranges fixtures "
        "wholesale, which would leave the recorded trajectory reaching "
        "into empty space.",
    )
    parser.add_argument(
        "--hide-scope",
        choices=["all_objects", "robot_only", "no_kitchen", "none"],
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


# Kitchen furniture, as distinct from the room itself. Hiding everything
# non-robot (hide_kitchen_background) leaves a bare skybox, which is a single
# flat background -- useless when the point is background variety. Hiding only
# the furniture keeps walls/floor/windows, so the arm still stands in a room
# that style-sweep and tinting can then vary.
_FURNITURE_PREFIXES = (
    "stack",
    "cab",
    "counter",
    "side",
    "bottom",
    "fridge",
    "top",
    "container",
    "distr",
    "microwave",
    "sink",
    "stovetop",
    "utensil",
    "oven",
    "micro",
    "coffee",
    "stool",
    "shelves",
    "plant",
    "toaster",
    "hood",
    "obj",
    "dishwasher",
)
_BACKDROP_PREFIXES = ("wall", "floor", "window")


def hide_kitchen_furniture(base_env):
    """Remove the kitchen but keep the room.

    Whitelist, not blacklist. Enumerating furniture prefixes to hide was tried
    first and leaked: props that did not match any listed prefix (a paper
    towel roll) stayed visible, and with the counter gone they floated in
    mid-air. Keeping only the room shell -- walls, floor, windows, lights --
    and hiding everything else means anything new to the scene is hidden by
    default rather than silently shipped.
    """
    model = base_env.sim.model
    for gid in range(model.ngeom):
        name = model.geom_id2name(gid) or ""
        if name.startswith(_ROBOT_GEOM_PREFIXES):
            continue
        if name.split("_")[0] not in _BACKDROP_PREFIXES:
            model.geom_rgba[gid, 3] = 0.0
            # Take the geometry out of collision too, not just out of sight.
            # Alpha alone left the arm bumping into invisible counters: poses
            # were rejected for hitting furniture the viewer cannot see, which
            # both shrank the usable workspace and showed up as frozen frames
            # (worst on the longest arm, IIWA+Robotiq140 at 21%). The room
            # shell stays in the whitelist, so the base still rests on a floor.
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0


def randomize_backdrop(base_env, rng):
    """Tint the room so the kitchen-less background is not one flat colour.

    Style-sweep already swaps wall/floor textures, but only 12 ways; a
    per-episode tint on top gives each episode a distinct room without
    needing new assets.
    """
    model = base_env.sim.model
    # Multiplicative tint kept above ~0.6: sampling down to 0.35 produced
    # near-black rooms, which carry no usable background signal. Allowing >1
    # brightens as well as darkens so the spread is around the base texture
    # rather than only below it.
    tint = np.array(
        [rng.uniform(0.62, 1.35), rng.uniform(0.62, 1.35), rng.uniform(0.62, 1.35), 1.0]
    )
    for gid in range(model.ngeom):
        name = model.geom_id2name(gid) or ""
        if name.startswith(_ROBOT_GEOM_PREFIXES):
            continue
        if name.split("_")[0] in _BACKDROP_PREFIXES:
            base = model.geom_rgba[gid].copy()
            if base[3] <= 0.0:
                continue
            model.geom_rgba[gid] = np.clip(base * tint, 0.0, 1.0)
            model.geom_rgba[gid, 3] = 1.0
    return tint[:3]


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


def gripper_opening_normalized(base_env) -> float:
    """Gripper opening in [0,1], 0 = fully closed, 1 = fully open.

    Normalised per gripper against its own joint travel, because the raw
    command means different things on different hardware: the same value maps
    to different finger separations on Robotiq85 / Robotiq140 / Panda /
    Rethink / Aloha. Without this the gripper channel would not be comparable
    across embodiments, which defeats the point of a cross-embodiment set.
    """
    sim = base_env.sim
    robot = base_env.robots[0]
    idx = None
    for attr in (
        "_ref_gripper_joint_pos_indexes",
        "_ref_gripper_joint_pos_indexes_dict",
    ):
        v = getattr(robot, attr, None)
        if isinstance(v, dict):
            v = v.get("right")
        if v is not None and len(v) > 0:
            idx = np.asarray(v, dtype=np.int64)
            break
    if idx is None or len(idx) == 0:
        return 0.0
    jids = [sim.model.jnt_qposadr.tolist().index(int(i)) for i in idx]
    lo = sim.model.jnt_range[jids, 0].astype(float)
    hi = sim.model.jnt_range[jids, 1].astype(float)
    q = np.asarray(sim.data.qpos[idx], dtype=float)
    span = np.where(np.abs(hi - lo) > 1e-9, hi - lo, 1.0)
    return float(np.clip(np.mean((q - lo) / span), 0.0, 1.0))


def solve_full_pose_ik(
    base_env,
    target_pos,
    target_xmat,
    fingertip_offset_local=None,
    restarts: int = 8,
    pos_tol: float = 5e-3,
    rot_tol_deg: float = 5.0,
    seed_qpos=None,
):
    """Solve the FULL 6-DOF eef pose, or report that this arm cannot.

    The pipeline used to fall back to a position-only solve whenever the pose
    solve missed, which silently threw the commanded orientation away: with
    that fallback PiPER and YAM ran at a 100% fallback rate, so their recorded
    gripper orientation was not the commanded one at any frame -- it was
    whatever the position solve happened to produce. A dataset whose eef poses
    are only two-thirds specified is not usable as a pose-conditioned dataset,
    so this either hits the commanded pose or says so.

    A single damped-least-squares descent is not enough to make that call
    honestly: it is a local method, and for the short arms it routinely stalls
    hundreds of millimetres away on poses that are in fact solvable (a broad
    restart search finds them). So try many seeds -- the previous solution
    first, for continuity, then the arm's kitchen-ready pose, then random
    draws inside the joint limits -- and keep the best.

    Returns (ok, pos_err_m, rot_err_deg). Joint limits are always respected;
    exceeding them was tested and visibly breaks the rendered arm.
    """
    import mujoco

    sim = base_env.sim
    robot = base_env.robots[0]
    sid = eef_site_id(robot)
    jpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    jids = [sim.model.jnt_qposadr.tolist().index(int(i)) for i in jpos_idx]
    limited = sim.model.jnt_limited[jids].astype(bool)
    lo = np.where(limited, sim.model.jnt_range[jids, 0], -np.pi)
    hi = np.where(limited, sim.model.jnt_range[jids, 1], np.pi)

    def measure():
        sim.forward()
        xm = sim.data.site_xmat[sid].reshape(3, 3).copy()
        p = sim.data.site_xpos[sid].copy()
        if fingertip_offset_local is not None:
            p = p + xm @ fingertip_offset_local
        return (
            float(np.linalg.norm(target_pos - p)),
            float(np.degrees(np.linalg.norm(_rot_to_omega(target_xmat @ xm.T)))),
        )

    q_prev = np.array(sim.data.qpos[jpos_idx], dtype=float)
    seeds = [q_prev]
    if seed_qpos is not None:
        seeds.append(np.clip(np.asarray(seed_qpos, dtype=float), lo, hi))
    rng = np.random.default_rng(0)
    best = None
    for k in range(restarts):
        if k < len(seeds):
            sim.data.qpos[jpos_idx] = seeds[k]
        elif k < len(seeds) + 2:
            sim.data.qpos[jpos_idx] = np.clip(
                q_prev + rng.normal(0, 0.3, len(jpos_idx)), lo, hi
            )
        else:
            sim.data.qpos[jpos_idx] = rng.uniform(lo, hi)
        sim.data.qvel[:] = 0.0
        solve_ik_to_eef_pose(
            base_env,
            target_pos,
            target_xmat,
            match_orientation=True,
            fingertip_offset_local=fingertip_offset_local,
            respect_joint_limits=True,
        )
        pe, re = measure()
        q = np.array(sim.data.qpos[jpos_idx], dtype=float)
        # rank on the commanded tolerances, tie-break on staying near the
        # previous frame so the arm does not jump between IK branches
        score = (
            max(pe - pos_tol, 0) * 1000
            + max(re - rot_tol_deg, 0)
            + 0.02 * np.linalg.norm(q - q_prev)
        )
        if best is None or score < best[0]:
            best = (score, q, pe, re)
        if pe <= pos_tol and re <= rot_tol_deg:
            break
        # Abandon early on a pose this arm is nowhere near. Arms that can hit
        # the pose land it on the first seed or two; arms that cannot were
        # burning the whole restart budget on every frame (PiPER/YAM ran
        # 12-16x slower than UR5e for a 0% success rate).
        if k >= 2 and best[2] > 5 * pos_tol and best[3] > 4 * rot_tol_deg:
            break

    _, q, pe, re = best
    sim.data.qpos[jpos_idx] = q
    sim.data.qvel[:] = 0.0
    sim.forward()
    return (pe <= pos_tol and re <= rot_tol_deg), pe, re


def _deep_self_penetration(base_env, thresh_m: float = 2e-3) -> float:
    """Worst penetration between NON-ADJACENT robot links, in metres.

    Neighbouring links always touch at their shared joint, so a raw contact
    count says nothing. What matters is one link driven deep inside another,
    which is what makes a segment visually disappear -- the same failure that
    made Jaco look like it had a missing link.
    """
    sim = base_env.sim
    d = sim.data._data if hasattr(sim.data, "_data") else sim.data
    body_of = {}
    for n in sim.model.geom_names:
        if n and n.startswith("robot0_"):
            gid = sim.model.geom_name2id(n)
            body_of[gid] = sim.model.geom_bodyid[gid]
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = body_of.get(c.geom1), body_of.get(c.geom2)
        if b1 is None or b2 is None or abs(b1 - b2) <= 1:
            continue
        worst = max(worst, -float(c.dist))
    return worst


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
    orientation_mode="strict",
    orientation_weight=0.15,
    respect_joint_limits=True,
):
    """IK to a target pose. If `fingertip_offset_local` is given, `target_pos`
    is matched at the fingertip/pad midpoint (Jacobian-transported from the
    site) instead of the site origin itself -- orientation is still matched
    at the site (that's what governs the visible gripper twist).

    `orientation_mode` only matters when match_orientation is True:

    "strict"    -- stack position and orientation into one 6-D task and solve
                   them together. Either both are met or the solve is pulled
                   away from the position target trying to reach an
                   unreachable orientation.
    "nullspace" -- soft-prioritised: same 6-D solve but with orientation
                   down-weighted by `orientation_weight`, so position still
                   converges to tolerance while orientation is driven as close
                   as the arm allows. This is what short arms (PiPER/YAM)
                   need: strict mode leaves them ~50-105deg off *and*
                   off-position, while the old fallback of simply dropping
                   orientation threw the gripper twist away entirely.
    """
    sim = base_env.sim
    robot = base_env.robots[0]
    sid = eef_site_id(robot)
    jpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    jvel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)

    # Keep every DLS step inside the arm's actual joint limits. Unclamped this
    # happily returns a "solution" the robot cannot hold -- harmless for the
    # original three embodiments (UR5e/Panda/IIWA have wide or continuous
    # joints and their solutions land in range anyway, verified unchanged),
    # but PiPER/YAM have tight one-sided limits (PiPER joint3 is -2.697..0,
    # joint5 +-1.22) and were being handed poses several radians outside them.
    # MuJoCo then clamps on the next forward(), which is what showed up as
    # ~150mm of eef error plus self-collisions in the reported solution.
    _lo = _hi = None
    try:
        if not respect_joint_limits:
            raise ValueError("joint limits intentionally ignored")
        jids = [sim.model.jnt_qposadr.tolist().index(int(i)) for i in jpos_idx]
        limited = sim.model.jnt_limited[jids].astype(bool)
        rng_lo = sim.model.jnt_range[jids, 0].astype(float).copy()
        rng_hi = sim.model.jnt_range[jids, 1].astype(float).copy()
        rng_lo[~limited] = -np.inf
        rng_hi[~limited] = np.inf
        _lo, _hi = rng_lo, rng_hi
    except (ValueError, IndexError, AttributeError):
        pass  # unusual joint layout -> fall back to the original unclamped solve

    # Stop as soon as the solve stops making progress. Joint-limit clamping
    # (added above) means an unreachable target no longer drifts out of range
    # to a fake solution -- it legitimately stalls against the limits instead,
    # and without this it would burn the entire n_iters budget on every one of
    # a few hundred timesteps, in each fallback stage. Measured on
    # JacoOmron+Robotiq140 that was ~7s per timestep; the arm is simply short
    # of the reference pose there and no extra iteration changes that.
    _best_cost = np.inf
    _stall = 0
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
        _cost = np.linalg.norm(pos_err) + (
            0.05 * np.linalg.norm(ori_err) if match_orientation else 0.0
        )
        if _cost < _best_cost - 1e-6:
            _best_cost = _cost
            _stall = 0
        else:
            _stall += 1
            if _stall >= 25:
                break

        jacp, jacr = _jac_site_compat(sim, sid)
        jp = jacp[:, jvel_idx]
        jr_full = jacr[:, jvel_idx]
        if r_world is not None:
            jp = jp - _skew(r_world) @ jr_full
        if match_orientation and orientation_mode == "nullspace":
            # Soft prioritisation: one 6-D least-squares task, but with the
            # orientation rows scaled down so position dominates the residual.
            # Position therefore still converges to tolerance while orientation
            # is pulled as far as the arm can manage.
            #
            # Deliberately NOT strict task-priority (dq = J+p e + N J+r e).
            # That was tried first and measurably failed: with a *damped*
            # pseudo-inverse, N = I - J+p Jp is not an exact projector, so the
            # secondary term leaks into position and accumulates over the
            # iteration budget -- PiPER went from 0.09mm/45deg to 436mm/22deg,
            # i.e. it bought orientation by destroying the thing that matters
            # more. Down-weighting cannot do that: the position residual is
            # never traded away because it carries full weight.
            j = np.vstack([jp, orientation_weight * jr_full])
            err = np.concatenate([pos_err, orientation_weight * ori_err])
        elif match_orientation:
            j = np.vstack([jp, jr_full])
            err = np.concatenate([pos_err, ori_err])
        else:
            j = jp
            err = pos_err

        a = j @ j.T + (damping**2) * np.eye(j.shape[0])
        dq = j.T @ np.linalg.solve(a, err)
        q = sim.data.qpos[jpos_idx].copy()
        q = q + step * dq
        if _lo is not None:
            q = np.clip(q, _lo, _hi)
        sim.data.qpos[jpos_idx] = q
    sim.forward()


# ---------------------------------------------------------------------------
# Env creation / replay
# ---------------------------------------------------------------------------


def restore_arm_state_by_joint_name(
    base_env, old_model_xml: str, flat_state: np.ndarray
) -> int:
    """Copy the recorded arm (non-gripper) qpos/qvel onto a freshly-built scene.

    Only used for the source robot. Its reference trajectory needs to start
    from the exact recorded joint configuration, but the source robot's own
    env is now built the same way every other embodiment is -- reconstructed
    from ep_meta with THIS install's gripper (see the ROOT-level comment: the
    stock Robotiq85 that atomic_new's UR5eOmron/PandaOmron recordings
    actually used has two independent actuators, which robosuite's "GRIP"
    composite-controller wrapper cannot drive at all -- control_dim mismatch,
    not just a cosmetic mesh difference -- so replaying the *stored* MJCF
    verbatim is not an option for those tasks). A byte-identical flat-state
    restore (mujoco's `set_state_from_flattened`) assumes the loaded model's
    DOF layout matches the recorded one exactly, which no longer holds once
    the gripper differs. Every ARM joint keeps its name and DOF across that
    swap, though (only `gripper0_*` joints change), so copy those by name --
    same technique as `_set_state_by_joint_name` in eval_lerobot_checkpoint.py
    for the equivalent gripper-swap problem. The gripper itself is left at
    its fresh-reset pose; only the arm trajectory feeds the reference
    fingertip/site path other embodiments align to.
    """
    import mujoco

    old = mujoco.MjModel.from_xml_string(old_model_xml)
    state = np.asarray(flat_state).reshape(-1)
    nq_old = old.nq
    qpos_old, qvel_old = state[1 : 1 + nq_old], state[1 + nq_old : 1 + nq_old + old.nv]
    old_idx = {}
    for j in range(old.njnt):
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j)
        old_idx[name] = (old.jnt_qposadr[j], old.jnt_dofadr[j], old.jnt_type[j])

    sim = base_env.sim
    new_model = sim.model
    data = sim.data
    sizes = {
        mujoco.mjtJoint.mjJNT_FREE: (7, 6),
        mujoco.mjtJoint.mjJNT_BALL: (4, 3),
        mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
        mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
    }
    moved = 0
    for name in new_model.joint_names:
        if not name or name.startswith("gripper0_") or name not in old_idx:
            continue
        qa_o, va_o, jt = old_idx[name]
        new_addr = new_model.get_joint_qpos_addr(name)
        new_qa = new_addr[0] if isinstance(new_addr, tuple) else new_addr
        new_vaddr = new_model.get_joint_qvel_addr(name)
        new_va = new_vaddr[0] if isinstance(new_vaddr, tuple) else new_vaddr
        nq, nv = sizes[mujoco.mjtJoint(jt)]
        data.qpos[new_qa : new_qa + nq] = qpos_old[qa_o : qa_o + nq]
        data.qvel[new_va : new_va + nv] = qvel_old[va_o : va_o + nv]
        moved += 1
    sim.forward()
    return moved


def target_controller_config(target_robot: str, osc_kp: float):
    """The controller config for a non-source embodiment: read from THAT
    robot's own atomic_new recording, not a generic per-robot file.

    barx's own pinned robosuite only ships one such file
    (default_pandaomron.json) -- reading from /data1/local/robosuite's
    fuller set (default_ur5eomron.json / default_iiwaomron.json) was the
    original motivation for putting that checkout on sys.path, which is
    exactly what caused the Robotiq85Gripper mismatch documented at the top
    of this file. Every one of the 3 robots has ITS OWN atomic_new dataset
    with a config that's already proven to work (that's what built atomic_new
    in the first place) -- use that instead, with no other install involved.

    Robots with no atomic_new recording of their own (e.g. JacoOmron, added
    only as a visual-robust target, never as an atomic_new source) have
    nothing to read here. robosuite's own load_composite_controller_config()
    can't stand in for it either: it doesn't have a per-robot JSON for every
    robot and silently falls back to a dual-arm "basic.json" template with no
    base/torso body parts at all -- wrong composite type for a single-arm
    Omron mobile-base robot, not just a missing body_part_ordering. A
    non-source embodiment's arm qpos gets frozen back to the IK-aligned pose
    every step regardless of this config's OSC gains (see replay_embodiment's
    per-step freeze/restore), so only the gripper/base routing actually
    matters here -- borrow another single-arm Omron composite's *whole*
    config (UR5eOmron's, which also uses Robotiq85Gripper) rather than
    hand-assembling a HYBRID_MOBILE_BASE structure from scratch.
    """
    source_robot = target_robot if target_robot in ATOMIC_NEW_DATASET else "UR5eOmron"
    cfg = deepcopy(
        json.loads(
            (
                ATOMIC_NEW_DATASET[source_robot] / "extras" / "dataset_meta.json"
            ).read_text()
        )["env_args"]["env_kwargs"]["controller_configs"]
    )
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


# What each robot's OWN atomic_new dataset actually recorded
# (dataset_meta.json env_kwargs.gripper_types). PandaOmron used an explicit
# Robotiq85Gripper override, not Panda's native PandaGripper. UR5eOmron and
# IIWAOmron have NO "gripper_types" key at all in their recording (confirmed
# via `"gripper_types" in env_kwargs` -- not just `.get()`, which can't tell
# a JSON null from a missing key), so robosuite's own "default" parameter
# value took over and resolved their native gripper (Robotiq85Gripper /
# Robotiq140Gripper respectively). The literal Python value None is NOT the
# same thing here and must not be used: robosuite's own docs (manipulation_env.py)
# spell out that gripper_types=None means "remove the gripper entirely" --
# passing it explicitly for UR5e broke composite-controller construction
# outright (AssertionError, body_part_ordering length mismatch) rather than
# just picking the wrong mesh. Every embodiment in a visual_robust_new
# episode must use ITS OWN entry here, not "whatever gripper the robot
# defaults to" -- otherwise Panda renders holding a PandaGripper it was
# never actually recorded with whenever it's an alternate embodiment instead
# of the source robot.
ATOMIC_NEW_GRIPPER = {
    "PandaOmron": "Robotiq85Gripper",
    "UR5eOmron": "default",
    "IIWAOmron": "default",
}


RobotSpec = namedtuple("RobotSpec", ["tag", "robot_class", "gripper"])


def parse_robot_spec(spec: str) -> "RobotSpec":
    """Parse one --robots entry.

    A bare 'RobotClass' keeps the original behaviour: tag == robot_class,
    gripper resolved from ATOMIC_NEW_GRIPPER (unchanged for the 3 embodiments
    every prior visual_robust_new run used). 'Tag:RobotClass:Gripper' lets the
    same robot class appear twice under different grippers -- e.g. Panda's own
    atomic_new Robotiq85Gripper (plain "PandaOmron") alongside its native
    PandaGripper ("PandaOmronPandaGripper:PandaOmron:default") -- which a bare
    class name can't express since results/tags are keyed by this string and
    two embodiments can't share one key.
    """
    parts = spec.split(":")
    if len(parts) == 1:
        return RobotSpec(tag=parts[0], robot_class=parts[0], gripper=None)
    if len(parts) == 3:
        tag, robot_class, gripper = parts
        return RobotSpec(tag=tag, robot_class=robot_class, gripper=gripper or None)
    raise ValueError(
        f"--robots entry {spec!r} must be 'RobotClass' or 'Tag:RobotClass:Gripper'"
    )


def parse_robot_specs(specs) -> list:
    return [parse_robot_spec(s) for s in specs]


def create_env(
    env_kwargs_base: dict,
    robot: str,
    camera_names,
    camera_h,
    camera_w,
    osc_kp: float,
    is_source_robot: bool = False,
    gripper_override: str | None = None,
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
    if is_source_robot:
        # Recorded gripper_types encodes what the reference action stream
        # actually assumes, not just cosmetics -- atomic_new's
        # PandaOmron/TurnOnSinkFaucet was recorded with Robotiq85Gripper, not
        # Panda's own default PandaGripper, and the two have different action
        # semantics. Leave the value exactly as recorded (including a bare
        # None, which is NOT the same as Kitchen's own "default" string --
        # composite Omron robots resolve them differently; forcing a falsy
        # recording to the literal "default" was observed to change the
        # composite controller's action space, e.g. UR5eOmron+Robotiq85Gripper
        # going from a correct 12-dim action_dim down to 11).
        pass
    else:
        # Non-source embodiments regenerate their scene from ep_meta rather
        # than loading raw XML, so nothing FORCES a particular gripper here
        # -- but it should still be the one THIS robot's own atomic_new
        # recording used (see ATOMIC_NEW_GRIPPER above), not a blind
        # "default" that silently gives Panda its native PandaGripper
        # instead of the Robotiq85Gripper every atomic_new/PandaOmron
        # episode actually has. gripper_override (from a "Tag:RobotClass:
        # Gripper" --robots entry) takes priority so the same robot class can
        # be rendered under a gripper other than what its own atomic_new
        # recording used.
        env_meta["env_kwargs"]["gripper_types"] = (
            gripper_override
            if gripper_override is not None
            else ATOMIC_NEW_GRIPPER.get(robot, "default")
        )
    # robocasa_x (which recorded atomic_new) uses "A"/"B" object splits; this
    # standalone robocasa only knows "pretrain"/"target"/None and otherwise
    # raises inside sample_kitchen_object_helper. The split only constrains
    # which instances a *freshly sampled* replacement object may be drawn
    # from (used by install_object_registry_fallback below when the exact
    # recorded instance isn't registered here) -- not something that needs to
    # match the original recording for a visual-robustness render.
    split = env_meta["env_kwargs"].get("obj_instance_split")
    if split not in (None, "pretrain", "target", "A", "B"):
        env_meta["env_kwargs"]["obj_instance_split"] = None
    if robot != "PandaOmron" and not is_source_robot:
        # Only re-derive a generic controller config for non-source
        # embodiments. The source robot's own recorded controller_configs is
        # what actually produced the stored raw_actions (12-dim, matching
        # MODALITY_ACTION) -- load_composite_controller_config's generic
        # default for e.g. UR5eOmron+Robotiq85Gripper computes a
        # differently-sized action space (observed: 11, not 12) and breaks
        # `env.step(raw_actions[t])` for every source-robot episode.
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


# atomic_new writes its front camera under this fixed key regardless of which
# robot actually recorded it (convert_mg_to_atomic_new.py's FRONT_CAMERA).
RECORDED_FRONT_CAMERA_KEY = "robot0_agentview_right"


def install_recorded_camera_override() -> None:
    """Render from the camera pose the episode actually carries in ep_meta.

    robocasa_x's Kitchen.set_cameras() rebuilds `_cam_configs` from
    CAM_CONFIGS (re-randomizing if enabled) on every `_load_model`; it never
    reads ep_meta["cam_configs"]. atomic_new's front camera is the
    per-episode-randomized view the mg HDF5s rendered, so without this
    override every embodiment gets a different, freshly-derived viewpoint
    instead.

    This just applies whatever is in ep_meta verbatim -- it does not by
    itself make different embodiments agree on a viewpoint. The recorded
    "robot0_agentview_right" entry is defined relative to `mobilebase0_support`
    (the ROBOT's own base frame), and different robot types' Omron bases sit
    at different world positions/heights (measured: Panda/UR5e base z=0.90,
    IIWA z=0.70), so naively reusing this hook alone would give each
    embodiment a different world-space view despite an identical local pose.
    `process_episode` handles that: right after building the source robot
    (whose local-frame pose IS correct for itself), it reads back the
    resulting WORLD camera pose and rewrites this same ep_meta entry into an
    absolute, parent_body-less pose before any other embodiment is built --
    see the "pin the camera in world space" comment there. This hook only
    needs to exist so that rewritten value actually reaches `_cam_configs`.
    """
    from robocasa.environments.kitchen.kitchen import Kitchen

    if getattr(Kitchen, "_recorded_camera_hooked", False):
        return
    original = Kitchen.set_cameras

    def set_cameras(self):
        original(self)
        recorded = (getattr(self, "_ep_meta", None) or {}).get("cam_configs") or {}
        for name, cfg in recorded.items():
            if name in self._cam_configs:
                self._cam_configs[name] = deepcopy(cfg)

    Kitchen.set_cameras = set_cameras
    Kitchen._recorded_camera_hooked = True


def capture_world_camera_pose(base_env, cam_name: str):
    """Read a compiled camera's absolute world pose off the live sim.

    Returns (pos: (3,) array, quat_wxyz: (4,) array) in the same convention
    MJCF's `<camera pos="..." quat="...">` attributes use, so the result can
    be written straight back into a cam_configs entry.
    """
    import mujoco

    sim = base_env.sim
    m = sim.model._model if hasattr(sim.model, "_model") else sim.model
    d = sim.data._data if hasattr(sim.data, "_data") else sim.data
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"camera {cam_name!r} not found in compiled model")
    pos = np.array(d.cam_xpos[cam_id], dtype=np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quat, np.ascontiguousarray(d.cam_xmat[cam_id], dtype=np.float64)
    )
    return pos, quat


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
    # Every embodiment (source included -- see the ROOT-level comment on why
    # verbatim stored-XML replay isn't viable for the Robotiq85 tasks)
    # rebuilds its own model from ep_meta, reusing the SAME
    # layout_id/style_id/objects so the kitchen scene itself stays identical
    # -- the same reconstruct-from-ep_meta trick used for the mg->atomic
    # LeRobot conversion (kitchen.py:591-608).
    base_env.set_ep_meta(dict(ep_meta))
    obs = env_wrapper.reset(unset_ep_meta=False)
    if is_source_robot:
        # Scene reconstruction alone gives a fresh-reset robot, not the
        # recorded one; put the reference trajectory's arm back where it
        # actually started (gripper excluded -- see restore_arm_state_by_joint_name).
        restore_arm_state_by_joint_name(
            base_env, initial_state["model"], initial_state["states"]
        )

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
        """Drive this arm to the commanded eef pose. Returns (ok, pos_err, rot_err).

        `ok` False means this embodiment cannot hold the commanded pose at
        this frame -- its links or joint limits do not allow it. The frame is
        still rendered (with the closest pose the arm could reach) but is
        flagged so it can be excluded per embodiment, which is the whole point
        of recording coverage: restrict the arm that cannot do it, keep every
        arm that can.
        """
        ok, pe, re = solve_full_pose_ik(
            base_env,
            ref_fingertip_pos_seq[t],
            target_xmat(t),
            fingertip_offset_local=this_fingertip_offset,
            pos_tol=POSE_POS_TOL_M,
            rot_tol_deg=POSE_ROT_TOL_DEG,
            seed_qpos=KITCHEN_INIT_QPOS.get(robot_name),
        )
        if not ok:
            # This arm cannot hold the commanded pose here, so the frame is
            # flagged unusable for it either way. Do not also leave the arm at
            # the best-effort 6-DOF compromise: that sacrifices position too
            # (PiPER averaged 340mm off), which renders the arm somewhere it
            # never goes. Recover the position-correct pose instead, so the
            # video stays physically sensible while `reachable` records that
            # the commanded orientation was not met.
            solve_ik_to_eef_pose(
                base_env,
                ref_fingertip_pos_seq[t],
                target_xmat(t),
                match_orientation=False,
                fingertip_offset_local=this_fingertip_offset,
                respect_joint_limits=True,
            )
            base_env.sim.forward()
            fp, xm = (
                fingertip_world_pose(base_env, this_fingertip_offset)[0],
                eef_world_pose(base_env)[1],
            )
            pe = float(np.linalg.norm(fp - ref_fingertip_pos_seq[t]))
            re = float(np.degrees(np.linalg.norm(_rot_to_omega(target_xmat(t) @ xm.T))))
        base_env.sim.data.qvel[:] = 0.0
        base_env.sim.forward()
        return ok, pe, re

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
        # Per-frame record of how well THIS embodiment actually managed to
        # hit the reference eef state. Arms differ in reach and joint
        # limits, so some frames of a trajectory recorded on a large arm
        # are simply not achievable on a small one -- rather than dropping
        # whole embodiments for that, every frame is kept and labelled, so
        # a consumer can filter to the poses a given eef could really
        # reach. Source robot defines the reference, so it is exact by
        # construction.
        "pos_err_m": [],
        "rot_err_deg": [],
        "reachable": [],
        "grip_norm": [],
    }
    for t in range(raw_actions.shape[0]):
        if is_source_robot:
            ok, p_err, r_err = True, 0.0, 0.0
        else:
            ok, p_err, r_err = align(t)
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
        buf["grip_norm"].append(gripper_opening_normalized(base_env))
        buf["pos_err_m"].append(p_err)
        buf["rot_err_deg"].append(r_err)
        buf["reachable"].append(bool(ok))
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
        # tag -> running reachability totals, written out by finalize()
        self.reach_stats: dict[str, dict] = {}

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
                np.asarray(ref["grip_norm"], dtype=np.float32).reshape(-1, 1),
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
        # Per-embodiment reachability, one column trio per tag, so a consumer
        # can select exactly the frames a given eef actually achieved without
        # having to re-run any IK.
        for tag in self.tags:
            r = results[tag]
            # the pose this embodiment actually holds, in the same
            # (xyz, quat_xyzw, gripper_opening_normalised) layout as
            # observation.state, so each arm's realised eef state is
            # recoverable per frame rather than only the reference's
            eef = np.concatenate(
                [
                    np.stack(r["fingertip_pos"][:ep_len]),
                    np.stack([_mat2quat(m) for m in r["eef_quat"][:ep_len]]),
                    np.asarray(r["grip_norm"][:ep_len], dtype=np.float32).reshape(
                        -1, 1
                    ),
                ],
                axis=1,
            ).astype(np.float32)
            columns[f"eef_state.{tag}"] = list(eef)
            columns[f"reachable.{tag}"] = np.asarray(
                r["reachable"][:ep_len], dtype=bool
            )
            columns[f"pos_err_m.{tag}"] = np.asarray(
                r["pos_err_m"][:ep_len], dtype=np.float32
            )
            columns[f"rot_err_deg.{tag}"] = np.asarray(
                r["rot_err_deg"][:ep_len], dtype=np.float32
            )
            # Present only for generators that measure it; without this the
            # value is computed per frame and then silently dropped, leaving
            # no way to check the shipped frames for occlusion.
            if "eef_visible_frac" in r:
                columns[f"eef_visible_frac.{tag}"] = np.asarray(
                    r["eef_visible_frac"][:ep_len], dtype=np.float32
                )
            if "eef_visible_center" in r:
                columns[f"eef_visible_center.{tag}"] = np.asarray(
                    r["eef_visible_center"][:ep_len], dtype=bool
                )
            for _k in ("eef_u", "eef_v"):
                if _k in r:
                    columns[f"{_k}.{tag}"] = np.asarray(
                        r[_k][:ep_len], dtype=np.float32
                    )
            self.reach_stats.setdefault(
                tag,
                {
                    "frames": 0,
                    "reachable": 0,
                    "pos_reachable": 0,
                    "pos_err_sum": 0.0,
                    "rot_err_sum": 0.0,
                },
            )
            st = self.reach_stats[tag]
            st["frames"] += ep_len
            st["reachable"] += int(np.count_nonzero(columns[f"reachable.{tag}"]))
            # Position-only is tracked separately because it is a genuinely
            # different capability: the short arms hit the reference *point*
            # essentially always and only miss the orientation, so collapsing
            # both into one number would read as "this arm reached nothing".
            st["pos_reachable"] += int(
                np.count_nonzero(columns[f"pos_err_m.{tag}"] <= REACH_POS_TOL_M)
            )
            st["pos_err_sum"] += float(np.sum(columns[f"pos_err_m.{tag}"]))
            st["rot_err_sum"] += float(np.sum(columns[f"rot_err_deg.{tag}"]))
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

        per_ep_reach = {
            tag: int(
                np.count_nonzero(
                    np.asarray(results[tag]["reachable"][:ep_len], dtype=bool)
                )
            )
            for tag in self.tags
        }
        self.episodes_f.write(
            json.dumps(
                {
                    "episode_index": ep_idx,
                    "tasks": [lang],
                    "length": ep_len,
                    "reachable_frames": per_ep_reach,
                }
            )
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
                "names": [
                    "fingertip_xyz(3)+quat_xyzw(4)+gripper_opening_normalized(1)"
                ],
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

        # Per-eef coverage manifest: what each embodiment actually managed to
        # produce. Arms differ in reach and joint limits, so no embodiment is
        # dropped for failing part of a trajectory -- this records which
        # fraction of frames each one genuinely achieved, so the dataset can
        # be filtered per eef instead of trimmed to the weakest arm.
        manifest = {
            "source_robot": source_robot,
            "reach_criteria": {
                "pos_tol_m": REACH_POS_TOL_M,
                "rot_tol_deg": REACH_ROT_TOL_DEG,
                "note": "a frame is reachable when that arm hit the reference fingertip "
                "position within pos_tol_m AND the reference gripper orientation "
                "within rot_tol_deg",
            },
            "per_frame_columns": {
                "reachable.<tag>": "bool, this eef achieved reference position AND orientation",
                "pos_err_m.<tag>": "float32, fingertip distance from the reference position",
                "rot_err_deg.<tag>": "float32, angle between achieved and reference gripper orientation",
            },
            "embodiments": {},
        }
        for tag in self.tags:
            st = self.reach_stats.get(
                tag,
                {
                    "frames": 0,
                    "reachable": 0,
                    "pos_reachable": 0,
                    "pos_err_sum": 0.0,
                    "rot_err_sum": 0.0,
                },
            )
            n = max(st["frames"], 1)
            manifest["embodiments"][tag] = {
                "total_frames": st["frames"],
                "reachable_frames": st["reachable"],
                "reachable_fraction": round(st["reachable"] / n, 4),
                "position_reachable_frames": st["pos_reachable"],
                "position_reachable_fraction": round(st["pos_reachable"] / n, 4),
                "mean_pos_err_mm": round(1000.0 * st["pos_err_sum"] / n, 3),
                "mean_rot_err_deg": round(st["rot_err_sum"] / n, 2),
                "is_source": tag == self.tags[0],
            }
        with open(self.meta_dir / "embodiment_coverage.json", "w") as f:
            json.dump(manifest, f, indent=2)


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
    if getattr(args, "style_sweep", False):
        # Every embodiment in this episode still shares one scene (they all
        # rebuild from this same ep_meta), so the cross-embodiment comparison
        # is unaffected -- only the appearance differs episode to episode.
        ep_meta = dict(ep_meta)
        ep_meta["style_id"] = episode_index % 12
    initial_state = {
        "states": states[0],
        "model": model_xml,
        "ep_meta": json.dumps(ep_meta),
    }
    source_robot = source_env_meta["env_kwargs"]["robots"]
    if isinstance(source_robot, (list, tuple)):
        source_robot = source_robot[0]

    specs = parse_robot_specs(args.robots)
    source_spec = next((s for s in specs if s.robot_class == source_robot), None)
    if source_spec is None:
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
    ordered_specs = [source_spec] + [s for s in specs if s is not source_spec]
    for spec in ordered_specs:
        is_source_robot = spec is source_spec
        env = create_env(
            source_env_meta,
            spec.robot_class,
            args.camera_names,
            args.camera_height,
            args.camera_width,
            args.osc_kp,
            is_source_robot=is_source_robot,
            gripper_override=spec.gripper,
        )
        env = _EnvWithCameras(env, args.camera_names)
        try:
            buf = replay_embodiment(
                env,
                initial_state,
                ep_meta,
                is_source_robot,
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
                robot_name=spec.robot_class,
            )
            if is_source_robot:
                # Pin the front camera in WORLD space before any other
                # embodiment is built. The recorded pose is correct as-is for
                # the source robot (it's relative to the source's own base),
                # but every other robot type's Omron base sits at a different
                # world position/height -- reusing the same local-frame pose
                # for them would silently shift the camera per robot (see
                # install_recorded_camera_override's docstring). Reading back
                # the ALREADY-compiled world pose and rewriting this ep_meta
                # entry as an absolute, parent_body-less camera makes every
                # later embodiment in this episode attach to the identical
                # world transform, so the visible background is the same
                # background, not just the same *scene*.
                # Only the static front/agentview camera gets pinned --
                # robot0_eye_in_hand is mounted on the gripper by design and
                # must keep following each embodiment's own hand.
                cam_key = RECORDED_FRONT_CAMERA_KEY
                if cam_key in args.camera_names and cam_key in ep_meta.get(
                    "cam_configs", {}
                ):
                    world_pos, world_quat = capture_world_camera_pose(env.env, cam_key)
                    old_cfg = ep_meta["cam_configs"][cam_key]
                    ep_meta["cam_configs"][cam_key] = {
                        "pos": world_pos.tolist(),
                        "quat": world_quat.tolist(),
                        "camera_attribs": dict(old_cfg.get("camera_attribs", {})),
                    }
        finally:
            env.env.close()
        results[spec.tag] = buf
        if is_source_robot:
            ref_fingertip_pos_seq = buf["fingertip_pos"]
            ref_site_xmat_seq = buf["eef_quat"]
            reference_opening_axis = buf["opening_axis"]
        print(
            f"[episode {episode_index}] {spec.tag} ({spec.robot_class}): {raw_actions.shape[0]} steps replayed "
            f"(opening_axis={np.round(buf['opening_axis'], 3)}, "
            f"fingertip_offset={np.round(np.linalg.norm(buf['fingertip_offset'])*1000, 1)}mm)"
        )

    if writer is not None:
        writer.add_episode(results, raw_actions, ep_meta["lang"])

    if save_examples:
        tags = [s.tag for s in ordered_specs]
        examples_dir = args.output_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        for cam in args.camera_names:
            save_preview(
                examples_dir, episode_index, tags, cam, results, args.preview_frames
            )
        save_videos(examples_dir, episode_index, tags, args.camera_names, results)
        save_hdf5(examples_dir, episode_index, tags, raw_actions, results)


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    patch_python_paths()
    args = parse_args()

    import robocasa  # noqa: F401 - registers RoboCasa envs

    # Must precede any env build: Sawyer's robot.xml (and several
    # robosuite_models grippers) are unloadable on this MuJoCo without it.
    install_mesh_inertia_compat()
    # Registers PiPER/YAM + their Omron composites, which have no robosuite or
    # robosuite_models wrapper of their own (converted from mujoco_menagerie).
    import custom_robots  # noqa: F401

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
    specs = parse_robot_specs(args.robots)
    source_spec = next((s for s in specs if s.robot_class == source_robot), None)
    if source_spec is None:
        raise ValueError(f"--robots must include the source robot {source_robot!r}")
    ordered_tags = [source_spec.tag] + [s.tag for s in specs if s is not source_spec]

    writer = None
    if args.write_lerobot:
        writer = CrossEmbodimentLerobotWriter(
            args.output_dir / "lerobot",
            ordered_tags,
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
