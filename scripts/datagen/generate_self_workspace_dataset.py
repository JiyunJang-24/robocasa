#!/usr/bin/env python3
"""Cross-embodiment data for arms that cannot replay the reference trajectory.

The visual-robust set replays one UR5e trajectory on every embodiment. Arms
with short links or tight wrists cannot hold those commanded eef poses at all:
measured over a full episode, PiPER and YAM achieve the 6-DOF pose on 0% of
frames -- PiPER's joint3/joint5 sit against their limits on every attempt --
so the only thing recorded for them was a position-only approximation with the
commanded orientation discarded.

Rather than ship that, this generates poses those arms *can* actually hold.
Trajectories are built in JOINT space and the eef pose is read off by forward
kinematics, which is the key choice: sampling Cartesian poses and hoping IK
converges reproduces exactly the failure above, whereas a joint-space path
inside the limits is reachable by construction, so every frame is a genuine
6-DOF pose for that embodiment (valid by definition, not by tolerance).

Poses are additionally constrained to keep the gripper inside the camera
frustum, since an arm waving outside the frame is not useful visual data.

State layout matches the main pipeline: (xyz, quat_xyzw, gripper_opening
normalised to each gripper's own travel).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCENE_PENETRATION_TOL_M = 2e-3
SELF_PENETRATION_TOL_M = 2e-3
# Fraction of frame width/height the end-effector must stay clear of the edge by
VIS_MARGIN_FRAC = 0.06

ROOT = Path("/data1/local/robocasa")
sys.path.insert(0, str(ROOT))

import generate_visual_robust_cross_embodiment_x as G  # noqa: E402


def camera_projection(base_env, cam_name: str, height: int, width: int):
    """World -> pixel projection for a MuJoCo camera."""
    sim = base_env.sim
    cid = sim.model.camera_name2id(cam_name)
    sim.forward()
    cam_pos = np.array(sim.data.cam_xpos[cid], dtype=float)
    cam_mat = np.array(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
    fovy = float(sim.model.cam_fovy[cid])
    f = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    def project(p_world):
        # MuJoCo cameras look down their local -Z, with +X right and +Y up.
        p_cam = cam_mat.T @ (np.asarray(p_world, dtype=float) - cam_pos)
        if p_cam[2] >= -1e-6:
            return None  # behind the camera
        u = width / 2.0 + f * (p_cam[0] / -p_cam[2])
        v = height / 2.0 - f * (p_cam[1] / -p_cam[2])
        return np.array([u, v])

    return project


def camera_unprojection(base_env, cam_name: str, height: int, width: int):
    """Pixel + depth -> world point: the inverse of camera_projection.

    Targets used to be drawn as a ball around the arm's resting fingertip and
    then filtered for being in shot. For arms whose rest pose sits above the
    view that filter throws away nearly everything -- YAM+Aloha rests 287px
    above the top of the frame, so no ball around it reaches the image at all,
    the pool came up empty and the episode froze. Choosing the pixel first and
    solving for the world point makes every target visible by construction.
    """
    sim = base_env.sim
    cid = sim.model.camera_name2id(cam_name)
    sim.forward()
    cam_pos = np.array(sim.data.cam_xpos[cid], dtype=float)
    cam_mat = np.array(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
    fovy = float(sim.model.cam_fovy[cid])
    f = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    def unproject(u, v, depth):
        p_cam = np.array(
            [(u - width / 2.0) * depth / f, -(v - height / 2.0) * depth / f, -depth],
            dtype=float,
        )
        return cam_pos + cam_mat @ p_cam

    return unproject


def env_penetration(base_env, thresh=1e-3):
    """Worst penetration between the robot and anything that is not the robot.

    A joint-space path is reachable by construction but says nothing about the
    scene: without this the arm happily sweeps through the counter and sink,
    which is both physically impossible and obviously wrong on video.
    """
    sim = base_env.sim
    d = sim.data._data if hasattr(sim.data, "_data") else sim.data
    # Only the moving arm counts. The mobile base rests on the floor, which is
    # a permanent, legitimate contact -- including it marked every pose as
    # colliding, so the pose pool came up empty and the arm stood still.
    arm_geoms = set()
    for n in sim.model.geom_names:
        if not n:
            continue
        if n.startswith("gripper0_"):
            arm_geoms.add(sim.model.geom_name2id(n))
        elif n.startswith("robot0_"):
            body = (
                sim.model.body_id2name(sim.model.geom_bodyid[sim.model.geom_name2id(n)])
                or ""
            )
            low = body.lower()
            if not any(k in low for k in ("base", "mount", "wheel", "caster", "torso")):
                arm_geoms.add(sim.model.geom_name2id(n))
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = c.geom1 in arm_geoms, c.geom2 in arm_geoms
        if a != b:  # exactly one side is the moving arm -> arm vs scene
            worst = max(worst, -float(c.dist))
    return worst


def self_penetration(base_env):
    """Worst penetration between parts of the arm that should never touch.

    Scoping matters more than the threshold here. Counting every robot-robot
    contact reports ~4mm on 100% of frames for every embodiment, because a
    closed gripper's fingers legitimately touch each other -- a baseline that
    drowns out the real signal. Excluded: neighbouring links (they share a
    joint) and gripper-internal contacts (fingers/knuckles). What remains is
    the arm folding through itself, which is what looks broken -- measured up
    to 188mm on randomly sampled UR5e configurations.
    """
    sim = base_env.sim
    d = sim.data._data if hasattr(sim.data, "_data") else sim.data
    body_of, is_grip = {}, {}
    for n in sim.model.geom_names:
        if not n or not (n.startswith("robot0_") or n.startswith("gripper0_")):
            continue
        gid = sim.model.geom_name2id(n)
        body_of[gid] = sim.model.geom_bodyid[gid]
        is_grip[gid] = n.startswith("gripper0_")
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 not in body_of or g2 not in body_of:
            continue
        if is_grip.get(g1) and is_grip.get(g2):
            continue  # fingers touching fingers
        if abs(body_of[g1] - body_of[g2]) <= 1:
            continue  # neighbouring links share a joint
        worst = max(worst, -float(c.dist))
    return worst


# Tilt away from the starting orientation, in degrees, that a pose may have.
# Sampling the joint ranges freely puts the gripper at orientations no operator
# would ever command -- upside down, wrist cranked over -- and the occlusion
# filter then selects whichever of those happen not to be blocked, which is how
# the eef ended up parked in a frame corner. Spin about the vertical (yaw) is
# unconstrained; only the tilt is capped, so position can still range widely.
MAX_TILT_DEG = 20.0


def tilt_from_initial_deg(xmat, xmat0):
    """Smallest rotation from the initial orientation once yaw is factored out.

    Allowing any yaw means the pose is acceptable if SOME rotation about the
    world vertical brings it within tolerance, so the measure is
    min over yaw of the residual rotation angle. Comparing euler angles
    directly would instead mix yaw into roll/pitch and reject perfectly good
    poses near gimbal configurations.
    """
    m = (
        np.asarray(xmat, dtype=float).reshape(3, 3)
        @ np.asarray(xmat0, dtype=float).reshape(3, 3).T
    )
    # trace(Rz(-psi) @ m) = cos(psi)*(m00+m11) + sin(psi)*(m01-m10) + m22,
    # so the best-case trace over psi has a closed form
    best = np.hypot(m[0, 0] + m[1, 1], m[0, 1] - m[1, 0]) + m[2, 2]
    return float(np.degrees(np.arccos(np.clip((best - 1.0) / 2.0, -1.0, 1.0))))


VIS_BALL_RADIUS_M = 0.08
VIS_N_RAYS = 14
# Every ray in the ball must reach: a pose whose eef neighbourhood is
# partly blocked is rejected outright rather than merely flagged.
VIS_CLEAR_MIN_FRAC = 1.0
VIS_SURFACE_EPS_M = 0.02


def _vis_ball_offsets(r=VIS_BALL_RADIUS_M, n=VIS_N_RAYS):
    """Centre point plus an even spread over a sphere of radius r."""
    pts = [np.zeros(3)]
    golden = (1 + 5**0.5) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rad = np.sqrt(max(0.0, 1 - z * z))
        th = 2 * np.pi * i / golden
        pts.append(r * np.array([rad * np.cos(th), rad * np.sin(th), z]))
    return np.array(pts)


def _gripper_body_ids(model):
    """Bodies making up the hand: a ray stopping on these still sees the eef."""
    hand = model.body_name2id("robot0_right_hand")
    out = {hand}
    for i in range(model.nbody):
        c = i
        while c != 0:
            if c == hand:
                out.add(i)
                break
            c = model.body_parentid[c]
    return out


def eef_visibility(
    base_env,
    cam_id,
    fingertip_pos,
    grip_bodies,
    offsets,
    project=None,
    frame_wh=None,
    margin_frac=0.0,
):
    """How much of the end-effector's neighbourhood the camera can actually see.

    The in-frame test elsewhere only projects the eef into the image, so it
    passes even when the eef is behind one of the arm's own links -- measured
    on accepted poses, that happens in 21-37% of frames. Casting rays to a ball
    of points around the eef, rather than to the single point, also marks down
    frames whose surroundings are occluded.

    Unobstructed is not the same as visible: a point outside the image is
    reached by an unblocked ray while being nowhere on screen, which let the
    gripper hang off the frame edge and still score a perfect 1.00. When a
    projection is supplied, points that fall outside the frame count as not
    seen.

    Returns (fraction of rays that see the eef, centre point seen).
    """
    import mujoco

    m = base_env.sim.model
    d = base_env.sim.data
    mj_m = m._model if hasattr(m, "_model") else m
    mj_d = d._data if hasattr(d, "_data") else d
    cp = np.array(d.cam_xpos[cam_id], dtype=float)
    gid = np.zeros(1, dtype=np.int32)
    clear, center_clear = 0, True
    for k, off in enumerate(offsets):
        tgt = np.asarray(fingertip_pos, dtype=float) + off
        if project is not None and frame_wh is not None:
            fw, fh = frame_wh
            uv = project(tgt)
            mxp, myp = margin_frac * fw, margin_frac * fh
            if not (
                uv is not None and mxp <= uv[0] < fw - mxp and myp <= uv[1] < fh - myp
            ):
                if k == 0:
                    center_clear = False
                continue
        v = tgt - cp
        dist = float(np.linalg.norm(v))
        if dist <= 1e-9:
            clear += 1
            continue
        hit = mujoco.mj_ray(mj_m, mj_d, cp, v / dist, None, 1, -1, gid)
        blocked = bool(
            gid[0] >= 0
            and hit < dist - VIS_SURFACE_EPS_M
            and m.geom_bodyid[gid[0]] not in grip_bodies
        )
        clear += int(not blocked)
        if k == 0:
            center_clear = not blocked
    return clear / len(offsets), center_clear


def joint_space_trajectory(lo, hi, n_steps, rng, margin=0.12, n_waypoints=6, smooth=9):
    """A smooth joint path that stays inside the limits by construction.

    Waypoints are drawn from the interior of the joint ranges (a margin keeps
    the arm off its hard stops, which is where it both looks strained and
    tends to self-intersect), linearly interpolated, then low-pass filtered so
    the motion is continuous rather than a sequence of jerks.
    """
    span = hi - lo
    lo_i, hi_i = lo + margin * span, hi - margin * span
    waypoints = rng.uniform(lo_i, hi_i, size=(n_waypoints, len(lo)))
    seg = np.linspace(0, n_waypoints - 1, n_steps)
    path = np.empty((n_steps, len(lo)))
    for j in range(len(lo)):
        path[:, j] = np.interp(seg, np.arange(n_waypoints), waypoints[:, j])
    if smooth > 1:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        for j in range(len(lo)):
            padded = np.pad(path[:, j], pad, mode="edge")
            path[:, j] = np.convolve(padded, k, mode="valid")[:n_steps]
    return np.clip(path, lo_i, hi_i)


def build_episode(
    env_wrapper,
    base_env,
    robot_name,
    n_steps,
    cam_names,
    h,
    w,
    rng,
    hide_scope,
    visible_cam,
    max_tries=40,
):
    """One episode of in-workspace, in-frame motion for this arm."""
    robot = base_env.robots[0]
    jpos = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    jids = [base_env.sim.model.jnt_qposadr.tolist().index(int(i)) for i in jpos]
    limited = base_env.sim.model.jnt_limited[jids].astype(bool)
    lo = np.where(limited, base_env.sim.model.jnt_range[jids, 0], -np.pi)
    hi = np.where(limited, base_env.sim.model.jnt_range[jids, 1], np.pi)

    grip_idx = None
    for attr in ("_ref_gripper_joint_pos_indexes",):
        v = getattr(robot, attr, None)
        if isinstance(v, dict):
            v = v.get("right")
        if v is not None and len(v) > 0:
            grip_idx = np.asarray(v, dtype=np.int64)
    if grip_idx is not None:
        gjids = [
            base_env.sim.model.jnt_qposadr.tolist().index(int(i)) for i in grip_idx
        ]
        glo = base_env.sim.model.jnt_range[gjids, 0].astype(float)
        ghi = base_env.sim.model.jnt_range[gjids, 1].astype(float)

    if hide_scope in ("all_objects", "robot_only", "no_kitchen"):
        G.fly_objects_away(base_env, 50.0, 5.0)
    if hide_scope == "robot_only":
        G.hide_kitchen_background(base_env)
    elif hide_scope == "no_kitchen":
        # room without the kitchen, tinted per episode so the background
        # actually varies instead of being one flat skybox
        G.hide_kitchen_furniture(base_env)
        G.randomize_backdrop(base_env, rng)

    project = camera_projection(base_env, visible_cam, h, w)
    off = G.measure_fingertip_offset_local(base_env)
    vis_cam_id = base_env.sim.model.camera_name2id(visible_cam)
    vis_grip_bodies = _gripper_body_ids(base_env.sim.model)
    vis_offsets = _vis_ball_offsets()
    # orientation the arm rests at after reset -- poses are kept near it
    _, xmat0 = G.eef_world_pose(base_env)
    sid = G.eef_site_id(robot)

    # Build the path from poses that are individually known good, instead of
    # sampling whole paths and hoping. Rejecting whole random paths does not
    # work here: in a kitchen most of the joint range puts the arm inside the
    # counter, so nearly every sampled path collided somewhere and the
    # hold-last-clean rule then froze the arm for most of the episode -- the
    # first version of this produced episodes that barely moved.
    # The render loop sweeps the gripper through its whole range, so a pose
    # cleared with the fingers in one position can collide at another -- that
    # frame then fell back to the previous pose and showed up as frozen. Clear
    # each pose at both extremes of the travel; which end is "open" differs
    # per gripper, so testing both is what makes this sound. This is why the
    # bulky Robotiq140 combos froze worst (up to 19% of frames).
    grip_extremes = [ghi, glo] if grip_idx is not None else [None]

    def pose_ok(q):
        base_env.sim.data.qpos[jpos] = q
        for gv in grip_extremes:
            if gv is not None:
                base_env.sim.data.qpos[grip_idx] = gv
            base_env.sim.forward()
            if env_penetration(base_env) > SCENE_PENETRATION_TOL_M:
                return False, False
            if self_penetration(base_env) > SELF_PENETRATION_TOL_M:
                return False, False
        # Judge the fingertip: the eef site sits up to 44mm behind it, and
        # checking one point while ray casting from the other let the gripper
        # drift to the frame edge (IIWA+Robotiq140 landed 2px from the top).
        fpw, _ = G.fingertip_world_pose(base_env, off)
        uv = project(fpw)
        # Keep a border so the end-effector reads as "in shot" rather than
        # clipped against the edge, and so a small pose change between the
        # checked samples cannot push it out.
        mx, my = VIS_MARGIN_FRAC * w, VIS_MARGIN_FRAC * h
        if not (uv is not None and mx <= uv[0] < w - mx and my <= uv[1] < h - my):
            return True, False
        # Landing inside the frame is not the same as being seen: this test
        # used to stop here, and the eef turned out to be hidden behind the
        # arm's own links in 21-37% of accepted frames. Ray-cast to a ball of
        # points around the eef -- to the neighbourhood, not just the point,
        # so a pose whose surroundings are blocked is rejected too. Done last
        # because it is the most expensive of the checks.
        # Reject wrist orientations far from the resting one before paying for
        # ray casts: this is what keeps the poses plausible, and it is cheap.
        xm = base_env.sim.data.site_xmat[sid].reshape(3, 3)
        if tilt_from_initial_deg(xm, xmat0) > MAX_TILT_DEG:
            return True, False
        frac, _ = eef_visibility(
            base_env,
            vis_cam_id,
            fpw,
            vis_grip_bodies,
            vis_offsets,
            project=project,
            frame_wh=(w, h),
            margin_frac=VIS_MARGIN_FRAC,
        )
        return True, (frac >= VIS_CLEAR_MIN_FRAC)

    span = hi - lo
    lo_i, hi_i = lo + 0.12 * span, hi - 0.12 * span

    # Command the orientation instead of sampling and hoping. Drawing joints
    # uniformly scatters the wrist over SO(3), and a +-20deg cone is a sliver
    # of it: measured, only 0.5-1.2% of in-frame joint samples land inside the
    # cap, so rejection sampling starved the pool and the arm stood still.
    # Solving for a commanded pose instead makes every candidate satisfy the
    # orientation spec by construction, and leaves position free to range.
    p0, _ = G.fingertip_world_pose(base_env, off)
    unproject = camera_unprojection(base_env, visible_cam, h, w)
    cam_pos = np.array(base_env.sim.data.cam_xpos[vis_cam_id], dtype=float)
    d0 = float(np.linalg.norm(p0 - cam_pos))  # the arm's natural working distance
    max_tilt = np.deg2rad(MAX_TILT_DEG)
    pool, seed_q = [], None
    budget = max_tries * 30
    for attempt in range(budget):
        # Pick where the gripper should appear on screen, then solve for the
        # world point at a plausible depth. Sampling in the image guarantees
        # the target is in shot for every arm, including those whose rest pose
        # is outside the view entirely.
        mxs, mys = 1.5 * VIS_MARGIN_FRAC * w, 1.5 * VIS_MARGIN_FRAC * h
        u_t = rng.uniform(mxs, w - mxs)
        v_t = rng.uniform(mys, h - mys)
        # Wide depth range: d0 is measured to the resting fingertip, which for
        # the short arms sits outside the view, so it is only a rough guide to
        # where their reachable in-frame volume actually is.
        tgt_p = unproject(u_t, v_t, d0 * rng.uniform(0.45, 1.7))
        yaw = rng.uniform(-np.pi, np.pi)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        a, b = rng.uniform(-max_tilt, max_tilt, 2)
        ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
        rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        tgt_R = rz @ np.asarray(xmat0, dtype=float).reshape(3, 3) @ rx @ ry

        ok, _, _ = G.solve_full_pose_ik(
            base_env, tgt_p, tgt_R, off, restarts=2, seed_qpos=seed_q
        )
        if not ok:
            continue
        q = np.array(base_env.sim.data.qpos[jpos], dtype=float)
        clean, visible = pose_ok(q)
        if clean and visible:
            pool.append(q)
            seed_q = q  # nearby targets solve faster from the last hit
        if len(pool) >= 40:
            break
    if len(pool) < 2:
        # Keep the episode physically valid rather than empty. These poses
        # satisfy collision and orientation but may sit off frame.
        for _ in range(max_tries * 10):
            q = rng.uniform(lo_i, hi_i)
            if pose_ok(q)[0]:
                pool.append(q)
            if len(pool) >= 6:
                break
    if len(pool) < 2:
        pool = [np.clip((lo_i + hi_i) / 2, lo_i, hi_i)] * 2

    # order greedily so consecutive targets are close -> smooth motion
    order = [pool.pop(0)]
    while pool:
        d = [np.linalg.norm(p - order[-1]) for p in pool]
        order.append(pool.pop(int(np.argmin(d))))

    # walk between waypoints, keeping only segments that stay collision-free
    # Keep walking until the episode is actually full. Previously the loop
    # gave up once it had cycled the waypoints a few times and the remainder
    # was padded by repeating the final pose -- which is why long arms in a
    # cluttered kitchen shipped mostly-static episodes (IIWA+Robotiq140 was
    # frozen on 69% of frames). Falling back to a random pool target keeps it
    # moving when consecutive waypoints happen to be unreachable from here.
    path = []
    cur = order[0]
    wi = 1
    guard = 0
    while len(path) < n_steps and guard < 60000:
        guard += 1
        tgt = order[wi % len(order)] if guard % 3 else order[rng.integers(len(order))]
        seg = max(2, int(np.linalg.norm(tgt - cur) / 0.04))
        ok_seg = True
        cand = []
        for a in np.linspace(0, 1, seg)[1:]:
            q = cur + a * (tgt - cur)
            # Require visibility along the segment, not just at the waypoints.
            # Checking only the endpoints let long arms swing the end-effector
            # out of shot mid-segment (UR5e+Robotiq85 was in-frame on 20% of
            # frames despite every waypoint being visible).
            clean, vis = pose_ok(q)
            if not (clean and vis):
                ok_seg = False
                break
            cand.append(q)
        # Keep the valid prefix rather than discarding the whole segment.
        # All-or-nothing rejection starved the arms whose waypoints are all
        # visible but not mutually connectable in frame: UR5e+Robotiq85 ended
        # with a single pose and sat frozen for 100% of the episode.
        if cand:
            path.extend(cand)
            cur = cand[-1] if not ok_seg else tgt
        wi += 1
        if guard > 400 and not path:
            break
    if not path:
        path = [cur]
    if len(path) < n_steps and len(path) > 1:
        # Ping-pong the path instead of holding the final pose. Every pose in
        # `path` is already known collision-free and in-frame, so retracing it
        # backwards stays valid while keeping the arm in motion -- holding the
        # last pose is what produced the frozen tails (IIWA+Robotiq140 sat
        # still for 50% of its frames).
        fwd = np.array(path)
        cycle = np.vstack([fwd, fwd[-2:0:-1]]) if len(fwd) > 2 else fwd
        reps = int(np.ceil(n_steps / len(cycle)))
        path = np.vstack([cycle] * reps)
    path = np.array(path[:n_steps])
    if len(path) < n_steps:  # single valid pose: nothing better than holding it
        path = np.vstack([path, np.repeat(path[-1:], n_steps - len(path), axis=0)])

    vis_n = sum(1 for q in path[::4] if pose_ok(q)[1])
    best_frac = vis_n / len(path[::4])

    # gripper opens/closes on its own slow cycle, independent of the arm
    grip_cycle = 0.5 - 0.5 * np.cos(np.linspace(0, rng.uniform(2, 5) * np.pi, n_steps))

    buf = {
        "images": {c: [] for c in cam_names},
        "eef_pos": [],
        "eef_quat": [],
        "fingertip_pos": [],
        "grip_norm": [],
        "pos_err_m": [],
        "rot_err_deg": [],
        "reachable": [],
        "eef_visible_frac": [],
        "eef_visible_center": [],
        "eef_u": [],
        "eef_v": [],
        "opening_axis": G.measure_opening_axis(base_env),
        "fingertip_offset": off,
    }
    # seed the hold-pose with the first collision-free sample so a colliding
    # first frame still has something valid to fall back to
    last_clean = path[0].copy()
    for q in path[::4]:
        base_env.sim.data.qpos[jpos] = q
        base_env.sim.forward()
        if env_penetration(base_env) <= SCENE_PENETRATION_TOL_M:
            last_clean = q.copy()
            break
    for t in range(n_steps):
        base_env.sim.data.qpos[jpos] = path[t]
        if grip_idx is not None:
            base_env.sim.data.qpos[grip_idx] = glo + grip_cycle[t] * (ghi - glo)
        base_env.sim.data.qvel[:] = 0.0
        base_env.sim.forward()
        if env_penetration(base_env) > SCENE_PENETRATION_TOL_M:
            # Hold the last clean pose rather than render the arm inside a
            # cabinet. Never skip the frame: the writer needs exactly n_steps
            # rows per episode, and dropping frames desynchronises every
            # embodiment's video from the shared state track.
            base_env.sim.data.qpos[jpos] = last_clean
            base_env.sim.forward()
        else:
            last_clean = path[t].copy()
        imgs = G.render_cameras(env_wrapper, base_env, cam_names, h, w)
        for c in cam_names:
            buf["images"][c].append(imgs[c])
        pos, xmat = G.eef_world_pose(base_env)
        fp, _ = G.fingertip_world_pose(base_env, off)
        buf["eef_pos"].append(pos)
        buf["eef_quat"].append(xmat)
        buf["fingertip_pos"].append(fp)
        # measured on the state that is being rendered, so the flag describes
        # this exact frame
        vf, vc = eef_visibility(
            base_env,
            vis_cam_id,
            fp,
            vis_grip_bodies,
            vis_offsets,
            project=project,
            frame_wh=(w, h),
            margin_frac=VIS_MARGIN_FRAC,
        )
        buf["eef_visible_frac"].append(vf)
        buf["eef_visible_center"].append(vc)
        _uv = project(fp)
        buf["eef_u"].append(float(_uv[0]) if _uv is not None else -1.0)
        buf["eef_v"].append(float(_uv[1]) if _uv is not None else -1.0)
        buf["grip_norm"].append(G.gripper_opening_normalized(base_env))
        # Reachable by construction: the pose came from a joint configuration
        # this arm is holding, so there is no IK residual to report.
        buf["pos_err_m"].append(0.0)
        buf["rot_err_deg"].append(0.0)
        buf["reachable"].append(True)
    return buf, best_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", required=True)
    ap.add_argument(
        "--source-lerobot-dir",
        type=Path,
        default=Path(
            "/data1/local/barx/datasets/atomic_new/UR5eOmron/"
            "pretrain/PnPSinkToCounter/lerobot"
        ),
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument(
        "--camera-names",
        nargs="+",
        default=["robot0_agentview_right", "robot0_eye_in_hand"],
    )
    ap.add_argument("--camera-height", type=int, default=180)
    ap.add_argument("--camera-width", type=int, default=320)
    ap.add_argument(
        "--hide-scope",
        choices=["all_objects", "robot_only", "no_kitchen", "none"],
        default="all_objects",
    )
    ap.add_argument("--style-sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    G.patch_python_paths()
    import robocasa  # noqa: F401

    G.install_mesh_inertia_compat()
    import custom_robots  # noqa: F401

    G.install_recorded_camera_override()
    G.install_object_registry_fallback()

    env_args = json.loads(
        (args.source_lerobot_dir / "extras" / "dataset_meta.json").read_text()
    )["env_args"]
    specs = G.parse_robot_specs(args.robots)
    tags = [s.tag for s in specs]
    writer = G.CrossEmbodimentLerobotWriter(
        args.output_dir / "lerobot",
        tags,
        args.camera_names,
        args.camera_height,
        args.camera_width,
    )

    vis_cam = args.camera_names[0]
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        _, _, ep_meta, _, _ = G.load_episode(args.source_lerobot_dir, ep)
        if args.style_sweep:
            ep_meta = dict(ep_meta)
            ep_meta["style_id"] = ep % 12
        results = {}
        fracs = {}
        for spec in specs:
            env = G.create_env(
                env_args,
                spec.robot_class,
                args.camera_names,
                args.camera_height,
                args.camera_width,
                300.0,
                is_source_robot=False,
                gripper_override=spec.gripper,
            )
            w = G._EnvWithCameras(env, args.camera_names)
            w.env.set_ep_meta(dict(ep_meta))
            w.reset(unset_ep_meta=False)
            buf, frac = build_episode(
                w,
                w.env,
                spec.robot_class,
                args.steps,
                args.camera_names,
                args.camera_height,
                args.camera_width,
                rng,
                args.hide_scope,
                vis_cam,
            )
            results[spec.tag] = buf
            fracs[spec.tag] = frac
            w.env.close()
        # the writer keys its shared action/state track off tags[0]
        dummy_actions = np.zeros((args.steps, 12), dtype=np.float32)
        writer.add_episode(
            results, dummy_actions, "move the gripper through the workspace"
        )
        print(
            f"[ep {ep}] "
            + "  ".join(f"{t}:in-frame {100*fracs[t]:.0f}%" for t in tags),
            flush=True,
        )
    writer.finalize(specs[0].robot_class)
    print("done ->", args.output_dir)


if __name__ == "__main__":
    main()
