#!/usr/bin/env python3
"""Regenerate RoboCasa v1.0 atomic human demos on another robot (one worker).

For each requested source episode:
  1. restore the source PandaOmron episode from extras/ (model.xml + states + ep_meta);
  2. build the target scene from the same ep_meta, copy every non-robot body/joint
     pose from the source, align base x/y/yaw, IK the arm to the source EEF pose;
  3. closed-loop OSC tracking of the source EEF (grasp-center + opening-axis
     aligned), base/torso/gripper commands copied from the source action;
  4. retry ladder: base back-off 0..30 cm (5 cm steps) x finger back-off
     0/1.5/3 cm along the approach axis; first attempt that passes
     env.is_success() AND the jump filter (base > 5 cm or EEF > 10 cm in a step)
     is kept;
  5. re-render the kept trajectory from its states (3 cams, 256x256) and write a
     per-episode bundle (parquet rows w/o global indices, 3 mp4s, extras,
     regen_info.json, episode stats). merge_all.py assembles LeRobot v2.1.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

import regen_common as RC  # noqa: E402  (also puts datagen/smoke_tests on sys.path)
from dataset_reproduce import (  # noqa: E402
    init_obs_utils,
    load_episode_payload,
    load_modality,
    read_episode_df,
    build_target_env_meta,
    create_env,
    reset_source_to_episode,
    robot_eef_pose,
    reorder_lerobot_actions_to_hdf5,
)
from test_dataset import (  # noqa: E402
    sync_controller_goals,
    retarget_action,
    source_eef_pose_from_saved_state,
)
from reproduce_ur5e_dataset import (  # noqa: E402
    place_target_at_backoff,
    grasp_center_in_site_frame,
    gripper_twist_matrix,
    base_world_pose,
)
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils  # noqa: E402
from robocasa.utils.lerobot_utils import reorder_hdf5_action, reorder_hdf5_state  # noqa: E402

CAMERAS = ["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]
FPS = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--target-robot", default="UR5eOmron")
    p.add_argument("--target-gripper", default=None)
    p.add_argument("--work-dir", type=Path, required=True,
                   help="per-episode bundles go to <work-dir>/<robot>/<task>/ep_XXXXXX")
    p.add_argument("--max-base-backoff", type=float, default=0.30)
    p.add_argument("--base-backoff-step", type=float, default=0.05)
    p.add_argument("--finger-backoffs", type=float, nargs="+", default=[0.0, 0.015, 0.03])
    p.add_argument("--jump-base", type=float, default=0.05)
    p.add_argument("--jump-eef", type=float, default=0.10)
    p.add_argument("--eef-gain", type=float, default=1.0)
    p.add_argument("--eef-rot-gain", type=float, default=2.5)
    p.add_argument("--grasp-depth-extra", type=float, default=0.01)
    p.add_argument("--ik-iters", type=int, default=250)
    p.add_argument("--img", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-failed-preview", action="store_true",
                   help="also write an agentview mp4 of the best failed attempt (debug)")
    return p.parse_args()


def eef_world(env):
    return robot_eef_pose(env)[0]


def rollout(source_env, target_env, source_states, source_actions, ep_meta, lowdim0,
            base_backoff, finger_backoff, grasp_center_src, args):
    src_pos0, src_xmat0 = robot_eef_pose(source_env)
    ik_res, ncon = place_target_at_backoff(
        target_env, ep_meta, lowdim0, base_backoff, src_pos0, src_xmat0, args,
        source_env=source_env)
    if ik_res > 2e-3 or ncon > 0:
        return dict(skipped=True, ik_residual=ik_res, init_contacts=ncon)

    model_file = target_env.get_state()["model"]
    init_state = target_env.get_state()["states"].copy()
    grasp_offset = (grasp_center_src - grasp_center_in_site_frame(target_env)
                    + np.array([0.0, 0.0, args.grasp_depth_extra - finger_backoff]))
    twist = gripper_twist_matrix(source_env, target_env)

    states, actions, track = [init_state], [], []
    base_prev, _ = base_world_pose(target_env)
    eef_prev = eef_world(target_env)
    max_base_step = max_eef_step = 0.0
    for t, a_src in enumerate(source_actions):
        d_pos, d_xmat = source_eef_pose_from_saved_state(source_env, source_states, t, a_src)
        d_pos = d_pos + d_xmat @ grasp_offset
        d_xmat = d_xmat @ twist
        a_tgt = retarget_action(a_src, source_env.env.robots[0], target_env, d_pos, d_xmat,
                                gain=args.eef_gain, rot_gain=args.eef_rot_gain)
        target_env.step(a_tgt)
        actions.append(a_tgt)
        states.append(target_env.get_state()["states"].copy())
        bp, _ = base_world_pose(target_env)
        ep = eef_world(target_env)
        max_base_step = max(max_base_step, float(np.linalg.norm(bp[:2] - base_prev[:2])))
        max_eef_step = max(max_eef_step, float(np.linalg.norm(ep - eef_prev)))
        base_prev, eef_prev = bp, ep
        track.append(float(np.linalg.norm(d_pos - ep)))
    success = bool(target_env.is_success().get("task", False))
    # restore source for the next attempt
    source_env.reset_to({"states": np.asarray(source_states[0]).reshape(-1)})
    jump = max_base_step > args.jump_base or max_eef_step > args.jump_eef
    return dict(skipped=False, success=success, jump=jump, ik_residual=ik_res,
                max_base_step=max_base_step, max_eef_step=max_eef_step,
                track_mean=float(np.mean(track)), track_max=float(np.max(track)),
                states=np.asarray(states), actions=np.asarray(actions), model_file=model_file)


def gripper_driver_idx(env):
    """Indices (into robot0_gripper_qpos) of the two actuated finger joints, so the
    16-D state keeps a 2-D gripper_qpos like the PandaOmron source."""
    robot = env.env.robots[0]
    names = list(robot.gripper[robot.arms[0]].joints)
    acts = robot.gripper[robot.arms[0]].actuators
    sim = env.env.sim
    act_joints = []
    for a in acts:
        aid = sim.model.actuator_name2id(a)
        act_joints.append(sim.model.joint_id2name(int(sim.model.actuator_trnid[aid][0])))
    idx = [names.index(j) for j in act_joints if j in names]
    return idx[:2] if len(idx) >= 2 else [0, 1]


def render_episode(render_env, states, model_file, ep_meta):
    """Re-render a trajectory from its sim states (robocasa convert_hdf5_lerobot style).
    states has T+1 entries (s_0..s_T); the observation for step t is s_t."""
    render_env.reset()
    render_env.reset_to({"states": states[0], "model": model_file, "ep_meta": json.dumps(ep_meta)})
    obs_list, rewards, succ = [], [], []
    for t in range(len(states) - 1):
        obs = render_env.reset_to({"states": states[t]})
        obs_list.append({k: np.asarray(v).copy() for k, v in obs.items()
                         if k.endswith("_image") or k.startswith("robot0_")})
        rewards.append(float(render_env.get_reward()))
    return obs_list, rewards


def encode_mp4(path, frames, fps=FPS):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "23", "g": "2"}
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def write_bundle(out_dir, ep_num, src_dir, att, obs_list, rewards, ep_meta, modality,
                 gidx, base_backoff, finger_backoff, args, source_len):
    import pandas as pd
    from lerobot.datasets.compute_stats import compute_episode_stats

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    T = len(obs_list)
    obs = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    obs["robot0_gripper_qpos"] = obs["robot0_gripper_qpos"][:, gidx]
    state = reorder_hdf5_state(obs, modality)
    action = reorder_hdf5_action(att["actions"], modality)
    assert state.shape == (T, 16) and action.shape == (T, 12), (state.shape, action.shape)
    done = np.zeros(T, dtype=bool)
    done[-1] = True
    df = pd.DataFrame({
        "observation.state": list(state),
        "action": list(action),
        "next.reward": np.asarray(rewards, dtype=np.float32),
        "next.done": done,
        "timestamp": (np.arange(T) / FPS).astype(np.float32),
        "frame_index": np.arange(T, dtype=np.int64),
    })
    df.to_parquet(tmp / "frames.parquet")
    for cam in CAMERAS:
        frames = obs[f"{cam}_image"]
        if frames.dtype != np.uint8:
            frames = np.clip(frames * (255 if frames.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        obs[f"{cam}_image"] = frames
        encode_mp4(tmp / f"{cam}.mp4", frames)
    # episode stats (lerobot v2.1 layout); images as (T,C,H,W) uint8 like the recorder
    ep_data = {
        "observation.state": state, "action": action,
        "next.reward": np.asarray(rewards, dtype=np.float32)[:, None],
        "next.done": done[:, None],
        "timestamp": (np.arange(T) / FPS).astype(np.float32)[:, None],
        "frame_index": np.arange(T)[:, None],
    }
    feats = {k: {"dtype": "float64", "shape": (v.shape[1],)} for k, v in ep_data.items()}
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        ep_data[key] = obs[f"{cam}_image"].transpose(0, 3, 1, 2)
        feats[key] = {"dtype": "video", "shape": (args.img, args.img, 3)}
    stats = compute_episode_stats(ep_data, feats)
    stats = {k: {s: np.asarray(v).tolist() for s, v in d.items()} for k, d in stats.items()}
    json.dump(stats, open(tmp / "stats.json", "w"))
    # extras: target-robot replay payload
    ex = tmp / "extras"
    ex.mkdir()
    np.savez_compressed(ex / "states.npz", states=att["states"][:-1])
    with gzip.open(ex / "model.xml.gz", "wt") as f:
        f.write(att["model_file"])
    json.dump(ep_meta, open(ex / "ep_meta.json", "w"), indent=4)
    info = dict(
        source_dataset=str(src_dir), source_episode=int(ep_num), source_length=int(source_len),
        length=int(T), task=args.task, lang=ep_meta.get("lang"),
        target_robot=args.target_robot, target_gripper=args.target_gripper,
        finger_kp=RC.FINGER_KP, pad_friction=RC.PAD_FRICTION,
        base_backoff_m=float(base_backoff), finger_backoff_m=float(finger_backoff),
        gripper_qpos_driver_idx=[int(i) for i in gidx],
        ik_residual_m=att["ik_residual"], track_mean_m=att["track_mean"],
        track_max_m=att["track_max"], max_base_step_m=att["max_base_step"],
        max_eef_step_m=att["max_eef_step"],
    )
    json.dump(info, open(tmp / "regen_info.json", "w"), indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)


def main():
    args = parse_args()
    if args.target_gripper is None:
        args.target_gripper = RC.ROBOT_GRIPPERS[args.target_robot]
    init_obs_utils()
    src_dir = RC.resolve_source_dir(args.task)
    dmeta = json.load(open(src_dir / "extras" / "dataset_meta.json"))
    src_env_meta = dmeta["env_args"]
    tgt_env_meta = build_target_env_meta(src_env_meta, args.target_robot, args.target_gripper)
    modality = load_modality(src_dir)
    task_dir = args.work_dir / args.target_robot / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    source_env = create_env(src_env_meta, seed=args.seed)
    target_env = create_env(tgt_env_meta, seed=args.seed)
    render_env = EnvUtils.create_env_for_data_processing(
        env_meta=tgt_env_meta, camera_names=CAMERAS, camera_height=args.img,
        camera_width=args.img, reward_shaping=False)

    base_ladder = list(np.arange(0.0, args.max_base_backoff + 1e-9, args.base_backoff_step))
    for ep_num in args.episodes:
        out_dir = task_dir / f"ep_{ep_num:06d}"
        res_path = task_dir / f"ep_{ep_num:06d}.result.json"
        if res_path.exists():
            continue
        t0 = time.time()
        result = dict(episode=ep_num, success=False, attempts=[])
        try:
            pq = next(src_dir.glob(f"data/chunk-*/episode_{ep_num:06d}.parquet"))
            df = read_episode_df(pq)
            src_states, ep_meta, model_xml = load_episode_payload(src_dir, ep_num)
            src_actions = reorder_lerobot_actions_to_hdf5(np.stack(df["action"].to_numpy()), modality)
            lowdim0 = np.asarray(df.iloc[0]["observation.state"], dtype=np.float64)
            reset_source_to_episode(source_env, src_states[0], ep_meta, model_xml)
            sync_controller_goals(source_env)
            gc_src = grasp_center_in_site_frame(source_env)
            kept = None
            for bb in base_ladder:
                for fb in args.finger_backoffs:
                    att = rollout(source_env, target_env, src_states, src_actions, ep_meta,
                                  lowdim0, bb, fb, gc_src, args)
                    meta = {k: v for k, v in att.items() if k not in ("states", "actions", "model_file")}
                    meta.update(base_backoff=float(bb), finger_backoff=float(fb))
                    result["attempts"].append(meta)
                    if att["skipped"]:
                        break  # finger back-off does not change initial feasibility
                    if att["success"] and not att["jump"]:
                        kept = (att, bb, fb)
                        break
                if kept:
                    break
            if kept:
                att, bb, fb = kept
                gidx = gripper_driver_idx(target_env)
                obs_list, rewards = render_episode(render_env, att["states"], att["model_file"], ep_meta)
                write_bundle(out_dir, ep_num, src_dir, att, obs_list, rewards, ep_meta, modality,
                             gidx, bb, fb, args, len(src_actions))
                result.update(success=True, base_backoff=float(bb), finger_backoff=float(fb))
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        result["seconds"] = round(time.time() - t0, 1)
        json.dump(result, open(res_path, "w"), indent=1)
        print(f"[{args.target_robot}/{args.task}] ep {ep_num:06d} success={result['success']} "
              f"attempts={len(result['attempts'])} {result['seconds']}s", flush=True)

    for e in (source_env, target_env, render_env):
        try:
            e.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
