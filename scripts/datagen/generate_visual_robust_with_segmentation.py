#!/usr/bin/env python3
"""Generate a visual-robust cross-embodiment dataset with robot segmentation
masks produced in the SAME pass as the RGB.

Why regenerate instead of annotating the published `visual_robust_new`:
recovering a mask for an existing video requires reproducing the exact sim
state behind every frame, and that turned out to be impossible. Replaying the
recorded actions open-loop diverges (measured: image difference grows from ~4
to ~40 over an episode, and only ~40% of frames stay close enough to trust),
and restoring the recorded per-frame states does not help either, because the
published video itself drifted away from those states when it was generated --
its late frames correspond to no recorded state at all. The published pixels
are the only record of what its robot did, and they are not invertible to a
pose.

Generating both together sidesteps the problem entirely: the mask describes
the very render it was taken from, so there is nothing to reproduce and
nothing to drift. Everything else (IK retargeting, camera world-pinning,
object hiding, the LeRobot schema) is the existing, validated generator --
this module only adds a segmentation capture and two extra video tracks.

Output: the normal dataset plus, per embodiment tag and camera, a
`observation.images.<tag>.<cam>_segmentation` video holding a 0/255 robot mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "datagen"))

import generate_visual_robust_cross_embodiment_x as gen  # noqa: E402
from add_robot_segmentation import robot_geom_ids, write_mask_video  # noqa: E402

SEG_SUFFIX = "_segmentation"

# Filled by the render_cameras patch, drained by the replay_embodiment patch.
_MASKS: dict[str, list] = {}


def _patched_render_cameras(env_wrapper, base_env, camera_names, camera_h, camera_w):
    images = gen._orig_render_cameras(
        env_wrapper, base_env, camera_names, camera_h, camera_w
    )
    sim = base_env.sim
    # Recomputed per call: replay_embodiment resets the env after create_env,
    # which rebuilds sim/model and dangles any geom-id set captured earlier.
    rgeoms = list(robot_geom_ids(sim.model))
    for cam in camera_names:
        seg = sim.render(
            width=camera_w, height=camera_h, camera_name=cam, segmentation=True
        )
        # flipud to match how render() hands back the colour buffer, so mask
        # pixel (i, j) is the same pixel as image (i, j).
        _MASKS.setdefault(cam, []).append(np.flipud(np.isin(seg[..., 1], rgeoms)))
    return images


def _patched_replay_embodiment(*args, **kwargs):
    _MASKS.clear()
    buf = gen._orig_replay_embodiment(*args, **kwargs)
    buf["masks"] = {cam: list(frames) for cam, frames in _MASKS.items()}
    _MASKS.clear()
    n_img = len(buf["images"][next(iter(buf["images"]))])
    for cam, frames in buf["masks"].items():
        if len(frames) != n_img:
            raise RuntimeError(
                f"segmentation capture desynchronised for {cam}: {len(frames)} masks "
                f"vs {n_img} rendered frames"
            )
    return buf


def _patched_add_episode(self, results, raw_actions, lang):
    ep_idx = self.episode_index  # read before the original increments it
    chunk = ep_idx // gen.CHUNK_SIZE
    gen._orig_add_episode(self, results, raw_actions, lang)
    ep_len = raw_actions.shape[0]
    for tag in self.tags:
        masks = results[tag].get("masks", {})
        for cam in self.camera_names:
            frames = masks.get(cam)
            if not frames:
                raise RuntimeError(f"no segmentation captured for {tag}.{cam}")
            out_dir = self.video_dir / f"chunk-{chunk:03d}" / f"{tag}.{cam}{SEG_SUFFIX}"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_mask_video(out_dir / f"episode_{ep_idx:06d}.mp4", frames[:ep_len])


def _patched_finalize(self, source_robot):
    gen._orig_finalize(self, source_robot)
    info_path = self.meta_dir / "info.json"
    info = json.loads(info_path.read_text())
    total_new = 0
    for tag in self.tags:
        for cam in self.camera_names:
            base_key = f"observation.images.{tag}.{cam}"
            seg_key = base_key + SEG_SUFFIX
            if base_key in info["features"] and seg_key not in info["features"]:
                info["features"][seg_key] = json.loads(
                    json.dumps(info["features"][base_key])
                )
                total_new += 1
    info["total_videos"] = info.get("total_videos", 0) + total_new * info.get(
        "total_episodes", 0
    )
    info_path.write_text(json.dumps(info, indent=2))
    print(f"[seg] registered {total_new} segmentation video tracks in info.json")


def install():
    gen._orig_render_cameras = gen.render_cameras
    gen._orig_replay_embodiment = gen.replay_embodiment
    gen._orig_add_episode = gen.CrossEmbodimentLerobotWriter.add_episode
    gen._orig_finalize = gen.CrossEmbodimentLerobotWriter.finalize

    gen.render_cameras = _patched_render_cameras
    gen.replay_embodiment = _patched_replay_embodiment
    gen.CrossEmbodimentLerobotWriter.add_episode = _patched_add_episode
    gen.CrossEmbodimentLerobotWriter.finalize = _patched_finalize


if __name__ == "__main__":
    install()
    gen.main()
