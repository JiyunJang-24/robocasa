#!/usr/bin/env python
"""Add a new embodiment's video tracks (already generated as its own
Panda+<new_robot> LeRobot dataset, e.g. via merge_visual_robust_shards.py)
into an EXISTING visual-robust cross-embodiment LeRobot dataset, without
touching the existing data/ (parquet) files -- the shared observation.state/
action/task_index columns are keyed off the (already-present) source robot
and are untouched; we only add new observation.images.<new_robot>.<cam>
video tracks and update meta/info.json accordingly.

Assumes both datasets were generated from the same source episodes in the
same order (0..N-1), so episode_index in the new dataset lines up 1:1 with
episode_index in the existing dataset -- we do NOT try to match content.
"""
import argparse
import json
import shutil
from pathlib import Path


def add_embodiment(existing_dir: Path, new_dir: Path, new_robot: str):
    existing_video_dir = existing_dir / "videos" / "chunk-000"
    new_video_dir = new_dir / "videos" / "chunk-000"

    new_tag_dirs = [
        d
        for d in new_video_dir.iterdir()
        if d.is_dir() and d.name.startswith(f"{new_robot}.")
    ]
    if not new_tag_dirs:
        raise ValueError(f"No {new_robot}.* video dirs found under {new_video_dir}")

    n_copied_total = 0
    for src_tag_dir in new_tag_dirs:
        dst_tag_dir = existing_video_dir / src_tag_dir.name
        dst_tag_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in src_tag_dir.glob("episode_*.mp4"):
            shutil.copy2(f, dst_tag_dir / f.name)
            n += 1
        print(f"  {src_tag_dir.name}: copied {n} videos")
        n_copied_total += n

    info_path = existing_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if new_robot in info["embodiment_tags"]:
        print(f"  [skip] {new_robot} already in embodiment_tags")
        return

    new_info = json.loads((new_dir / "meta" / "info.json").read_text())
    prefix = f"observation.images.{new_robot}."
    cams = sorted(
        k[len(prefix) :] for k in new_info["features"] if k.startswith(prefix)
    )
    for cam in cams:
        key = f"observation.images.{new_robot}.{cam}"
        info["features"][key] = new_info["features"][key]

    info["embodiment_tags"].append(new_robot)
    info["robot_type"] = f"cross_embodiment[{','.join(info['embodiment_tags'])}]"
    n_tags = len(info["embodiment_tags"])
    n_cams_per_tag = sum(
        1
        for k in info["features"]
        if k.startswith(f"observation.images.{info['embodiment_tags'][0]}.")
    )
    info["total_videos"] = info["total_episodes"] * n_tags * n_cams_per_tag
    info_path.write_text(json.dumps(info, indent=2))
    print(
        f"  meta/info.json updated: embodiment_tags={info['embodiment_tags']}, "
        f"total_videos={info['total_videos']}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--existing-lerobot-dir", required=True, type=Path)
    p.add_argument("--new-lerobot-dir", required=True, type=Path)
    p.add_argument("--new-robot", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(
        f"[add-embodiment] {args.new_robot}: {args.new_lerobot_dir} -> {args.existing_lerobot_dir}"
    )
    add_embodiment(args.existing_lerobot_dir, args.new_lerobot_dir, args.new_robot)


if __name__ == "__main__":
    main()
