#!/usr/bin/env python
"""Merge N parallel-GPU shard outputs of generate_visual_robust_cross_embodiment.py
back into a single LeRobot dataset with correct global episode_index/index/
task_index and a consolidated tasks.jsonl/episodes.jsonl/info.json.

Each shard was launched with a disjoint, contiguous, ascending block of
source `--episodes` (e.g. shard0=0..26, shard1=27..53, ...), so within a
shard, local episode position i (0-based, in completion order == submission
order since main() processes `episodes` in list order) maps 1:1 to the
global/source episode index `shard_start + i`. We rely on that to rename
files; we do NOT try to recover the mapping from content alone.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def merge(shard_dirs_and_starts, out_dir: Path, examples_from: Path | None):
    out_lerobot = out_dir / "lerobot"
    data_dir = out_lerobot / "data" / "chunk-000"
    video_dir = out_lerobot / "videos" / "chunk-000"
    meta_dir = out_lerobot / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    global_task_to_id: dict[str, int] = {}
    total_frames = 0
    episodes_rows = []
    info_template = None

    for shard_dir, start in shard_dirs_and_starts:
        shard_lerobot = shard_dir / "lerobot"
        shard_tasks = {}
        with open(shard_lerobot / "meta" / "tasks.jsonl") as f:
            for line in f:
                row = json.loads(line)
                shard_tasks[row["task_index"]] = row["task"]
        remap = {}
        for local_idx, task in shard_tasks.items():
            if task not in global_task_to_id:
                global_task_to_id[task] = len(global_task_to_id)
            remap[local_idx] = global_task_to_id[task]

        shard_episodes = {}
        with open(shard_lerobot / "meta" / "episodes.jsonl") as f:
            for line in f:
                row = json.loads(line)
                shard_episodes[row["episode_index"]] = row

        local_indices = sorted(shard_episodes.keys())
        for local_idx in local_indices:
            global_idx = start + local_idx
            ep_row = shard_episodes[local_idx]
            ep_len = ep_row["length"]

            src_parquet = (
                shard_lerobot
                / "data"
                / "chunk-000"
                / f"episode_{local_idx:06d}.parquet"
            )
            df = pd.read_parquet(src_parquet)
            df["episode_index"] = global_idx
            df["task_index"] = df["task_index"].map(remap)
            df["index"] = df["index"] - df["index"].iloc[0] + total_frames
            df.to_parquet(data_dir / f"episode_{global_idx:06d}.parquet")

            for tag_cam_dir in (shard_lerobot / "videos" / "chunk-000").iterdir():
                if not tag_cam_dir.is_dir():
                    continue
                src_video = tag_cam_dir / f"episode_{local_idx:06d}.mp4"
                if not src_video.exists():
                    continue
                dst_dir = video_dir / tag_cam_dir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_dir / f"episode_{global_idx:06d}.mp4")

            episodes_rows.append(
                {
                    "episode_index": global_idx,
                    "tasks": ep_row["tasks"],
                    "length": ep_len,
                }
            )
            total_frames += ep_len

        shard_info = json.loads((shard_lerobot / "meta" / "info.json").read_text())
        if info_template is None:
            info_template = shard_info

    episodes_rows.sort(key=lambda r: r["episode_index"])
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for row in episodes_rows:
            f.write(json.dumps(row) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(global_task_to_id.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    info = dict(info_template)
    n_ep = len(episodes_rows)
    n_tags = len(info_template["embodiment_tags"])
    n_cams = len(info_template["features"]) and sum(
        1 for k in info_template["features"] if k.startswith("observation.images.")
    ) // max(n_tags, 1)
    info["total_episodes"] = n_ep
    info["total_frames"] = total_frames
    info["total_tasks"] = len(global_task_to_id)
    info["total_videos"] = n_ep * n_tags * n_cams
    info["total_chunks"] = (max(n_ep - 1, 0) // info_template["chunks_size"]) + 1
    info["splits"] = {"train": f"0:{n_ep}"}
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    if examples_from is not None and examples_from.exists():
        shutil.copytree(examples_from, out_dir / "examples", dirs_exist_ok=True)

    print(
        f"[merge] {len(episodes_rows)} episodes, {total_frames} frames, "
        f"{len(global_task_to_id)} tasks -> {out_lerobot}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shard-dirs",
        nargs="+",
        required=True,
        help="shard output dirs in order, e.g. .../PickPlaceSinkToCounter_shard0 ...",
    )
    p.add_argument(
        "--shard-starts",
        nargs="+",
        type=int,
        required=True,
        help="global episode-index start for each shard dir, same order/length as --shard-dirs",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    assert len(args.shard_dirs) == len(args.shard_starts)
    shard_dirs_and_starts = [
        (Path(d), s) for d, s in zip(args.shard_dirs, args.shard_starts)
    ]
    examples_from = shard_dirs_and_starts[0][0] / "examples"
    merge(shard_dirs_and_starts, args.output_dir, examples_from)


if __name__ == "__main__":
    main()
