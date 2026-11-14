#!/usr/bin/env python3
"""Assemble per-episode regen bundles into LeRobot v2.1 datasets.

<work-dir>/<robot>/<task>/ep_XXXXXX/  ->  <out-root>/<robot>/<task>/lerobot/
Episodes are ordered by source episode index and renumbered 0..N-1;
regen_info.json (with source_episode) is kept under extras/episode_XXXXXX/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regen_common as RC  # noqa: E402
from regen_replay import CAMERAS, FPS  # noqa: E402


def merge_task(work_task: Path, out: Path, robot: str, task: str):
    from lerobot.datasets.compute_stats import aggregate_stats

    eps = sorted(p for p in work_task.glob("ep_*") if p.is_dir() and not p.name.endswith(".tmp"))
    if not eps:
        print(f"[merge] {robot}/{task}: no episodes")
        return 0
    src = RC.resolve_source_dir(task)
    src_info = json.load(open(src / "meta" / "info.json"))
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    # task table: language instructions first (in order of appearance), task name last
    langs = []
    for e in eps:
        l = json.load(open(e / "regen_info.json"))["lang"]
        if l not in langs:
            langs.append(l)
    task_index = {l: i for i, l in enumerate(langs)}
    name_idx = len(langs)

    episodes, ep_stats, gidx = [], [], 0
    for new_i, e in enumerate(eps):
        info = json.load(open(e / "regen_info.json"))
        df = pd.read_parquet(e / "frames.parquet")
        T = len(df)
        ti = task_index[info["lang"]]
        df.insert(0, "annotation.human.task_name", np.full(T, name_idx, dtype=np.int64))
        df.insert(0, "annotation.human.task_description", np.full(T, ti, dtype=np.int64))
        df["episode_index"] = np.full(T, new_i, dtype=np.int64)
        df["index"] = np.arange(gidx, gidx + T, dtype=np.int64)
        df["task_index"] = np.full(T, ti, dtype=np.int64)
        chunk = new_i // 1000
        dpath = out / "data" / f"chunk-{chunk:03d}" / f"episode_{new_i:06d}.parquet"
        dpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dpath, index=False)
        for cam in CAMERAS:
            vp = out / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{cam}" / f"episode_{new_i:06d}.mp4"
            vp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(e / f"{cam}.mp4", vp)
        xd = out / "extras" / f"episode_{new_i:06d}"
        shutil.copytree(e / "extras", xd)
        shutil.copy2(e / "regen_info.json", xd / "regen_info.json")
        st = json.load(open(e / "stats.json"))
        for k, v in {"annotation.human.task_description": ti, "annotation.human.task_name": name_idx,
                     "episode_index": new_i, "task_index": ti}.items():
            st[k] = {"min": [v], "max": [v], "mean": [float(v)], "std": [0.0], "count": [T]}
        st["index"] = {"min": [gidx], "max": [gidx + T - 1], "mean": [gidx + (T - 1) / 2],
                       "std": [float(np.std(np.arange(T)))], "count": [T]}
        ep_stats.append({"episode_index": new_i, "stats": st})
        episodes.append({"episode_index": new_i, "tasks": [info["lang"]], "length": T})
        gidx += T

    meta = out / "meta"
    with open(meta / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in episodes)
    with open(meta / "episodes_stats.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in ep_stats)
    with open(meta / "tasks.jsonl", "w") as f:
        for l in langs:
            f.write(json.dumps({"task_index": task_index[l], "task": l}) + "\n")
        f.write(json.dumps({"task_index": name_idx, "task": task}) + "\n")
    agg = aggregate_stats([{k: {s: np.asarray(v) for s, v in d.items()} for k, d in x["stats"].items()}
                           for x in ep_stats])
    json.dump({k: {s: np.asarray(v).tolist() for s, v in d.items()} for k, d in agg.items()},
              open(meta / "stats.json", "w"), indent=4)
    shutil.copy2(src / "meta" / "modality.json", meta / "modality.json")
    emb = json.load(open(src / "meta" / "embodiment.json"))
    emb.update(robot_name=robot, robot_type=robot, embodiment_tag=f"robocasa_{robot.lower()}")
    json.dump(emb, open(meta / "embodiment.json", "w"), indent=4)
    info = dict(src_info)
    n_eps = len(eps)
    info.update(robot_type=robot, total_episodes=n_eps, total_frames=gidx, total_tasks=len(langs) + 1,
                total_videos=n_eps * len(CAMERAS), total_chunks=(n_eps - 1) // 1000 + 1,
                splits={"train": f"0:{n_eps}"}, fps=FPS)
    json.dump(info, open(meta / "info.json", "w"), indent=4)
    # dataset-level extras (env args for the target robot)
    dm = json.load(open(src / "extras" / "dataset_meta.json"))
    dm["env_args"]["env_kwargs"]["robots"] = robot
    dm["env_args"]["env_kwargs"]["gripper_types"] = json.load(open(eps[0] / "regen_info.json"))["target_gripper"]
    dm["total"] = gidx
    dm["regen_source"] = str(src)
    json.dump(dm, open(out / "extras" / "dataset_meta.json", "w"), indent=4)
    print(f"[merge] {robot}/{task}: {n_eps} episodes, {gidx} frames -> {out}")
    return n_eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--robots", nargs="+", default=["UR5eOmron"])
    p.add_argument("--tasks", nargs="+", default=None)
    a = p.parse_args()
    for robot in a.robots:
        tasks = a.tasks or sorted(d.name for d in (a.work_dir / robot).iterdir() if d.is_dir())
        for task in tasks:
            merge_task(a.work_dir / robot / task, a.out_root / robot / task / "lerobot", robot, task)


if __name__ == "__main__":
    main()
