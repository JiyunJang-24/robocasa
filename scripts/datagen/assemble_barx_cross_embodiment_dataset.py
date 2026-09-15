#!/usr/bin/env python3
"""
Assemble the generated MimicGen outputs into a single cross-embodiment dataset
tree with a manifest, in the spirit of BARX's dataset/manifest.csv.

Scans <root>/<Robot>/pretrain/<Task>/<Task>_<Robot>/<timestamp>/demo.hdf5,
keeps the newest run per (robot, task), and writes:

    <root>/manifest.csv    one row per hdf5 (embodiment, task, demos, bytes, sha256, path)
    <root>/summary.json    success stats per (robot, task) from important_stats.json
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py

ROOT = Path("/data1/local/BARX/datasets/cross_embodiment")

# robosuite robot class -> BARX paper embodiment name
EMBODIMENT_NAMES = {
    "PandaOmron": "Panda",
    "UR5eOmron": "UR5e",
    "IIWAOmron": "IIWA",
    "Kinova3Omron": "Kinova3",
    "JacoOmron": "Jaco",
}

TASK_NAMES = {
    "TurnOnSinkFaucet": "Turn On Sink Faucet",
    "PickPlaceCounterToSink": "PnP Counter to Sink",
    "PickPlaceSinkToCounter": "PnP Sink to Counter",
    "AdjustToasterOvenTemperature": "Adjust Toaster Oven Temperature (bonus)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_runs(robot_dir: Path, task: str, robot: str) -> list[Path]:
    base = robot_dir / "pretrain" / task / f"{task}_{robot}"
    if not base.exists():
        return []
    return sorted(
        p for p in base.iterdir() if p.is_dir() and (p / "demo.hdf5").exists()
    )


def main() -> None:
    rows = []
    summary = {}
    for robot_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        robot = robot_dir.name
        if robot not in EMBODIMENT_NAMES:
            continue
        for task in TASK_NAMES:
            runs = completed_runs(robot_dir, task, robot)
            if not runs:
                continue
            combo_runs = []
            combo_demos = 0
            for run in runs:
                demo_path = run / "demo.hdf5"
                with h5py.File(demo_path, "r") as f:
                    demos = sorted(f["data"].keys())
                    num_demos = len(demos)
                    action_dim = f[f"data/{demos[0]}/actions"].shape[1] if demos else 0
                if num_demos == 0:
                    continue
                stats_path = run / "important_stats.json"
                stats = json.load(open(stats_path)) if stats_path.exists() else {}
                videos = (
                    sorted((run / "videos").glob("*.mp4"))
                    if (run / "videos").exists()
                    else []
                )
                rows.append(
                    {
                        "release_group": "cross-embodiment (generated)",
                        "embodiment": EMBODIMENT_NAMES[robot],
                        "task": TASK_NAMES[task],
                        "demonstrations": num_demos,
                        "action_dim": action_dim,
                        "bytes": demo_path.stat().st_size,
                        "sha256": sha256(demo_path),
                        "relative_path": str(demo_path.relative_to(ROOT)),
                    }
                )
                combo_demos += num_demos
                combo_runs.append(
                    {
                        "run_dir": str(run.relative_to(ROOT)),
                        "successful_demos": num_demos,
                        "videos": [str(v.relative_to(ROOT)) for v in videos],
                        "mimicgen_stats": stats,
                    }
                )
            if combo_runs:
                summary[f"{EMBODIMENT_NAMES[robot]}/{TASK_NAMES[task]}"] = {
                    "successful_demos_total": combo_demos,
                    "runs": combo_runs,
                }

    manifest_path = ROOT / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {manifest_path} ({len(rows)} rows)")
    for row in rows:
        print(
            f"  {row['embodiment']:8s} {row['task']:22s} demos={row['demonstrations']} "
            f"action_dim={row['action_dim']}"
        )


if __name__ == "__main__":
    main()
