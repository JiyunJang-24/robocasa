#!/usr/bin/env python3
"""
Build every dataset file listed in BARX/dataset/manifest.csv.

For each manifest row this orchestrator produces
<raw-root>/<relative_path> in the normalized BARX HDF5 layout:

- mg/... rows: MimicGen generation (generate_barx_cross_embodiment_dataset.py)
  from the RoboCasa v1.0 human source demos, retried in passes until the shard
  has its demonstration count (or --demos-cap / --max-passes bounds), then
  rendered with render_barx_normalized_hdf5.py.
- human/... rows: the RoboCasa v1.0 human (non-MimicGen) demonstrations are
  converted directly. The v1.0 human data is PandaOmron + PandaGripper, which
  is exactly BARX's Panda-OG embodiment, so human/PandaOGGripperOmron is a real
  conversion and human/PandaOmron + human/JacoOmron are symbolic links to it
  (no Robotiq85-Panda or Jaco human teleoperation exists on this machine).
- Flip Mug Upright rows are reported as unbuildable: it is BARX's new task and
  has no source demonstrations in any local archive.

Run with the conda robocasa python. Parallelize with --num-workers/--worker-index
(one process per GPU), e.g.:

    for i in 0 1 2 3; do
      MUJOCO_EGL_DEVICE_ID=$i /home/fick17/anaconda3/envs/robocasa/bin/python \
        build_all_manifest_datasets.py --num-workers 4 --worker-index $i &
    done

The full-fidelity run (100 demos per mg shard) is a multi-week compute job at
the observed MimicGen success rates; use --demos-cap for scaled builds. The
same command is safe to re-run: existing outputs are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py

ROOT = Path("/data1/local/robocasa")
PYTHON = sys.executable
MANIFEST = Path("/data1/local/BARX/dataset/manifest.csv")
DEFAULT_RAW_ROOT = Path("/data1/local/BARX/datasets/barx-raw")
DEFAULT_RUNS_ROOT = Path("/data1/local/BARX/datasets/mg_runs")
GENERATOR = ROOT / "generate_barx_cross_embodiment_dataset.py"
RENDERER = ROOT / "render_barx_normalized_hdf5.py"
V1_SOURCE_ROOT = ROOT / "datasets" / "v1.0"

TASK_TO_SOURCE = {
    "PnP Counter to Sink": "PickPlaceCounterToSink",
    "PnP Sink to Counter": "PickPlaceSinkToCounter",
    "Turn On Sink Faucet": "TurnOnSinkFaucet",
    "Flip Mug Upright": None,  # new BARX task: no source demos exist locally
}
PNP_TASKS = {"PnP Counter to Sink", "PnP Sink to Counter"}

# manifest robot directory -> (generator --target-robot, --target-gripper)
ROBOT_DIR_TO_TARGET = {
    "IIWAOmron": ("IIWAOmron", "default"),  # Robotiq140
    "Kinova3Omron": ("Kinova3Omron", "default"),  # Robotiq85
    "UR5eOmron": ("UR5eOmron", "default"),  # Robotiq85
    "PandaOmron": ("PandaOmron", "Robotiq85Gripper"),  # BARX "Panda"
    "PandaOGOmron": ("PandaOmron", "default"),  # BARX "Panda-OG"
    "JacoOmron": ("JacoOmron", "default"),  # Robotiq85 (BARX class)
}

HUMAN_CANONICAL_DIR = "PandaOGGripperOmron"  # v1.0 human demos: Panda + PandaGripper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--demos-cap",
        type=int,
        default=None,
        help="Cap demos per shard (default: manifest count)",
    )
    parser.add_argument(
        "--attempt-factor",
        type=float,
        default=8.0,
        help="MimicGen attempts per still-missing demo per pass",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Generation passes (new seeds) per mg shard",
    )
    parser.add_argument("--max-attempts-per-pass", type=int, default=60)
    parser.add_argument(
        "--tasks", nargs="*", default=None, help="Filter by manifest task names"
    )
    parser.add_argument(
        "--robots",
        nargs="*",
        default=None,
        help="Filter by manifest robot directory names",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--skip-mg", action="store_true")
    parser.add_argument("--skip-human", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest() -> list[dict]:
    with MANIFEST.open(newline="") as f:
        return list(csv.DictReader(f))


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.strftime('%F %T')} {' '.join(map(str, cmd))}\n")
        log.flush()
        return subprocess.run(list(map(str, cmd)), stdout=log, stderr=log).returncode


def demo_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with h5py.File(path, "r") as f:
            return len(f["data"].keys())
    except OSError:
        return 0


# --------------------------------------------------------------------------
# human/ rows
# --------------------------------------------------------------------------


def build_source_raw(task: str, n: int, work_dir: Path) -> Path:
    """Convert v1.0 human lerobot episodes to a raw robomimic HDF5."""
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "mg_ur5e_gen", ROOT / "generate_mimicgen_ur5e_dataset.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.patch_python_paths()

    args = SimpleNamespace(
        task=task,
        split="pretrain",
        source_root=V1_SOURCE_ROOT,
        source_lerobot_dir=None,
        work_dir=work_dir,
        source_n=n,
        force_source_raw=False,
    )
    source_dir = gen.latest_lerobot_dir(args.source_root, args.split, task)
    return gen.build_source_raw_hdf5(args, source_dir)


def build_human_rows(rows: list[dict], args) -> dict[str, str]:
    results = {}
    human_rows = [r for r in rows if r["relative_path"].startswith("human/")]
    canonical_by_task: dict[str, Path] = {}

    for row in human_rows:
        rel = Path(row["relative_path"])
        robot_dir, task_dir = rel.parts[1], rel.parts[2]
        out = args.raw_root / rel
        source_task = TASK_TO_SOURCE[row["task"]]
        if source_task is None:
            results[str(rel)] = "unbuildable (Flip Mug Upright has no source demos)"
            continue
        n = int(row["demonstrations"])
        if args.demos_cap is not None:
            n = min(n, args.demos_cap)

        if robot_dir == HUMAN_CANONICAL_DIR:
            if out.is_file() and demo_count(out) >= n:
                results[str(rel)] = f"exists ({demo_count(out)} demos)"
                canonical_by_task[task_dir] = out
                continue
            if args.dry_run:
                results[str(rel)] = f"would convert {n} v1.0 human demos"
                continue
            raw = build_source_raw(source_task, n, args.runs_root / "human_work")
            obj_names = ["obj"] if row["task"] in PNP_TASKS else []
            cmd = [
                PYTHON,
                RENDERER,
                "--inputs",
                raw,
                "--output",
                out,
                "--max-demos",
                n,
                "--obj-names",
                *obj_names,
            ]
            code = run(cmd, args.runs_root / "logs" / f"human_{task_dir}.log")
            got = demo_count(out)
            results[str(rel)] = (
                f"converted {got}/{n} demos" if code == 0 else f"FAILED (exit {code})"
            )
            if got:
                canonical_by_task[task_dir] = out

    # Symlink the non-canonical human embodiments to the canonical conversion.
    for row in human_rows:
        rel = Path(row["relative_path"])
        robot_dir, task_dir = rel.parts[1], rel.parts[2]
        if robot_dir == HUMAN_CANONICAL_DIR or TASK_TO_SOURCE[row["task"]] is None:
            continue
        target = canonical_by_task.get(task_dir)
        out = args.raw_root / rel
        if target is None:
            results[str(rel)] = "skipped (canonical human conversion missing)"
            continue
        if out.is_symlink() or out.is_file():
            results[str(rel)] = "exists (link)"
            continue
        if args.dry_run:
            results[str(rel)] = f"would symlink -> {target}"
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.symlink_to(os.path.relpath(target, out.parent))
        results[str(rel)] = f"symlink -> {os.path.relpath(target, out.parent)}"
    return results


# --------------------------------------------------------------------------
# mg/ rows
# --------------------------------------------------------------------------


def generation_runs(
    runs_root: Path, robot_dir: str, target_robot: str, task: str
) -> list[Path]:
    base = (
        runs_root
        / robot_dir
        / target_robot
        / "pretrain"
        / task
        / f"{task}_{target_robot}"
    )
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def build_mg_row(row: dict, args) -> str:
    rel = Path(row["relative_path"])
    out = args.raw_root / rel
    robot_dir = rel.parts[1]
    source_task = TASK_TO_SOURCE[row["task"]]
    if source_task is None:
        return "unbuildable (Flip Mug Upright has no source demos)"
    n_target = int(row["demonstrations"])
    if args.demos_cap is not None:
        n_target = min(n_target, args.demos_cap)
    if out.is_file() and demo_count(out) >= n_target:
        return f"exists ({demo_count(out)} demos)"
    if args.dry_run:
        return f"would generate {n_target} demos (seed {row['seed']})"

    target_robot, target_gripper = ROBOT_DIR_TO_TARGET[robot_dir]
    manifest_seed = int(row["seed"])
    runs_root = args.runs_root / "shards" / str(rel.parent)
    log = (
        args.runs_root
        / "logs"
        / f"mg_{robot_dir}_{source_task}_seed{manifest_seed}.log"
    )

    collected = sum(
        demo_count(p / "demo.hdf5")
        for p in generation_runs(runs_root, robot_dir, target_robot, source_task)
    )
    for pass_idx in range(args.max_passes):
        if collected >= n_target:
            break
        remaining = n_target - collected
        attempts = min(
            max(5, math.ceil(remaining * args.attempt_factor)),
            args.max_attempts_per_pass,
        )
        seed = manifest_seed if pass_idx == 0 else manifest_seed * 1000 + pass_idx
        cmd = [
            PYTHON,
            GENERATOR,
            "--task",
            source_task,
            "--target-robot",
            target_robot,
            "--target-gripper",
            target_gripper,
            "--seed",
            seed,
            "--num-demos",
            attempts,
            "--attempts-only",
            "--output-root",
            runs_root / robot_dir,
            "--fallback-video-n",
            2,
        ]
        code = run(cmd, log)
        if code != 0:
            return f"FAILED generation pass {pass_idx} (exit {code}, see {log})"
        collected = sum(
            demo_count(p / "demo.hdf5")
            for p in generation_runs(runs_root, robot_dir, target_robot, source_task)
        )

    inputs = [
        p / "demo.hdf5"
        for p in generation_runs(runs_root, robot_dir, target_robot, source_task)
        if demo_count(p / "demo.hdf5") > 0
    ]
    if not inputs:
        return f"NO SUCCESSES after {args.max_passes} passes (see {log})"
    obj_names = ["obj"] if row["task"] in PNP_TASKS else []
    cmd = [
        PYTHON,
        RENDERER,
        "--inputs",
        *inputs,
        "--output",
        out,
        "--max-demos",
        n_target,
        "--obj-names",
        *obj_names,
    ]
    code = run(cmd, log)
    got = demo_count(out)
    if code != 0 or got == 0:
        return f"FAILED render (exit {code}, see {log})"
    tag = "complete" if got >= n_target else "partial"
    return f"{tag}: {got}/{n_target} demos"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    rows = read_manifest()

    if args.tasks:
        rows = [r for r in rows if r["task"] in args.tasks]
    if args.robots:
        rows = [r for r in rows if Path(r["relative_path"]).parts[1] in args.robots]
    if args.seeds is not None:
        # human rows have no seed; keep them regardless of the seed filter
        rows = [
            r for r in rows if not r["seed"].strip() or int(r["seed"]) in args.seeds
        ]

    report: dict[str, str] = {}

    if not args.skip_human and args.worker_index == 0:
        report.update(build_human_rows(rows, args))

    if not args.skip_mg:
        mg_rows = [r for r in rows if r["relative_path"].startswith("mg/")]
        mg_rows = mg_rows[args.worker_index :: args.num_workers]
        for i, row in enumerate(mg_rows):
            status = build_mg_row(row, args)
            report[row["relative_path"]] = status
            print(
                f"[{i + 1}/{len(mg_rows)}] {row['relative_path']}: {status}", flush=True
            )

    report_path = args.runs_root / f"build_report_worker{args.worker_index}.json"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n=== worker {args.worker_index}: {len(report)} rows processed ===")
    for rel, status in report.items():
        print(f"  {rel}: {status}")


if __name__ == "__main__":
    main()
