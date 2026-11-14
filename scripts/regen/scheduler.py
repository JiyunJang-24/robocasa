#!/usr/bin/env python3
"""Chunked, resumable scheduler for regen_replay.py.

Splits (robot, task, episodes) into chunks, runs at most --procs workers at once,
spreads EGL rendering round-robin across --gpus (MuJoCo physics is CPU-only), and
holds back new launches while available RAM < --min-free-gb. A chunk whose
episodes all have *.result.json is skipped, so re-running resumes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regen_common as RC  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--robots", nargs="+", default=["UR5eOmron"])
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--max-episodes", type=int, default=None, help="first N episodes per task (pilot)")
    p.add_argument("--min-free-gb", type=float, default=60.0)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="args passed to regen_replay.py")
    return p.parse_args()


def task_episodes(task):
    src = RC.resolve_source_dir(task)
    return sorted(int(p.stem.split("_")[-1]) for p in src.glob("data/chunk-*/episode_*.parquet"))


def main():
    a = parse_args()
    log_dir = a.work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for robot, task in itertools.product(a.robots, a.tasks):
        eps = task_episodes(task)[: a.max_episodes]
        tdir = a.work_dir / robot / task
        todo = [e for e in eps if not (tdir / f"ep_{e:06d}.result.json").exists()]
        for i in range(0, len(todo), a.chunk):
            jobs.append((robot, task, todo[i : i + a.chunk]))
    print(f"[sched] {len(jobs)} chunks, procs={a.procs}, gpus={a.gpus}", flush=True)

    running = {}
    gpu_cycle = itertools.cycle(a.gpus)
    t0 = time.time()
    done = 0
    while jobs or running:
        for pid, (proc, job, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                done += 1
                del running[pid]
                print(f"[sched] done {job[0]}/{job[1]} {job[2][0]}..{job[2][-1]} rc={proc.returncode} "
                      f"({done} finished, {len(jobs)} queued, {time.time()-t0:.0f}s)", flush=True)
        while jobs and len(running) < a.procs:
            free_gb = psutil.virtual_memory().available / 1e9
            if free_gb < a.min_free_gb:
                print(f"[sched] memory guard: {free_gb:.0f} GB free, holding", flush=True)
                break
            robot, task, eps = jobs.pop(0)
            gpu = next(gpu_cycle)
            env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                       MUJOCO_EGL_DEVICE_ID=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            fh = open(log_dir / f"{robot}_{task}_{eps[0]:06d}.log", "a")
            cmd = [sys.executable, str(HERE / "regen_replay.py"), "--task", task,
                   "--target-robot", robot, "--work-dir", str(a.work_dir),
                   "--episodes", *map(str, eps), *a.extra]
            proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[proc.pid] = (proc, (robot, task, eps), fh)
            time.sleep(2)  # stagger env construction
        time.sleep(5)

    summary = {}
    for robot, task in itertools.product(a.robots, a.tasks):
        rs = [json.load(open(p)) for p in (a.work_dir / robot / task).glob("ep_*.result.json")]
        summary[f"{robot}/{task}"] = dict(n=len(rs), success=sum(r["success"] for r in rs),
                                          errors=sum("error" in r for r in rs))
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(a.work_dir / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
