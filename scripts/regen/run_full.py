#!/usr/bin/env python3
"""End-to-end orchestrator: canonical regen -> merge -> camera-shift re-renders.

One process pool over a dependency graph, per (robot, task):
  regen chunks (regen_replay.py) -> merge (merge_all.py) -> per variant:
  render chunks (render_shift.py render) -> finalize (render_shift.py finalize).

Launch priority: finalize > merge > --first-variants renders > canonical regen
(task-major, all robots of a task before the next task) > --later-variants
renders, which only start once every first variant of every (robot, task) is
finalized. Stage completion is recorded as marker files under
<work-dir>/_state, so re-running resumes. Episodes already regenerated in
--reuse-dir (e.g. the validation run, same code) are copied instead of re-run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scheduler import task_episodes  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--robots", nargs="+", default=["UR5eOmron", "IIWAOmron", "Kinova3Omron"])
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True, help="<out-root>/{final_v3,source_<variant>}/<Robot>/<Task>/lerobot")
    p.add_argument("--reuse-dir", type=Path, default=None)
    p.add_argument("--first-variants", nargs="*", default=["combined_shift_v2"])
    p.add_argument("--later-variants", nargs="*", default=["wrist_shift_v2", "front_shift_v2"])
    p.add_argument("--procs", type=int, default=12, help="total concurrent sim/render processes (keep <= 12)")
    p.add_argument("--max-load", type=float, default=30.0, help="hold launches while 1-min loadavg exceeds this")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--regen-chunk", type=int, default=2)
    p.add_argument("--render-chunk", type=int, default=10)
    p.add_argument("--min-free-gb", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=2)
    return p.parse_args()


class Orchestrator:
    def __init__(self, a):
        self.a = a
        self.state = a.work_dir / "_state"
        self.state.mkdir(parents=True, exist_ok=True)
        self.logs = a.work_dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.running = {}  # job key -> (proc, fh, t0)
        self.fails = {}
        self.eps = {t: task_episodes(t) for t in a.tasks}
        self.pairs = [(r, t) for t, r in itertools.product(a.tasks, a.robots)]  # task-major
        self.gpu = itertools.cycle(a.gpus)
        self.audit()

    def audit(self):
        """Startup only (no workers alive): drop stale locks, and treat the result.json /
        ep folder pair as the single source of truth -- an episode whose pair disagrees
        (or that errored) is deleted and re-run, and every merged/rendered output of its
        (robot, task) is invalidated, since re-merging renumbers episodes."""
        busy = subprocess.run(["pgrep", "-f", "regen_replay.py|render_shift.py"], capture_output=True, text=True).stdout
        assert not busy.strip(), f"workers still alive, kill them first: {busy.split()}"
        for lk in itertools.chain(self.a.work_dir.glob("*/*/.locks"), self.a.out_root.glob("*/*/*/lerobot/.locks")):
            shutil.rmtree(lk)
        for r, t in self.pairs:
            bad = []
            for res in (self.a.work_dir / r / t).glob("ep_*.result.json"):
                ep_dir = res.with_name(res.name.replace(".result.json", ""))
                d = json.load(open(res))
                if "error" in d or bool(d.get("success")) != ep_dir.is_dir():
                    bad.append(ep_dir.name)
                    res.unlink()
                    shutil.rmtree(ep_dir, ignore_errors=True)
            for tmp in (self.a.work_dir / r / t).glob("ep_*.tmp"):
                shutil.rmtree(tmp)
            if bad:
                print(f"[audit] {r}/{t}: re-run {sorted(bad)}; invalidating merge + shifted outputs", flush=True)
                self.mark("merge", r, t).unlink(missing_ok=True)
                for v in self.a.first_variants + self.a.later_variants:
                    self.mark("final", v, r, t).unlink(missing_ok=True)
                    shutil.rmtree(self.shifted(v, r, t), ignore_errors=True)

    # ---- paths / markers
    def canon(self, r, t):
        return self.a.out_root / "final_v3" / r / t / "lerobot"

    def shifted(self, v, r, t):
        return self.a.out_root / f"source_{v}" / r / t / "lerobot"

    def mark(self, *k):
        return self.state / ("__".join(map(str, k)) + ".done")

    def done(self, *k):
        return self.mark(*k).exists()

    def regen_done(self, r, t, e):
        return (self.a.work_dir / r / t / f"ep_{e:06d}.result.json").exists()

    def n_canon(self, r, t):
        return json.load(open(self.canon(r, t) / "meta" / "info.json"))["total_episodes"]

    def rendered(self, v, r, t, e):
        return (self.shifted(v, r, t) / ".render" / f"episode_{e:06d}.json").exists()

    # ---- job generation (key, cmd, marker-or-None)
    def reuse(self, r, t, e):
        src = self.a.reuse_dir and self.a.reuse_dir / r / t
        if not src or not (src / f"ep_{e:06d}.result.json").exists():
            return False
        dst = self.a.work_dir / r / t
        dst.mkdir(parents=True, exist_ok=True)
        if (src / f"ep_{e:06d}").is_dir():
            shutil.copytree(src / f"ep_{e:06d}", dst / f"ep_{e:06d}", dirs_exist_ok=True)
        shutil.copy2(src / f"ep_{e:06d}.result.json", dst / f"ep_{e:06d}.result.json")
        return True

    def variant_jobs(self, v, r, t):
        if self.done("final", v, r, t):
            return []
        n = self.n_canon(r, t)
        todo = [e for e in range(n) if not self.rendered(v, r, t, e)]
        if not todo and self.busy("render", v, r, t):
            return [None]  # last chunks still rendering
        todo = [e for e in todo if e not in self.busy("render", v, r, t)]
        base = [sys.executable, str(HERE / "render_shift.py")]
        args = ["--canonical", str(self.canon(r, t)), "--out", str(self.shifted(v, r, t)), "--robot", r, "--variant", v]
        if not todo:
            return [(("final", v, r, t), base + ["finalize", *args], self.mark("final", v, r, t))]
        return [(("render", v, r, t, c[0]), base + ["render", *args, "--episodes", *map(str, c)], None)
                for c in (todo[i:i + self.a.render_chunk] for i in range(0, len(todo), self.a.render_chunk))]

    def busy(self, kind, *k):
        """Episodes of in-flight jobs, so a re-chunked todo list never relaunches them."""
        return {e for key, (*_, eps) in self.running.items() if key[0] == kind and key[1:-1] == k for e in eps}

    def ready_jobs(self):
        finals, merges, first, regen, later = [], [], [], [], []
        all_first_done = True
        for r, t in self.pairs:
            if not self.done("merge", r, t):
                all_first_done = False
                todo = [e for e in self.eps[t] if not self.regen_done(r, t, e) and not self.reuse(r, t, e)]
                if not todo and not self.busy("regen", r, t):
                    merges.append((("merge", r, t), [sys.executable, str(HERE / "merge_all.py"), "--work-dir",
                                   str(self.a.work_dir), "--out-root", str(self.a.out_root / "final_v3"),
                                   "--robots", r, "--tasks", t], self.mark("merge", r, t)))
                todo = [e for e in todo if e not in self.busy("regen", r, t)]
                for i in range(0, len(todo), self.a.regen_chunk):
                    c = todo[i:i + self.a.regen_chunk]
                    regen.append((("regen", r, t, c[0]), [sys.executable, str(HERE / "regen_replay.py"), "--task", t,
                                  "--target-robot", r, "--work-dir", str(self.a.work_dir),
                                  "--episodes", *map(str, c)], None))
                continue
            for v in self.a.first_variants:
                js = self.variant_jobs(v, r, t)
                all_first_done &= not js
                (finals if js and js[0] and js[0][0][0] == "final" else first).extend(js)
        if all_first_done:
            for r, t in self.pairs:
                for v in self.a.later_variants:
                    js = self.variant_jobs(v, r, t)
                    (finals if js and js[0] and js[0][0][0] == "final" else later).extend(js)
        return [j for j in finals + merges + first + regen + later if j is not None]

    # ---- loop
    def launch(self, key, cmd, marker):
        gpu = next(self.gpu)
        env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                   MUJOCO_EGL_DEVICE_ID=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        fh = open(self.logs / ("_".join(map(str, key)) + ".log"), "a")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
        eps = [int(x) for x in cmd[cmd.index("--episodes") + 1:]] if "--episodes" in cmd else []
        self.running[key] = (proc, fh, time.time(), marker, eps)

    def reap(self):
        for key, (proc, fh, t0, marker, _) in list(self.running.items()):
            if proc.poll() is None:
                continue
            fh.close()
            del self.running[key]
            ok = proc.returncode == 0
            if ok and marker is not None:
                marker.write_text(json.dumps(dict(seconds=time.time() - t0, finished=time.time())))
            if not ok:
                self.fails[key] = self.fails.get(key, 0) + 1
            print(f"[full] {'ok ' if ok else 'ERR'} {key} {time.time() - t0:.0f}s", flush=True)

    def run(self):
        t_start = time.time()
        while True:
            self.reap()
            jobs = [j for j in self.ready_jobs()
                    if j[0] not in self.running and self.fails.get(j[0], 0) <= self.a.max_retries]
            if not jobs and not self.running:
                break
            for key, cmd, marker in jobs:
                if len(self.running) >= self.a.procs:
                    break
                if psutil.virtual_memory().available / 1e9 < self.a.min_free_gb or os.getloadavg()[0] > self.a.max_load:
                    break
                self.launch(key, cmd, marker)
                time.sleep(2)  # stagger env construction
            time.sleep(10)
        print(f"[full] all done in {time.time() - t_start:.0f}s; failed keys: "
              f"{[k for k, n in self.fails.items() if n > self.a.max_retries]}", flush=True)


if __name__ == "__main__":
    Orchestrator(parse_args()).run()
