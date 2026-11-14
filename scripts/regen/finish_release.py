#!/usr/bin/env python3
"""After run_full.py exits: tree-level MANIFEST_<tree>.md5 for every release tree and a
release/RELEASE_DONE.json completion marker (per-tree dataset count, validation status).

  finish_release.py --release <out-root> [--wait-pid PID]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def md5(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--wait-pid", type=int, default=None)
    p.add_argument("--trees", nargs="+", default=["final_v3", "source_combined_shift_v2",
                                                   "source_wrist_shift_v2", "source_front_shift_v2"])
    a = p.parse_args()
    while a.wait_pid and Path(f"/proc/{a.wait_pid}").exists():
        time.sleep(60)
    summary = {}
    for tree in a.trees:
        root = a.release / tree
        man = a.release / f"MANIFEST_{tree}.md5"
        files = sorted(f for f in root.rglob("*") if f.is_file())
        with open(man.with_suffix(".tmp"), "w") as fh:
            for f in files:
                fh.write(f"{md5(f)}  {f.relative_to(a.release)}\n")
        os.replace(man.with_suffix(".tmp"), man)
        ds = sorted(root.glob("*/*/lerobot"))
        reports = [d / "validation_report.json" for d in ds]
        bad = [str(d) for d, r in zip(ds, reports) if r.exists() and not json.load(open(r)).get("ok")]
        summary[tree] = dict(datasets=len(ds), files=len(files), bytes=sum(f.stat().st_size for f in files),
                             validation_failed=bad,
                             unfinished=[str(d) for d in ds if (d / ".render").exists() or (d / ".locks").exists()])
        print(tree, summary[tree]["datasets"], summary[tree]["bytes"], flush=True)
    json.dump(dict(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trees=summary),
              open(a.release / "RELEASE_DONE.json", "w"), indent=1)


if __name__ == "__main__":
    main()
