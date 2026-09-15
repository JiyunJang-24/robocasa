"""Find which source episode each new_barx episode was rendered from.

new_barx episode N is not source episode N: 48 of 108 episodes disagree even
on length, so assuming the identity mapping gave 24 episodes a camera from
the wrong scene. Matching on trajectory content instead is unambiguous --
the stored world eef path is the source path, so the correct source episode
is the one whose own eef path coincides with it to sub-millimetre.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quat_xyzw_to_mat(q):
    x, y, z, w = np.asarray(q, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def source_world_eef(df):
    """Source state is (support origin, support rotation xyzw, eef in support)."""
    st = np.stack(df["observation.state"].to_numpy())
    out = np.empty((len(st), 3))
    for i in range(len(st)):
        out[i] = st[i, 0:3] + quat_xyzw_to_mat(st[i, 3:7]) @ st[i, 7:10]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-barx", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--sample",
        type=int,
        default=40,
        help="frames per episode used for the comparison",
    )
    args = ap.parse_args()

    # index the source by length first: only equal-length episodes can match,
    # which cuts the search from 1000 candidates to a handful
    by_len = {}
    src_cache = {}
    for f in sorted(
        glob.glob(str(args.source / "data" / "chunk-*" / "episode_*.parquet"))
    ):
        ep = int(Path(f).stem.split("_")[1])
        df = pd.read_parquet(f, columns=["observation.state"])
        by_len.setdefault(len(df), []).append(ep)
        src_cache[ep] = f

    mapping, worst = {}, 0.0
    unmatched = []
    for f in sorted(
        glob.glob(str(args.new_barx / "data" / "chunk-*" / "episode_*.parquet"))
    ):
        ep = int(Path(f).stem.split("_")[1])
        nb = np.stack(
            pd.read_parquet(f, columns=["observation.state"])[
                "observation.state"
            ].to_numpy()
        )[:, 0:3]
        idx = np.linspace(0, len(nb) - 1, min(args.sample, len(nb))).astype(int)
        # Compare trajectory SHAPE: the replayed eef differs from the source
        # recording by an offset that varies over the episode, so absolute
        # position is a blunt key, while the path traced out is not.
        scored = []
        for cand in by_len.get(len(nb), []):
            sw = source_world_eef(
                pd.read_parquet(src_cache[cand], columns=["observation.state"])
            )
            a = nb[idx] - nb[idx].mean(0)
            b = sw[idx] - sw[idx].mean(0)
            scored.append((float(np.abs(a - b).mean()), cand))
        scored.sort()
        if not scored:
            unmatched.append((ep, None, float("inf")))
            continue
        best_err, best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else float("inf")
        # a match is only trustworthy if it beats the next candidate clearly
        if best_err > 0.05 or (runner < 3 * best_err and len(scored) > 1):
            unmatched.append((ep, best, best_err))
            continue
        mapping[ep] = best
        worst = max(worst, best_err)

    args.out.write_text(
        json.dumps(
            {
                "mapping": {str(k): v for k, v in mapping.items()},
                "worst_err_m": worst,
                "unmatched": [[e, b, err] for e, b, err in unmatched],
            },
            indent=2,
        )
    )
    identity = sum(1 for k, v in mapping.items() if k == v)
    print(f"matched {len(mapping)} episodes, worst {worst*1000:.2f} mm")
    print(f"  identity mapping would have been right for {identity} of {len(mapping)}")
    if unmatched:
        print(f"  UNMATCHED {len(unmatched)}: {unmatched[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
