#!/usr/bin/env python3
"""
Merge the successful demos (demo.hdf5) from one or more MimicGen generation
runs (possibly different seeds/passes of the same task+embodiment) into a
single robomimic-style raw HDF5 with sequential demo_0..demo_{N-1} indices,
ready to feed into robocasa's official convert_hdf5_lerobot.py.

Usage:
    python merge_mimicgen_success_shards.py --output merged.hdf5 \
        --inputs run1/demo.hdf5 run2/demo.hdf5 [--max-demos 1000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-demos", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    demo_index = 0
    with h5py.File(args.output, "w") as out:
        data_grp = out.create_group("data")
        env_args_written = False
        for input_path in args.inputs:
            with h5py.File(input_path, "r") as source:
                if not env_args_written:
                    for key, value in source["data"].attrs.items():
                        if key == "total":
                            continue
                        data_grp.attrs[key] = value
                    env_args_written = True
                for demo_key in sorted(
                    source["data"].keys(), key=lambda k: int(k.split("_")[-1])
                ):
                    if args.max_demos is not None and demo_index >= args.max_demos:
                        break
                    source.copy(f"data/{demo_key}", data_grp, name=f"demo_{demo_index}")
                    demo_index += 1
        data_grp.attrs["total"] = demo_index
    print(
        f"wrote {args.output} ({demo_index} demos merged from {len(args.inputs)} files)"
    )


if __name__ == "__main__":
    main()
