#!/usr/bin/env python3
"""Side-by-side PNG: source PandaOmron vs regenerated robot, 3 cams x 5 timesteps."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

CAMS = ["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]


def read_video(p):
    with av.open(str(p)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True, help="work-dir/<robot>/<task>/ep_XXXXXX")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    info = json.load(open(a.bundle / "regen_info.json"))
    src = Path(info["source_dataset"])
    e = info["source_episode"]
    rows = []
    vids_t = {c: read_video(a.bundle / f"{c}.mp4") for c in CAMS}
    vids_s = {c: read_video(next(src.glob(f"videos/chunk-*/observation.images.{c}/episode_{e:06d}.mp4"))) for c in CAMS}
    T = min(len(vids_t[CAMS[0]]), len(vids_s[CAMS[0]]))
    for t in np.linspace(0, T - 1, 5).astype(int):
        rows.append(np.concatenate([vids_s[c][t] for c in CAMS] + [np.full((256, 8, 3), 255, np.uint8)]
                                   + [vids_t[c][t] for c in CAMS], axis=1))
    img = Image.fromarray(np.concatenate(rows, axis=0))
    d = ImageDraw.Draw(img)
    d.text((5, 5), f"source PandaOmron ep{e} | {info['target_robot']} ({info['task']}) "
                   f"base_backoff={info['base_backoff_m']:.2f} finger_backoff={info['finger_backoff_m']:.3f}",
           fill=(255, 0, 0))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
