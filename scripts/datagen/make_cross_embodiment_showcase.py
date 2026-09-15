#!/usr/bin/env python3
"""
Concatenate one labeled success rollout per embodiment into a single showcase
video per task: <root>/showcase/<task>_cross_embodiment.mp4
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path("/data1/local/BARX/datasets/cross_embodiment")
ROBOTS = ["PandaOmron", "UR5eOmron", "IIWAOmron", "Kinova3Omron"]
LABELS = {
    "PandaOmron": "Panda (baseline)",
    "UR5eOmron": "UR5e",
    "IIWAOmron": "IIWA",
    "Kinova3Omron": "Kinova3",
}
TASKS = ["TurnOnSinkFaucet", "PickPlaceCounterToSink", "AdjustToasterOvenTemperature"]
SIZE = 512
FPS = 12


def success_videos(robot: str, task: str) -> list[Path]:
    base = ROOT / robot / "pretrain" / task / f"{task}_{robot}"
    if not base.exists():
        return []
    videos = []
    for run in sorted(base.iterdir()):
        videos.extend(sorted((run / "videos").glob("success_demo_*.mp4")))
    return videos


def label_frame(frame: np.ndarray, text: str, subtext: str) -> np.ndarray:
    frame = cv2.resize(frame, (SIZE, SIZE))
    cv2.rectangle(frame, (0, 0), (SIZE, 54), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame, subtext, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )
    return frame


def main() -> None:
    out_dir = ROOT / "showcase"
    out_dir.mkdir(exist_ok=True)
    for task in TASKS:
        out_path = out_dir / f"{task}_cross_embodiment.mp4"
        writer = imageio.get_writer(out_path, fps=FPS)
        segments = 0
        for robot in ROBOTS:
            videos = success_videos(robot, task)
            if not videos:
                print(f"[skip] no success video for {robot}/{task}")
                continue
            reader = imageio.get_reader(videos[0])
            for frame in reader:
                writer.append_data(label_frame(frame, LABELS[robot], task))
            reader.close()
            segments += 1
        writer.close()
        print(f"wrote {out_path} ({segments} embodiments)")


if __name__ == "__main__":
    main()
