#!/usr/bin/env python3
"""
Build an Excel workbook comparing every RoboCasa v1.0 atomic task against the
four tasks BARX's RoboCasa-X benchmark provides, so it's clear which atomic
tasks BARX already has an equivalent for (and which do not).

BARX benchmark tasks (barx/benchmark.py TASKS) and their underlying RoboCasa
task classes (confirmed by reading BARX/robocasa_x/robocasa/environments/kitchen
/single_stage/kitchen_pnp.py and .../atomic/kitchen_sink.py):
  - pnp_counter_to_sink   -> PnPCounterToSink   == atomic PickPlaceCounterToSink
  - pnp_sink_to_counter   -> PnPSinkToCounter   == atomic PickPlaceSinkToCounter
  - turn_on_sink_faucet   -> ManipulateSinkFaucet(behavior="turn_on")
                                                 == atomic TurnOnSinkFaucet
  - flip_mug_upright      -> new BARX-only Kitchen subclass, NOT a RoboCasa
                                                 atomic task (no source demos)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

V1_ROOT = Path("/data1/local/robocasa/datasets/v1.0")
OUTPUT = Path("/data1/local/BARX/datasets/barx_atomic_task_comparison.xlsx")

BARX_TASK_FOR_ATOMIC = {
    "PickPlaceCounterToSink": "pnp_counter_to_sink (PnP Counter to Sink)",
    "PickPlaceSinkToCounter": "pnp_sink_to_counter (PnP Sink to Counter)",
    "TurnOnSinkFaucet": "turn_on_sink_faucet (Turn On Sink Faucet)",
}
BARX_ONLY_TASKS = [
    (
        "flip_mug_upright (Flip Mug Upright)",
        "BARX-only: new task, no RoboCasa atomic equivalent / no source demos",
    ),
]


def count_episodes(task: str, split: str) -> int:
    return len(
        list(
            (V1_ROOT / split / "atomic" / task).glob(
                "*/lerobot/data/chunk-*/episode_*.parquet"
            )
        )
    )


def main() -> None:
    atomic_tasks = sorted(
        p.name for p in (V1_ROOT / "pretrain" / "atomic").iterdir() if p.is_dir()
    )
    target_tasks = {
        p.name for p in (V1_ROOT / "target" / "atomic").iterdir() if p.is_dir()
    }

    rows = []
    for task in atomic_tasks:
        barx_match = BARX_TASK_FOR_ATOMIC.get(task, "")
        rows.append(
            {
                "RoboCasa v1.0 atomic task": task,
                "pretrain episodes": count_episodes(task, "pretrain"),
                "target episodes": count_episodes(task, "target")
                if task in target_tasks
                else 0,
                "In BARX benchmark?": "YES" if barx_match else "no",
                "BARX task": barx_match,
            }
        )
    for barx_name, note in BARX_ONLY_TASKS:
        rows.append(
            {
                "RoboCasa v1.0 atomic task": "(none)",
                "pretrain episodes": 0,
                "target episodes": 0,
                "In BARX benchmark?": "BARX-only",
                "BARX task": barx_name,
            }
        )

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="atomic_vs_barx", index=False)
        summary = pd.DataFrame(
            {
                "metric": [
                    "RoboCasa v1.0 atomic tasks (pretrain split)",
                    "Atomic tasks covered by a BARX benchmark task",
                    "Atomic tasks with NO BARX equivalent",
                    "BARX benchmark tasks total",
                    "BARX tasks with an atomic-task source (reusable actions)",
                    "BARX tasks with no atomic equivalent (Flip Mug Upright)",
                ],
                "value": [
                    len(atomic_tasks),
                    len(BARX_TASK_FOR_ATOMIC),
                    len(atomic_tasks) - len(BARX_TASK_FOR_ATOMIC),
                    len(BARX_TASK_FOR_ATOMIC) + len(BARX_ONLY_TASKS),
                    len(BARX_TASK_FOR_ATOMIC),
                    len(BARX_ONLY_TASKS),
                ],
            }
        )
        summary.to_excel(writer, sheet_name="summary", index=False)

    # Styling pass
    from openpyxl import load_workbook

    wb = load_workbook(OUTPUT)
    ws = wb["atomic_vs_barx"]
    header_fill = PatternFill("solid", fgColor="305496")
    match_fill = PatternFill("solid", fgColor="C6EFCE")
    barx_only_fill = PatternFill("solid", fgColor="FFE699")
    no_match_fill = PatternFill("solid", fgColor="F2F2F2")

    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    match_col = [c.value for c in ws[1]].index("In BARX benchmark?") + 1
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        status = row[match_col - 1].value
        fill = (
            match_fill
            if status == "YES"
            else (barx_only_fill if status == "BARX-only" else no_match_fill)
        )
        for cell in row:
            cell.fill = fill

    widths = [34, 17, 15, 18, 46]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    ws2 = wb["summary"]
    for cell in ws2[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
    ws2.column_dimensions["A"].width = 55
    ws2.column_dimensions["B"].width = 10

    wb.save(OUTPUT)
    print(f"wrote {OUTPUT}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
