#!/usr/bin/env bash
# RoboCasa port of shortcut-learning-in-grps's
# dataset_gen/single_gpu/visual_robust_cross_embodiment.sh
#
# Replays trajectories from a v1.0 atomic LeRobot dataset across several
# embodiments (same raw action sequence, eef-aligned start, all objects
# flown out of frame) so every embodiment's grip-site tracks the same path.
#
# Usage:
#   bash visual_robust_cross_embodiment.sh
#   SOURCE_LEROBOT_DIR=/data1/local/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/lerobot \
#     EPISODES="0 1 2" bash visual_robust_cross_embodiment.sh
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-robocasa}"
PYTHON_BIN="/home/fick17/anaconda3/envs/${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import robosuite" >/dev/null 2>&1; then
    echo "robosuite is not importable from $PYTHON_BIN" >&2
    exit 1
fi

ROOT="/data1/local/robocasa"
SOURCE_LEROBOT_DIR="${SOURCE_LEROBOT_DIR:-${ROOT}/datasets/v1.0/pretrain/atomic/PickPlaceSinkToCounter/20250819/lerobot}"
EPISODES="${EPISODES:-0}"
ROBOTS="${ROBOTS:-PandaOmron UR5eOmron IIWAOmron}"
HIDE_SCOPE="${HIDE_SCOPE:-robot_only}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/datasets/visual_robust_cross_embodiment/$(basename "$(dirname "$(dirname "$SOURCE_LEROBOT_DIR")")")}"
CAMERA_NAMES="${CAMERA_NAMES:-robot0_agentview_left robot0_agentview_right robot0_eye_in_hand}"
SETTLE_STEPS="${SETTLE_STEPS:-10}"
OSC_KP="${OSC_KP:-300}"
# Position+orientation IK matching is now the default (validated: <0.05deg
# error, needed for the gripper-twist correction to mean anything). Set
# MATCH_ORIENTATION=0 to fall back to position-only.
MATCH_ORIENTATION="${MATCH_ORIENTATION:-1}"
NUM_EXAMPLE_EPISODES="${NUM_EXAMPLE_EPISODES:-3}"
WRITE_LEROBOT="${WRITE_LEROBOT:-1}"

MATCH_ORI_FLAG=()
if [ "$MATCH_ORIENTATION" != "1" ]; then
    MATCH_ORI_FLAG=(--no-match-orientation)
fi
LEROBOT_FLAG=()
if [ "$WRITE_LEROBOT" != "1" ]; then
    LEROBOT_FLAG=(--no-lerobot)
fi

echo "=== source: ${SOURCE_LEROBOT_DIR} ==="
echo "=== episodes: ${EPISODES} ==="
echo "=== robots: ${ROBOTS} ==="
echo "=== hide scope: ${HIDE_SCOPE} ==="
echo "=== output: ${OUTPUT_DIR} ==="

MUJOCO_GL=egl "$PYTHON_BIN" "${ROOT}/generate_visual_robust_cross_embodiment.py" \
    --source-lerobot-dir "$SOURCE_LEROBOT_DIR" \
    --episodes $EPISODES \
    --robots $ROBOTS \
    --hide-scope "$HIDE_SCOPE" \
    --output-dir "$OUTPUT_DIR" \
    --camera-names $CAMERA_NAMES \
    --settle-steps "$SETTLE_STEPS" \
    --osc-kp "$OSC_KP" \
    --num-example-episodes "$NUM_EXAMPLE_EPISODES" \
    "${MATCH_ORI_FLAG[@]}" "${LEROBOT_FLAG[@]}"
