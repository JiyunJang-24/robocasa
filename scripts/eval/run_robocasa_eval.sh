#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT="${ROBOCASA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)}"
ROBOSUITE_SRC="${ROBOSUITE_SRC:-${ROOT}/third_party/robosuite}"
# LeRobot is not vendored here -- see CLAUDE.md "Eval" for what this needs.
: "${LEROBOT_SRC:?set LEROBOT_SRC to a LeRobot src/ checkout}"
: "${PYTHON:?set PYTHON to that LeRobot checkouts interpreter}"
cd "${ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROBOSUITE_SRC}:${ROOT}:${LEROBOT_SRC}:${ROOT}/.eval_extra_site:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a pretrained_model dir}"
DATASET="${DATASET:?set DATASET to a lerobot dataset dir, e.g. datasets/v1.0/target/atomic/<Task>/<date>/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/eval/$(basename "${CHECKPOINT%/pretrained_model}")}"
NUM_EPISODES="${NUM_EPISODES:-506}"
NUM_VIDEO_EPISODES="${NUM_VIDEO_EPISODES:-20}"
VIDEO_SKIP="${VIDEO_SKIP:-2}"
DEVICE="${DEVICE:-cuda}"
VIDEO_CAMERAS="${VIDEO_CAMERAS:-robot0_agentview_right robot0_eye_in_hand}"
EXTRA_STEPS="${EXTRA_STEPS:-50}"

mkdir -p "${OUTPUT_DIR}/logs"
rm -f "${OUTPUT_DIR}/results.json"
rm -f "${OUTPUT_DIR}"/videos/*.mp4 2>/dev/null || true

exec "${PYTHON}" "${SCRIPT_DIR}/eval_lerobot_checkpoint.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --robosuite-src "${ROBOSUITE_SRC}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --num-episodes "${NUM_EPISODES}" \
  --extra-steps "${EXTRA_STEPS}" \
  --save-videos \
  --num-video-episodes "${NUM_VIDEO_EPISODES}" \
  --video-cameras ${VIDEO_CAMERAS} \
  --video-skip "${VIDEO_SKIP}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S).log"
