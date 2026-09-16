#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_robocasa_x_eval.sh [ROBOT] [GRIPPER] [TASK]
#
# Examples:
#   ./run_robocasa_x_eval.sh UR5eOmron dataset PnPCounterToSink
#   ./run_robocasa_x_eval.sh IIWAOmron dataset TurnOnSinkFaucet
#
# Environment overrides:
#   CHECKPOINT, DATASET, OUTPUT_DIR, NUM_EPISODES, NUM_VIDEO_EPISODES,
#   VIDEO_CAMERAS, EXTRA_STEPS, DEVICE, TASK_TEXT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT="${ROBOCASA_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)}"
BARX_ROOT="${BARX_ROOT:-${ROOT}/third_party/barx}"
ROBOSUITE_SRC="${ROBOSUITE_SRC:-${ROOT}/third_party/robosuite}"
# LeRobot is not vendored in this repo -- point LEROBOT_SRC / PYTHON at
# whatever LeRobot checkout/venv actually trained the checkpoint you're
# evaluating. See CLAUDE.md section "Eval" for what this needs to contain.
: "${LEROBOT_SRC:?set LEROBOT_SRC to a LeRobot src/ checkout}"
: "${PYTHON:?set PYTHON to that LeRobot checkouts interpreter, e.g. its venv bin/python}"

ROBOT="${1:-${ROBOT:-UR5eOmron}}"
GRIPPER="${2:-${GRIPPER:-Robotiq85Gripper}}"
TASK="${3:-${TASK:-PnPSinkToCounter}}"

cd "${BARX_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROBOSUITE_SRC}:${BARX_ROOT}/robocasa_x:${LEROBOT_SRC}:${ROOT}/.eval_extra_site:${PYTHONPATH:-}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a pretrained_model dir}"
DATASET="${DATASET:-${BARX_ROOT}/datasets/atomic_new/${ROBOT}/pretrain/${TASK}/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/eval/eval_plus100_barx_baseline_pretrained_p900_i1000_u1000_eval_all_${ROBOT}_${GRIPPER}_${TASK}}"
NUM_EPISODES="${NUM_EPISODES:-100}"
NUM_VIDEO_EPISODES="${NUM_VIDEO_EPISODES:-20}"
VIDEO_SKIP="${VIDEO_SKIP:-2}"
DEVICE="${DEVICE:-cuda}"
# Left unset by default: eval_lerobot_checkpoint.py now defaults
# --video-cameras (and --video-height/--video-width) to exactly what the
# checkpoint's own input_features use, so the saved video is a literal
# recording of what the policy was fed -- same camera(s), same resolution/
# aspect ratio. Rendering at a different aspect ratio than the policy's own
# (e.g. a square buffer for a wide 180x320 input) changes how much of the
# scene is visible, since MuJoCo's fovy is a *vertical* FOV -- the old
# hardcoded "robot0_agentview_right robot0_agentview_left" both showed a
# camera the checkpoint never had access to AND cropped the one it did.
# Set VIDEO_CAMERAS explicitly to override (e.g. for --random-actions mode,
# which has no checkpoint to derive a default from).
VIDEO_CAMERAS="${VIDEO_CAMERAS:-}"
EXTRA_STEPS="${EXTRA_STEPS:-100}"
# robosuite ships two different Robotiq 85 models whose grip_site frames differ
# by 90deg, so --gripper alone does not pin down which one. Point this at the
# dataset whose gripper you want and it is copied verbatim.
GRIPPER_DONOR_DATASET="${GRIPPER_DONOR_DATASET:-}"

# The only Robotiq 85 that exists in our data is the one recorded in the
# UR5eOmron datasets (robotiq_gripper_85.xml, root body
# `robotiq_85_adapter_link`). robosuite's Robotiq85Gripper *class* resolves to a
# different model (2f85_robosuite.xml, root `base_mount`, 13 bodies vs 8), so
# passing --gripper alone silently substitutes a gripper no dataset was ever
# recorded with. Pin it to the UR5e recording unless told otherwise.
# Robotiq140Gripper needs no such treatment: its class already resolves to the
# same model the IIWAOmron datasets recorded.
if [ -z "${GRIPPER_DONOR_DATASET}" ] && [ "${GRIPPER}" = "Robotiq85Gripper" ]; then
  CANDIDATE="${BARX_ROOT}/datasets/atomic_new/UR5eOmron/pretrain/${TASK}/lerobot"
  if [ -d "${CANDIDATE}" ]; then
    GRIPPER_DONOR_DATASET="${CANDIDATE}"
    echo "[eval] Robotiq85Gripper pinned to the UR5e recording: ${CANDIDATE}"
  else
    echo "[warn] no UR5eOmron recording for ${TASK}; Robotiq85Gripper will fall back" >&2
    echo "[warn] to robosuite's 2f85_robosuite model, which differs from the datasets" >&2
  fi
fi

if [ ! -d "${DATASET}" ]; then
  echo "[error] dataset not found: ${DATASET}" >&2
  exit 1
fi

TASK_TEXT="${TASK_TEXT:-$("${PYTHON}" - "${DATASET}" <<'PY'
import json
import sys
from pathlib import Path

tasks = Path(sys.argv[1]) / "meta" / "tasks.jsonl"
with tasks.open("r") as f:
    line = f.readline()
print(json.loads(line)["task"])
PY
)}"

mkdir -p "${OUTPUT_DIR}/logs"
rm -f "${OUTPUT_DIR}/results.json"
rm -f "${OUTPUT_DIR}"/videos/*.mp4 2>/dev/null || true

echo "[eval] robot=${ROBOT} gripper=${GRIPPER} task=${TASK}"
echo "[eval] dataset=${DATASET}"
echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] output=${OUTPUT_DIR}"
echo "[eval] task_text=${TASK_TEXT}"

exec "${PYTHON}" "${SCRIPT_DIR}/eval_lerobot_checkpoint.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --lerobot-src "${LEROBOT_SRC}" \
  --robosuite-src "${ROBOSUITE_SRC}" \
  --robocasa-src "${BARX_ROOT}/robocasa_x" \
  --robocasa-x-compat \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --task "${TASK_TEXT}" \
  --robot "${ROBOT}" \
  --gripper "${GRIPPER}" \
  ${GRIPPER_DONOR_DATASET:+--gripper-donor-dataset "${GRIPPER_DONOR_DATASET}"} \
  --num-episodes "${NUM_EPISODES}" \
  --extra-steps "${EXTRA_STEPS}" \
  --save-videos \
  --num-video-episodes "${NUM_VIDEO_EPISODES}" \
  ${VIDEO_CAMERAS:+--video-cameras ${VIDEO_CAMERAS}} \
  --video-skip "${VIDEO_SKIP}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S).log"
