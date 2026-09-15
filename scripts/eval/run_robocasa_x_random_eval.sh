#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_robocasa_x_random_eval.sh [ROBOT] [GRIPPER] [TASK] [EPISODES] [MAX_STEPS]
#
# Examples:
#   ./run_robocasa_x_random_eval.sh IIWAOmron Robotiq140Gripper PnPCounterToSink 0 350
#   ./run_robocasa_x_random_eval.sh UR5eOmron Robotiq85Gripper PnPCounterToSink 0,3,7 500
#   ./run_robocasa_x_random_eval.sh IIWAOmron Robotiq140Gripper PnPCounterToSink first:10 350
#
# Environment overrides:
#   DATASET, OUTPUT_DIR, JOB_NAME, VIDEO_CAMERAS, VIDEO_SKIP,
#   NUM_VIDEO_EPISODES, DEVICE, TASK_TEXT, SEED, RANDOM_ACTION_SCALE

ROOT="/data1/local/robocasa"
BARX_ROOT="/data1/local/barx"

ROBOT="${1:-${ROBOT:-IIWAOmron}}"
GRIPPER="${2:-${GRIPPER:-Robotiq140Gripper}}"
TASK="${3:-${TASK:-PnPCounterToSink}}"
EPISODES_ARG="${4:-${EPISODES:-0}}"
MAX_STEPS="${5:-${MAX_STEPS:-350}}"

cd "${BARX_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data1/local/robosuite:${BARX_ROOT}/robocasa_x:/data1/local/lerobot-VAI/src:${ROOT}/.eval_extra_site:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/data1/local/lerobot-VAI/venv-lerobot-eval/bin/python}"
DATASET="${DATASET:-${BARX_ROOT}/datasets/atomic/${ROBOT}/pretrain/${TASK}/lerobot}"
DEVICE="${DEVICE:-cuda}"
VIDEO_CAMERAS="${VIDEO_CAMERAS:-robot0_agentview_right robot0_eye_in_hand}"
VIDEO_SKIP="${VIDEO_SKIP:-2}"
NUM_VIDEO_EPISODES="${NUM_VIDEO_EPISODES:-999999}"
SEED="${SEED:-0}"
RANDOM_ACTION_SCALE="${RANDOM_ACTION_SCALE:-1.0}"

if [ ! -d "${DATASET}" ]; then
  echo "[error] dataset not found: ${DATASET}" >&2
  exit 1
fi

TASK_TEXT="${TASK_TEXT:-$("${PYTHON}" - "${DATASET}" "${EPISODES_ARG}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
episodes_arg = sys.argv[2]
episode_index = 0
if episodes_arg and not episodes_arg.startswith("first:"):
    episode_index = int(episodes_arg.replace(",", " ").split()[0])

task_index = 0
parquet = dataset / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
try:
    import pandas as pd
    task_index = int(pd.read_parquet(parquet, columns=["task_index"])["task_index"].iloc[0])
except Exception:
    pass

tasks = {}
with (dataset / "meta" / "tasks.jsonl").open("r") as f:
    for line in f:
        item = json.loads(line)
        tasks[int(item["task_index"])] = item["task"]
print(tasks.get(task_index, next(iter(tasks.values()))))
PY
)}"

episode_label="${EPISODES_ARG// /_}"
episode_label="${episode_label//,/_}"
episode_label="${episode_label//:/_}"
JOB_NAME="${JOB_NAME:-random_${ROBOT}_${GRIPPER}_${TASK}_episodes_${episode_label}_steps_${MAX_STEPS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/${JOB_NAME}}"

mkdir -p "${OUTPUT_DIR}/logs"
rm -f "${OUTPUT_DIR}/results.json"
rm -f "${OUTPUT_DIR}"/videos/*.mp4 2>/dev/null || true

episode_args=()
if [[ "${EPISODES_ARG}" == first:* ]]; then
  episode_args+=(--num-episodes "${EPISODES_ARG#first:}")
else
  read -r -a selected_episodes <<< "${EPISODES_ARG//,/ }"
  episode_args+=(--episodes "${selected_episodes[@]}")
fi

echo "[random-eval] robot=${ROBOT} gripper=${GRIPPER} task=${TASK}"
echo "[random-eval] dataset=${DATASET}"
echo "[random-eval] output=${OUTPUT_DIR}"
echo "[random-eval] episodes=${EPISODES_ARG}"
echo "[random-eval] max_steps=${MAX_STEPS}"
echo "[random-eval] random_action_scale=${RANDOM_ACTION_SCALE}"
echo "[random-eval] task_text=${TASK_TEXT}"

exec "${PYTHON}" "${ROOT}/eval_lerobot_checkpoint.py" \
  --random-actions \
  --random-action-scale "${RANDOM_ACTION_SCALE}" \
  --dataset "${DATASET}" \
  --lerobot-src /data1/local/lerobot-VAI/src \
  --robosuite-src /data1/local/robosuite \
  --robocasa-src "${BARX_ROOT}/robocasa_x" \
  --robocasa-x-compat \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --task "${TASK_TEXT}" \
  --robot "${ROBOT}" \
  --gripper "${GRIPPER}" \
  "${episode_args[@]}" \
  --max-steps "${MAX_STEPS}" \
  --save-videos \
  --num-video-episodes "${NUM_VIDEO_EPISODES}" \
  --video-cameras ${VIDEO_CAMERAS} \
  --video-skip "${VIDEO_SKIP}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/random_eval_$(date +%Y%m%d_%H%M%S).log"
