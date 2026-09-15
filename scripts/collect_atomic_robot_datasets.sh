#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."
CONDA_ENV="${CONDA_ENV:-robocasa}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODE="${MODE:-all}"
TASK_NAME="${TASK_NAME:-}"
ROBOT="${ROBOT:-PandaOmron}"
GRIPPER_TYPES="${GRIPPER_TYPES:-default}"
COLLECT_ROOT="${COLLECT_ROOT:-${REPO_ROOT}/datasets/generated}"
CONFIG="${CONFIG:-single-arm-opposed}"
SPLIT="${SPLIT:-pretrain}"
CAMERA="${CAMERA:-robot0_frontview}"
CONTROLLER="${CONTROLLER:-OSC_POSE}"
LAYOUT="${LAYOUT:-}"
STYLE="${STYLE:-}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/datasets/v1.0}"
VIDEO_CAMERAS="${VIDEO_CAMERAS:-robot0_agentview_left robot0_eye_in_hand}"
VIDEO_SKIP="${VIDEO_SKIP:-5}"
MAX_EPISODES="${MAX_EPISODES:-}"

if [[ "${MODE}" == "replay" ]]; then
  replay_cmd=(
    python3 -m robocasa.scripts.dataset_scripts.replay_atomic_robot_datasets
    --source-root "${SOURCE_ROOT}"
    --output-root "${COLLECT_ROOT}"
    --split "${SPLIT}"
    --robot "${ROBOT}"
    --gripper-type "${GRIPPER_TYPES}"
    --camera-names ${VIDEO_CAMERAS}
    --video-skip "${VIDEO_SKIP}"
  )

  if [[ -n "${TASK_NAME}" ]]; then
    replay_cmd+=(--task-name "${TASK_NAME}")
  fi
  if [[ -n "${CONTROLLER}" ]]; then
    replay_cmd+=(--controller "${CONTROLLER}")
  fi
  if [[ -n "${MAX_EPISODES}" ]]; then
    replay_cmd+=(--max-episodes "${MAX_EPISODES}")
  fi

  echo "Replaying atomic datasets with robot=${ROBOT} gripper=${GRIPPER_TYPES}"
  "${replay_cmd[@]}"
  exit 0
fi

if [[ "${MODE}" == "one" ]]; then
  if [[ -z "${TASK_NAME}" ]]; then
    echo "Error: TASK_NAME is required when MODE=one."
    exit 1
  fi
  TASKS=("${TASK_NAME}")
else
  mapfile -t TASKS < <(
    python3 - <<'PY'
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS

for task_name in ATOMIC_TASK_DATASETS.keys():
    print(task_name)
PY
  )
fi

for task in "${TASKS[@]}"; do
  task_out="${COLLECT_ROOT}/${ROBOT}/${GRIPPER_TYPES}/${task}"
  mkdir -p "${task_out}"

  cmd=(
    python3 -m robocasa.scripts.collect_demos
    --directory "${task_out}"
    --environment "${task}"
    --robots "${ROBOT}"
    --gripper_types "${GRIPPER_TYPES}"
    --config "${CONFIG}"
    --camera "${CAMERA}"
    --controller "${CONTROLLER}"
    --split "${SPLIT}"
  )

  if [[ -n "${LAYOUT}" ]]; then
    cmd+=(--layout "${LAYOUT}")
  fi
  if [[ -n "${STYLE}" ]]; then
    cmd+=(--style "${STYLE}")
  fi

  echo "Collecting task=${task} robot=${ROBOT} gripper=${GRIPPER_TYPES} -> ${task_out}"
  "${cmd[@]}"
done
