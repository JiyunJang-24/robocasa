#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."

export MODE="${MODE:-replay}"
export ROBOT="${ROBOT:-UR5eOmron}"
export GRIPPER_TYPES="${GRIPPER_TYPES:-Robotiq85Gripper}"
export SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/datasets/v1.0}"
export COLLECT_ROOT="${COLLECT_ROOT:-${REPO_ROOT}/datasets/generated/ur5e_robotiq85}"
export SPLIT="${SPLIT:-pretrain}"
export TASK_NAME="${TASK_NAME:-}"
export MAX_EPISODES="${MAX_EPISODES:-}"
export VIDEO_CAMERAS="${VIDEO_CAMERAS:-robot0_eye_in_hand robot0_agentview_left}"
export VIDEO_SKIP="${VIDEO_SKIP:-5}"
export MODE="replay"

bash "${SCRIPT_DIR}/collect_atomic_robot_datasets.sh"
