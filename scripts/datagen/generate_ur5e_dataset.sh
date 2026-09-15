#!/usr/bin/env bash
# Generate UR5eOmron-reproduced atomic datasets from the PandaOmron source demos.
#
# Usage:
#   bash generate_ur5e_dataset.sh                          # default: CloseDrawer, all episodes
#   TASK=OpenDrawer bash generate_ur5e_dataset.sh          # one task
#   TASK=all bash generate_ur5e_dataset.sh                 # every atomic task (base-motion tasks included; check summary.json)
#   MAX_EPISODES=10 TASK=CloseDrawer bash generate_ur5e_dataset.sh
#
# Env vars:
#   TASK          atomic task name or "all"        (default: CloseDrawer)
#   MAX_EPISODES  cap per task, empty = all        (default: empty)
#   TARGET_ROBOT  robosuite robot name             (default: UR5eOmron)
#   TARGET_GRIPPER gripper ("default" = UR5e's Robotiq85) (default: default)
#   OUTPUT_ROOT   output directory root            (default: datasets/reproduced_ur5e)
#   KEEP_FAILED   1 = also save failed rollouts    (default: 0)
#   VIDEO_SKIP    render every Nth frame           (default: 5)
set -euo pipefail

ROOT="/data1/local/robocasa"
ATOMIC_ROOT="${ROOT}/datasets/v1.0/pretrain/atomic"

TASK="${TASK:-AdjustToasterOvenTemperature}"
MAX_EPISODES="${MAX_EPISODES:-}"
TARGET_ROBOT="${TARGET_ROBOT:-UR5eOmron}"
TARGET_GRIPPER="${TARGET_GRIPPER:-default}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/reproduced_ur5e}"
KEEP_FAILED="${KEEP_FAILED:-1}"
VIDEO_SKIP="${VIDEO_SKIP:-5}"

# conda activate robocasa (with a robust fallback to the env's absolute python,
# because conda activate can pick up a wrong python if the shell PATH is dirty)
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate robocasa 2>/dev/null || true
PYTHON_BIN="${CONDA_BASE}/envs/robocasa/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset CUDA_VISIBLE_DEVICES

run_one_task () {
    local task="$1"
    local args=(--task "${task}"
                --target-robot "${TARGET_ROBOT}"
                --target-gripper "${TARGET_GRIPPER}"
                --output-root "${OUTPUT_ROOT}"
                --video-skip "${VIDEO_SKIP}")
    if [ -n "${MAX_EPISODES}" ]; then
        args+=(--max-episodes "${MAX_EPISODES}")
    fi
    if [ "${KEEP_FAILED}" = "1" ]; then
        args+=(--keep-failed)
    fi
    echo "=== [${task}] target=${TARGET_ROBOT}/${TARGET_GRIPPER} ==="
    "${PYTHON_BIN}" "${ROOT}/reproduce_ur5e_dataset.py" "${args[@]}"
}

cd "${ROOT}"
if [ "${TASK}" = "all" ]; then
    for task_dir in "${ATOMIC_ROOT}"/*/; do
        task="$(basename "${task_dir}")"
        run_one_task "${task}" || echo "[warn] task ${task} failed; continuing"
    done
else
    run_one_task "${TASK}"
fi
