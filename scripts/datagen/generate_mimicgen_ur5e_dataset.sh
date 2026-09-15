#!/usr/bin/env bash
# Generate a UR5eOmron RoboCasa atomic dataset with MimicGen.
#
# Usage:
#   bash generate_mimicgen_ur5e_dataset.sh
#   TASK=AdjustToasterOvenTemperature NUM_DEMOS=5 bash generate_mimicgen_ur5e_dataset.sh
#   TASK=OpenDrawer SOURCE_N=20 NUM_DEMOS=50 NUM_PROCS=4 bash generate_mimicgen_ur5e_dataset.sh
set -euo pipefail

ROOT="/data1/local/robocasa"
CONDA_ENV="${CONDA_ENV:-robocasa}"

TASK="${TASK:-AdjustToasterOvenTemperature}"
SPLIT="${SPLIT:-pretrain}"
TARGET_ROBOT="${TARGET_ROBOT:-UR5eOmron}"
TARGET_GRIPPER="${TARGET_GRIPPER:-default}"
NUM_DEMOS="${NUM_DEMOS:-1}"
NUM_PROCS="${NUM_PROCS:-1}"
SOURCE_N="${SOURCE_N:-10}"
SEED="${SEED:-1}"
VIDEO_SKIP="${VIDEO_SKIP:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/datasets/mimicgen_ur5e}"
WORK_DIR="${WORK_DIR:-${ROOT}/datasets/mimicgen_work}"
VIDEO_CAMERA="${VIDEO_CAMERA:-robot0_agentview_center}"
ATTEMPTS_ONLY="${ATTEMPTS_ONLY:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
RENDER_ATTEMPT_VIDEO="${RENDER_ATTEMPT_VIDEO:-0}"
BASE_BACKOFF="${BASE_BACKOFF:-0.10}"
GRASP_DEPTH_EXTRA="${GRASP_DEPTH_EXTRA:-0.01}"
ROT_GAIN="${ROT_GAIN:-2.5}"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

PYTHON_BIN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${ROOT}:/data1/local/robosuite:${ROOT}/mimicgen:${ROOT}/robomimic:${PYTHONPATH:-}"

cd "${ROOT}"
cmd=(
    "${PYTHON_BIN}" "${ROOT}/generate_mimicgen_ur5e_dataset.py"
    --task "${TASK}" \
    --split "${SPLIT}" \
    --target-robot "${TARGET_ROBOT}" \
    --target-gripper "${TARGET_GRIPPER}" \
    --num-demos "${NUM_DEMOS}" \
    --num-procs "${NUM_PROCS}" \
    --source-n "${SOURCE_N}" \
    --seed "${SEED}" \
    --video-skip "${VIDEO_SKIP}" \
    --video-camera ${VIDEO_CAMERA} \
    --output-root "${OUTPUT_ROOT}" \
    --work-dir "${WORK_DIR}" \
    --base-backoff "${BASE_BACKOFF}" \
    --grasp-depth-extra "${GRASP_DEPTH_EXTRA}" \
    --rot-gain "${ROT_GAIN}"
)

if [ "${ATTEMPTS_ONLY}" = "1" ]; then
    cmd+=(--attempts-only)
fi
if [ "${PREPARE_ONLY}" = "1" ]; then
    cmd+=(--prepare-only)
fi
if [ "${RENDER_ATTEMPT_VIDEO}" = "1" ]; then
    cmd+=(--render-attempt-video)
fi

"${cmd[@]}"
