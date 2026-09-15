#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/local/robocasa
PYTHON=${PYTHON:-/data1/local/lerobot-VAI/venv-lerobot-eval/bin/python}
CHECKPOINT=${CHECKPOINT:-$ROOT/outputs/train/dp_embodiment/frozen/checkpoint_030000}
VARIANT=${VARIANT:-$(basename "$(dirname "$CHECKPOINT")")}
DATASET=${DATASET:-/data1/local/barx/datasets/atomic_new/UR5eOmron/pretrain/PnPSinkToCounter/lerobot}
OUT_ROOT=${OUT_ROOT:-$ROOT/outputs/eval}
NUM_EPISODES=${NUM_EPISODES:-100}
NUM_VIDEO_EPISODES=${NUM_VIDEO_EPISODES:-20}
MAX_STEPS=${MAX_STEPS:-}
FIXED_CAMERA_EPISODE=${FIXED_CAMERA_EPISODE:-}
DEVICE=${DEVICE:-cuda}
GPU_OFFSET=${GPU_OFFSET:-0}
LAUNCH_STAGGER_SECONDS=${LAUNCH_STAGGER_SECONDS:-0}
TASK=PnPSinkToCounter
TASK_TEXT=${TASK_TEXT:-$("$PYTHON" - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

with (Path(sys.argv[1]) / "meta" / "tasks.jsonl").open() as f:
    print(json.loads(f.readline())["task"])
PY
)}
PREFIX=${PREFIX:-eval_plus100_barx_pnpsink_dp_embodiment_${VARIANT}_eval_all}
PAIRS=("IIWAOmron Robotiq140Gripper" "JacoOmron PandaGripper" "JacoOmron Robotiq85Gripper" "PandaOmron PandaGripper" "PandaOmron Robotiq85Gripper" "UR5eOmron Robotiq85Gripper")
PAIR_START=${PAIR_START:-0}
PAIR_COUNT=${PAIR_COUNT:-${#PAIRS[@]}}
PAIRS=("${PAIRS[@]:PAIR_START:PAIR_COUNT}")
export MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONUNBUFFERED=1
export PYTHONPATH="/data1/local/robosuite:/data1/local/barx/robocasa_x:/data1/local/lerobot-VAI/src:$ROOT/.eval_extra_site:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
if [[ "${ALLOW_EXISTING_OUTPUT:-0}" != 1 ]]; then
  for pair in "${PAIRS[@]}"; do
    read -r robot grip <<< "$pair"
    out="$OUT_ROOT/${PREFIX}_${robot}_${grip}_${TASK}"
    if [[ -f "$out/results.json" ]] || find "$out/videos" -maxdepth 1 -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
      echo "[error] refusing to mix a new run with existing output: $out" >&2
      echo "Move/archive that directory first, or set ALLOW_EXISTING_OUTPUT=1 deliberately." >&2
      exit 1
    fi
  done
fi
if [[ "$DEVICE" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
      | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}'
  )
else
  GPU_IDS=()
fi
if [[ "$DEVICE" == cuda* ]] && ((${#GPU_IDS[@]} == 0)); then
  echo "[error] DEVICE=$DEVICE requested, but no usable NVIDIA GPU was detected" >&2
  exit 1
fi
pids=()
job_idx=0
for pair in "${PAIRS[@]}"; do
  read -r robot grip <<< "$pair"
  out="$OUT_ROOT/${PREFIX}_${robot}_${grip}_${TASK}"
  mkdir -p "$out/logs"
  if ((${#GPU_IDS[@]})); then
    gpu_id=${GPU_IDS[$(((job_idx + GPU_OFFSET) % ${#GPU_IDS[@]}))]}
    gpu_env=(env CUDA_VISIBLE_DEVICES="$gpu_id")
    echo "Launching $robot + $grip on physical GPU $gpu_id"
  else
    gpu_env=()
  fi
  eval_args=()
  if [[ -n "$MAX_STEPS" ]]; then
    eval_args+=(--max-steps "$MAX_STEPS")
  fi
  if [[ -n "$FIXED_CAMERA_EPISODE" ]]; then
    eval_args+=(--fixed-camera-episode "$FIXED_CAMERA_EPISODE")
  fi
  if [[ "$grip" == "Robotiq85Gripper" ]]; then
    eval_args+=(--gripper-donor-dataset "$DATASET")
  fi
  "${gpu_env[@]}" "$PYTHON" "$ROOT/eval_lerobot_checkpoint.py" --checkpoint "$CHECKPOINT" --dataset "$DATASET" \
    --lerobot-src /data1/local/lerobot-VAI/src --robosuite-src /data1/local/robosuite \
    --robocasa-src /data1/local/barx/robocasa_x --robocasa-x-compat --output-dir "$out" \
    --device "$DEVICE" --task "$TASK_TEXT" \
    --robot "$robot" --gripper "$grip" --num-episodes "$NUM_EPISODES" --extra-steps 100 \
    --save-videos --num-video-episodes "$NUM_VIDEO_EPISODES" --video-skip 2 "${eval_args[@]}" >"$out/logs/eval.log" 2>&1 &
  pids+=("$!")
  ((job_idx += 1))
  if ((LAUNCH_STAGGER_SECONDS > 0)) && ((job_idx < ${#PAIRS[@]})); then
    sleep "$LAUNCH_STAGGER_SECONDS"
  fi
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "Completed ${#pids[@]} BARX PnPSinkToCounter evaluations (status=$status)"
exit "$status"
