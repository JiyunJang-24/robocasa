#!/usr/bin/env bash
set -euo pipefail

# Evaluate all language-conditioned diffusion checkpoints on PnPSinkToCounter.
# One process is launched per checkpoint/embodiment pair.
ROOT=/data1/local/robocasa
PYTHON=${PYTHON:-/data1/local/lerobot-VAI/venv-lerobot-eval/bin/python}
DATASET=${DATASET:-/data1/local/robocasa/datasets/visual_robust_new_35embod/UR5eOmron_PnPSinkToCounter/lerobot}
OUT=${OUT:-$ROOT/outputs/eval/dp_embodiment_pnp}
NUM_EPISODES=${NUM_EPISODES:-50}
DEVICE=${DEVICE:-cuda}
ROBOTS=(Panda IIWA UR5e Jaco Kinova3 Sawyer)
read -r -a MODES <<< "${MODES:-finetune}"  # set MODES='scratch online finetune frozen' for all four
mkdir -p "$OUT"
export MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONUNBUFFERED=1
export PYTHONPATH="/data1/local/robosuite:/data1/local/robocasa:/data1/local/lerobot-VAI/src:/data1/local/robocasa/.eval_extra_site:${PYTHONPATH:-}"

pids=()
for mode in "${MODES[@]}"; do
  ckpt="$ROOT/outputs/train/dp_embodiment/$mode/checkpoint_030000"
  for robot in "${ROBOTS[@]}"; do
    name="${mode}_${robot}"
    dir="$OUT/$name"
    mkdir -p "$dir/logs"
    "$PYTHON" "$ROOT/eval_lerobot_checkpoint.py" \
      --checkpoint "$ckpt" --dataset "$DATASET" \
      --lerobot-src /data1/local/lerobot-VAI/src \
      --robosuite-src /data1/local/robosuite \
      --robocasa-src /data1/local/barx/robocasa_x \
      --robocasa-x-compat --output-dir "$dir" --device "$DEVICE" \
      --task "Pick the object from the sink and place it on the counter." \
      --robot "$robot" --gripper Robotiq85Gripper \
      --num-episodes "$NUM_EPISODES" --extra-steps 50 \
      --save-videos --num-video-episodes 5 --video-skip 2 \
      >"$dir/logs/eval.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "Completed ${#pids[@]} evaluations; status=$status"
exit "$status"
