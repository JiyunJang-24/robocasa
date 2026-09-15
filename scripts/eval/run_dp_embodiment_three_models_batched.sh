#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/local/robocasa
NUM_EPISODES=${NUM_EPISODES:-100}
NUM_VIDEO_EPISODES=${NUM_VIDEO_EPISODES:-20}
VARIANTS=(scratch finetune)
OFFSETS=(0 2)

for pair_start in 0 1 2 3 4 5; do
  echo "[$(date -Is)] starting embodiment batch $((pair_start + 1))/6"
  pids=()
  for i in "${!VARIANTS[@]}"; do
    variant=${VARIANTS[$i]}
    checkpoint="$ROOT/outputs/train/dp_embodiment/$variant/checkpoint_030000"
    env \
      VARIANT="$variant" \
      GPU_OFFSET="${OFFSETS[$i]}" \
      PAIR_START="$pair_start" \
      PAIR_COUNT=1 \
      CHECKPOINT="$checkpoint" \
      NUM_EPISODES="$NUM_EPISODES" \
      NUM_VIDEO_EPISODES="$NUM_VIDEO_EPISODES" \
      "$ROOT/run_dp_embodiment_barx_pnpsink_parallel.sh" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if ((status != 0)); then
    echo "[$(date -Is)] batch failed; inspect per-embodiment logs" >&2
    exit 1
  fi
  echo "[$(date -Is)] completed embodiment batch $((pair_start + 1))/6"
done

echo "[$(date -Is)] scratch and finetune evaluations completed"
