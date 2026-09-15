#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/local/robocasa
ACTIVE_SESSION=${ACTIVE_SESSION:-dp_embodiment_eval}
NUM_EPISODES=${NUM_EPISODES:-100}
NUM_VIDEO_EPISODES=${NUM_VIDEO_EPISODES:-20}
VARIANTS=(scratch online finetune)

echo "[$(date -Is)] waiting for the running frozen evaluation ($ACTIVE_SESSION)"
while tmux has-session -t "$ACTIVE_SESSION" 2>/dev/null; do
  sleep 60
done

for variant in "${VARIANTS[@]}"; do
  checkpoint="$ROOT/outputs/train/dp_embodiment/$variant/checkpoint_030000"
  if [[ ! -d "$checkpoint" ]]; then
    echo "[$(date -Is)] missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  echo "[$(date -Is)] starting $variant (6 embodiments)"
  env \
    VARIANT="$variant" \
    CHECKPOINT="$checkpoint" \
    NUM_EPISODES="$NUM_EPISODES" \
    NUM_VIDEO_EPISODES="$NUM_VIDEO_EPISODES" \
    "$ROOT/run_dp_embodiment_barx_pnpsink_parallel.sh"
  echo "[$(date -Is)] completed $variant"
done

echo "[$(date -Is)] all 24 evaluations completed (frozen + 3 queued variants)"
