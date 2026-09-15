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

mapfile -t ATOMIC_TASKS < <(
  python3 - <<'PY'
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS

for task_name in ATOMIC_TASK_DATASETS.keys():
    print(task_name)
PY
)

python3 -m robocasa.scripts.download_datasets \
  --tasks "${ATOMIC_TASKS[@]}" \
  --source human mimicgen \
  --split pretrain target \
  "$@"
