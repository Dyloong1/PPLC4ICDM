#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/ltx_3d}"
python -m baselines.learned.ltx_3d.train \
    --config configs/baselines/ltx_3d.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
