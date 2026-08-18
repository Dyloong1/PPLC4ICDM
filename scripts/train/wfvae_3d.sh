#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/wfvae_3d}"
python -m baselines.learned.wfvae_3d.train \
    --config configs/baselines/wfvae_3d.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
