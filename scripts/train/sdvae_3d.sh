#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/sdvae_3d}"
python -m baselines.learned.sdvae_3d.train \
    --config configs/baselines/sdvae_3d.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
