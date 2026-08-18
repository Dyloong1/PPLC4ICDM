#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/forecasters/pixel_unet_ar}"
python -m forecasters.pixel_unet_ar.train \
    --config configs/forecasters/pixel_unet_ar.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
