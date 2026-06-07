#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/forecasters/pixel_tx_cil}"
python -m forecasters.pixel_tx_cil.train \
    --config configs/forecasters/pixel_tx_cil.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
