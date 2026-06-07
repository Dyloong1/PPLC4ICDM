#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/forecasters/latent_unet_ar}"
PPLC_CKPT="${PPLC_CKPT:-./checkpoints/pplc_64x.pt}"
python -m forecasters.latent_unet_ar.train \
    --config configs/forecasters/latent_unet_ar.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --pplc_ckpt "${PPLC_CKPT}" \
    "$@"
