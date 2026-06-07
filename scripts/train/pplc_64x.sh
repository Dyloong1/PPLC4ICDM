#!/usr/bin/env bash
# Train the headline 64x PPLC checkpoint.
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/jhtdb}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/pplc_64x_reproduced}"

python -m pplc.train \
    --config configs/pplc_64x.yaml \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
