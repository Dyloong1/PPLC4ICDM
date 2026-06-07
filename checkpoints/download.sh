#!/usr/bin/env bash
# Download pretrained PPLC + baseline + forecaster checkpoints.
#
# The actual release URL will be inserted post-rebuttal; for the
# anonymous submission window, this file is a placeholder that
# documents the expected layout. Reviewers who want to test the eval
# pipeline before the release can train PPLC themselves with
# `bash scripts/train/pplc_64x.sh` (~24h on RTX 5090) and skip the
# rest by setting `--method pplc` in the eval driver.
set -euo pipefail

CKPT_DIR="$(dirname "$(readlink -f "$0")")"
RELEASE_URL="${PPLC_RELEASE_URL:-https://example.com/pplc-icdm2026-release}"

mkdir -p "$CKPT_DIR/baselines" "$CKPT_DIR/forecasters"

# ===== PPLC (headline + native-1024 upper bound) =====
curl -L "$RELEASE_URL/pplc_64x.pt"             -o "$CKPT_DIR/pplc_64x.pt"
curl -L "$RELEASE_URL/pplc_native_1024.pt"     -o "$CKPT_DIR/pplc_native_1024.pt"

# ===== 6 learned baselines =====
for name in sdvae_3d dcae_3d rae_3d wfvae_3d cosmos_3d ltx_3d; do
    curl -L "$RELEASE_URL/baselines/${name}.pt" -o "$CKPT_DIR/baselines/${name}.pt"
done

# ===== 4 forecasters =====
for name in latent_tx_cil latent_unet_ar pixel_tx_cil pixel_unet_ar; do
    curl -L "$RELEASE_URL/forecasters/${name}.pt" -o "$CKPT_DIR/forecasters/${name}.pt"
done

# ===== POD basis (optional, ~6 GB) =====
if [[ "${INCLUDE_POD_BASIS:-0}" == "1" ]]; then
    curl -L "$RELEASE_URL/pod_basis_K2048.npy" -o "$CKPT_DIR/pod_basis_K2048.npy"
fi

echo "Done. Checkpoints in $CKPT_DIR/"
