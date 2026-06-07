# PPLC core

Implementation of the Patch-Pivot Latent Compressor (Section 3 of the
paper). All four pieces are in this directory:

| File | What it implements |
|---|---|
| `model.py` | The PPLC architecture: per-patch mean–fluctuation split (mean stored exactly as a 4-vector) + reversible 3D Haar wavelet front-end + single stride-2 conv encoder + `ConvTranspose3d` decoder + inverse Haar. 64× tier uses `latent_channels=4`, spatial latent shape `(4, 8, 8, 8) = 2052 floats per 32³ patch ≈ 63.9×`. |
| `haar_wavelet.py` | Level-3 3D Haar forward / inverse transform. Zero-parameter, exactly invertible. Used as a front-end to decouple coarse-scale energy from fine-scale fluctuations. |
| `losses.py` | Training objective `L_G = L1 + 0.5·L_grad + 0.01·L_KL + 0.01·L_adv + 0.1·L_consist` plus the 3D patch discriminator (hinge loss). Consistency loss is `L1(D(z(x)), D(z(shift_k(x))))` with circular shift `k = 8` voxels — enforces approximate shift-equivariance of the latent. |
| `reassemble.py` | Both reassembly schemes: `reassemble_naive` (stride-32, non-overlapping; matches the time-cost number in Table 1) and `reassemble_hann` (stride-16 with Hann overlap-add, 2³ patch coverage; the headline reconstruction-quality number). |
| `dataset.py` | 32³ random patch sampler from 256³ pre-normalized frames. |

## Architecture summary

- Input: a 32³ field with 4 channels (u, v, w, p)
- Mean–fluctuation split: 4-vector mean stored exactly; the rest goes
  through the learned codec
- Level-3 3D Haar wavelet: 4×32³ → 32×4³ (8 sub-bands × 4 channels)
- Encoder: one stride-2 3D conv → `(latent_channels, 8, 8, 8)` latent
  block. Headline 64×: `latent_channels = 4`, so the latent is
  `4 × 8 × 8 × 8 = 2048` floats per 32³ patch.
- Decoder: mirror — `ConvTranspose3d` to upscale, then inverse Haar,
  then add mean back.
- Total params: ~45 M.

## Training

```bash
python -m pplc.train --config ../configs/pplc_64x.yaml
```

Defaults from the headline run (matches the camera-ready checkpoint):

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, `(β₁, β₂) = (0.9, 0.999)` |
| Gen learning rate | `1e-4` |
| Disc learning rate | `5e-5` |
| Mixed precision | bfloat16 |
| Epochs | 80 (early-stop, patience 15) |
| Effective batch (patches) | 256 |
| Consist shift `k` | 8 voxels (≈ 25% of patch side) |
| `β_KL` | 0.01 |
| `λ_grad` | 0.5 |
| `λ_adv` | 0.01 |
| `λ_consist` | 0.1 |

All values are also serialized in `configs/pplc_64x.yaml`.
