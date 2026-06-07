# PPLC4ICDM — code release for "PPLC: Patch-Pivot Latent Compression for Turbulence at 64×" (anonymous submission, ICDM 2026)

This repository reproduces every quantitative result, table, and figure
in the paper. It is intentionally minimal — only the code paths that
feed the paper are included; ~80 superseded ablations and auxiliary
physics diagnostics from the working repo are out of scope.

---

## What's reproducible

| Asset | Reproduced by |
|---|---|
| **Table 1** — 13 methods × 9 metrics, mean ± std across 3 zero-shot 1024³ eval frames | `scripts/tables/table1_main.py` |
| **Table 2** — PPLC component ablation (channel-heavy → spatial → +consistency → +Hann) | `scripts/tables/table2_ablation.py` |
| **Table 3** — Compression-ratio sweep (34×, 64×, 128×, 254×) + train-resolution ablation (128³ vs 256³) | `scripts/tables/table3_compression_sweep.py` |
| **Table 4** — Forecaster comparison (latent + Transformer + direct-τ vs latent + UNet + AR vs pixel ×2) | `scripts/tables/table4_forecaster.py` |
| **Figure 1** — 4-row × 13-col baseline slice / zoom / error / error-zoom panel, ordered by rel-L2 | `scripts/figures/fig1_baselines_with_error.py` |
| **Figure 2** — PPLC zero-shot vs PPLC native-1024 slice + diff comparison | `scripts/figures/fig2_zeroshot_vs_native.py` |
| **Figure 3** — Consistency-loss ablation, energy spectrum E(k) | `scripts/figures/fig3_ablation_consist_spectrum.py` |
| **Figure 4** — Compression-vs-fidelity Pareto curve | `scripts/figures/fig4_compression_sweep_pareto.py` |

All four figure scripts are also re-runnable from the cached eval
JSONs and slice NPZ already shipped under `assets/`, so figure
regeneration is < 60 seconds and needs no GPU.

---

## Quick start (eval pretrained models, no GPU training)

```bash
# 1. Environment
git clone <repo>
cd PPLC4ICDM
pip install -r requirements.txt

# 2. Download JHTDB test frames (frames 800, 900, 1000 of isotropic1024coarse)
#    Requires a free JHTDB account: https://turbulence.pha.jhu.edu/
python data/download_jhtdb.py --frames 800 900 1000 --out data/jhtdb/

# 3. Download pretrained checkpoints (~3 GB)
bash checkpoints/download.sh

# 4. Reproduce Table 1 (~2 hours on RTX 5090; 4 analytic methods are CPU)
python scripts/eval/zeroshot_1024.py \
    --method all \
    --frames 800 900 1000 \
    --data_dir data/jhtdb/ \
    --ckpt_dir checkpoints/ \
    --out_dir cache/zeroshot_1024/

python scripts/tables/table1_main.py \
    --cache_dir cache/zeroshot_1024/ \
    --out tables/table1.md

# 5. Render Figure 1 (uses cached slices + eval JSONs; < 60 s, no GPU)
python scripts/figures/fig1_baselines_with_error.py \
    --cache_dir cache/zeroshot_1024/ \
    --slices assets/slices_z512.npz \
    --out figures/fig1_baselines_with_error.png
```

Repeat steps 4–5 with `scripts/tables/table{2,3,4}_*.py` and
`scripts/figures/fig{2,3,4}_*.py` for the rest.

---

## Full reproduction (train from scratch)

See [REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md). One PPLC 64×
training run is ~24h on a single RTX 5090; the 6 learned baselines
together are ~6 GPU-days.

---

## Repo layout

```
PPLC4ICDM/
├── README.md                   ← you are here
├── REPRODUCING_THE_PAPER.md    ← step-by-step recipe
├── INSTALL.md                  ← env setup
├── requirements.txt            ← Q4.1
├── LICENSE
├── CITATION.cff
│
├── pplc/                       ← Section 0 of the paper (the method)
│   ├── model.py                ← spatial-8 4-channel PPLC architecture
│   ├── haar_wavelet.py         ← reversible level-3 3D Haar front-end
│   ├── losses.py               ← L1 + grad + KL + adv + consist
│   ├── reassemble.py           ← naive + Hann overlap-add reassembly
│   └── dataset.py              ← 32³ patch loader from 256³ frames
│
├── baselines/
│   ├── analytic/               ← Stride-4, POD, Wavelet, ZFP, TT-SVD
│   └── learned/                ← SD-VAE-3D, DC-AE-3D, RAE-3D, WF-VAE-3D, Cosmos-CV-3D, LTX-Video-VAE-3D
│
├── forecasters/                ← Table 4
│   ├── latent_tx_cil/          ← latent + Transformer + direct-τ + CIL (β_m=0.5)
│   ├── latent_unet_ar/         ← latent + UNet + AR (τ=10 trained, AR to τ=20)
│   ├── pixel_tx_cil/
│   └── pixel_unet_ar/
│
├── physics/                    ← shared evaluation module (apples-to-apples)
│   ├── metrics.py              ← ε, Ω, β, divergence
│   ├── spectra.py              ← E(k) radial shell-binned
│   └── reconstruction_metrics.py  ← rel-L1, rel-L2, MAE, RMSE, PSNR
│
├── scripts/
│   ├── train/                  ← one train.sh per method
│   ├── eval/                   ← zeroshot_1024.py + in_dist_256.py
│   ├── tables/                 ← table{1,2,3,4}.py
│   └── figures/                ← fig{1,2,3,4}.py
│
├── configs/                    ← all hyperparameters in YAML (PPLC + 6 baselines + 4 forecasters)
├── data/                       ← JHTDB downloader + dataset README
├── checkpoints/                ← pretrained model placeholders + download.sh
├── assets/                     ← cached slices NPZ + small visualizations
└── docs/                       ← architecture diagrams, derivations
```

---

## Headline results

PPLC headline numbers at 64× compression, zero-shot on JHTDB isotropic
1024³ frames 800/900/1000 (mean ± std across the 3 frames):

| Method | rel-L2 ↓ | ε ratio (→ 1.0) | Ω ratio (→ 1.0) | β (DNS ≈ −1.50) | rel-div ↓ |
|---|---:|---:|---:|---:|---:|
| Wavelet (db4, lev 3) | 0.0525 ± 0.0017 | 0.869 ± 0.006 | 0.828 ± 0.006 | −1.519 ± 0.012 | 6.99 ± 0.20 |
| WF-VAE-3D | 0.0503 ± 0.0014 | 1.076 ± 0.006 | 1.022 ± 0.006 | −1.517 ± 0.012 | 7.92 ± 0.19 |
| **PPLC (zero-shot, 256³→1024³)** | **0.0520 ± 0.0011** | **1.191 ± 0.011** | **1.109 ± 0.009** | **−1.522 ± 0.013** | **9.75 ± 0.13** |
| **PPLC (in-dist., 1024³)** | **0.0492 ± 0.0014** | **1.023 ± 0.008** | **0.965 ± 0.008** | **−1.517 ± 0.011** | **8.19 ± 0.16** |

The complete 13-method × 9-metric table (incl. rel-L1, MAE, RMSE, PSNR,
inference time) is produced by `scripts/tables/table1_main.py`.

---

## Dataset

JHTDB `isotropic1024coarse` — a publicly available DNS of forced
homogeneous isotropic turbulence at `Re_λ ≈ 433`.

- **Statistics**: 1024 frames at `Δt = 0.05` dimensionless; each frame
  is `4 × 1024 × 1024 × 1024` (u, v, w, p) float32. Domain `[0, 2π]³`,
  kinematic viscosity `ν = 1.85 × 10⁻⁴`.
- **Train / val / test split**:
  - Compressor: frames 0–700 train (down-sampled to 256³ via stride-4),
    700–800 val, **800 / 900 / 1000** test at full 1024³.
  - Forecaster: frames 0–800 train, 800–900 val, 900–1000 test
    (all at 256³, in-distribution).
- **Pre-processing**: per-channel z-score normalization, then 32³
  random patch cropping during training. No data exclusion.
- **Download**: `python data/download_jhtdb.py` (free JHTDB account
  required); see [`data/README.md`](data/README.md). Source URL:
  https://turbulence.pha.jhu.edu/datasets.aspx

---

## Pretrained models

Hosted at an anonymized Zenodo / Hugging Face release (TBD post-rebuttal).
`bash checkpoints/download.sh` fetches:

- `pplc_64x.pt` — PPLC headline checkpoint (zero-shot variant trained on 256³)
- `pplc_native_1024.pt` — PPLC trained on 1024³ (data control)
- `baselines/<name>_3d.pt` — six learned baselines (SD-VAE-3D, DC-AE-3D,
  RAE-3D, WF-VAE-3D, Cosmos-CV-3D, LTX-Video-VAE-3D)
- `forecasters/<name>.pt` — four forecasters
- `pod_basis_K2048.npy` — precomputed POD basis (~6 GB; optional)

---

## Citation

```bibtex
@inproceedings{pplc2026,
  title     = {PPLC: Patch-Pivot Latent Compression for Turbulence at 64×},
  author    = {Anonymous},
  booktitle = {ICDM},
  year      = {2026}
}
```

## License

MIT. JHTDB data is governed by the [JHTDB Terms of Use](https://turbulence.pha.jhu.edu/).
