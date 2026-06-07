# Reproducing the paper

Step-by-step recipe for the ICDM reviewers. Every numeric value in
Tables 1–4 and every panel in Figures 1–4 can be regenerated from
this document.

---

## 0. Environment

```bash
git clone <repo> && cd PPLC4ICDM
pip install -r requirements.txt
```

Put your JHTDB token in `~/.jhtdb_token`. See [INSTALL.md](INSTALL.md)
for full details.

---

## 1. Reproduce **only the tables and figures**, from cached eval JSONs (no GPU, no download)

Smallest reproducible unit. Uses the cached per-method eval JSONs and
the pre-extracted z=512 slices NPZ shipped under `assets/`. ~60 s
end-to-end:

```bash
# Table 1 — main 64× zero-shot comparison
python scripts/tables/table1_main.py \
    --cache_dir assets/cached_eval_jsons/zeroshot_1024/ \
    --out tables/table1.md

# Table 2 — PPLC component ablation
python scripts/tables/table2_ablation.py \
    --cache_dir assets/cached_eval_jsons/ablation/ \
    --out tables/table2.md

# Table 3 — compression-ratio sweep
python scripts/tables/table3_compression_sweep.py \
    --cache_dir assets/cached_eval_jsons/sweep/ \
    --out tables/table3.md

# Table 4 — forecaster comparison
python scripts/tables/table4_forecaster.py \
    --cache_dir assets/cached_eval_jsons/forecast/ \
    --out tables/table4.md

# Figure 1 — baseline slice + error panel
python scripts/figures/fig1_baselines_with_error.py \
    --slices assets/slices_z512.npz \
    --cache_dir assets/cached_eval_jsons/zeroshot_1024/ \
    --out figures/fig1.png

# Figure 2 — PPLC zero-shot vs native-1024
python scripts/figures/fig2_zeroshot_vs_native.py \
    --slices assets/slices_z512.npz \
    --out figures/fig2.png

# Figure 3 — consistency-loss spectrum ablation
python scripts/figures/fig3_ablation_consist_spectrum.py \
    --cache_dir assets/cached_eval_jsons/ablation/ \
    --out figures/fig3.png

# Figure 4 — compression-vs-fidelity Pareto curve
python scripts/figures/fig4_compression_sweep_pareto.py \
    --cache_dir assets/cached_eval_jsons/sweep/ \
    --out figures/fig4.png
```

All output paths are deterministic — `tables/table1.md` and
`figures/fig1.png` should byte-match the versions in the paper appendix.

---

## 2. Re-evaluate the pretrained checkpoints (~2 hours on RTX 5090)

```bash
# Download test frames + checkpoints
python data/download_jhtdb.py --frames 800 900 1000 --out data/jhtdb/
bash checkpoints/download.sh

# Re-run the full Table 1 eval (13 methods × 3 frames)
python scripts/eval/zeroshot_1024.py \
    --method all \
    --frames 800 900 1000 \
    --data_dir data/jhtdb/ \
    --ckpt_dir checkpoints/ \
    --out_dir cache/zeroshot_1024/

# Then rebuild Table 1 / Figure 1 against the fresh caches
python scripts/tables/table1_main.py --cache_dir cache/zeroshot_1024/ --out tables/table1.md
python scripts/figures/fig1_baselines_with_error.py \
    --slices assets/slices_z512.npz \
    --cache_dir cache/zeroshot_1024/ \
    --out figures/fig1.png
```

Per-method invocations are also available:

```bash
python scripts/eval/zeroshot_1024.py --method pplc        --frames 800 900 1000 ...
python scripts/eval/zeroshot_1024.py --method sdvae_3d    --frames 800 900 1000 ...
python scripts/eval/zeroshot_1024.py --method wavelet     --frames 800 900 1000 ...
# etc.
```

---

## 3. Train from scratch

### 3.1 PPLC (~24h, 1× RTX 5090)

```bash
# Download training frames 0–699 at full 1024³ (~11 TB; or skip if you
# already have JHTDB data on disk and just want the 256³ stride-4 split)
python data/download_jhtdb.py --frames 0-699 --out data/jhtdb/

# Pre-compute z-score statistics (one-shot, ~10 min)
python scripts/train/compute_stats.py --data_dir data/jhtdb/ --out data/jhtdb/stats.npz

# Train PPLC headline 64× checkpoint
bash scripts/train/pplc_64x.sh

# Or with explicit args
python -m pplc.train \
    --config configs/pplc_64x.yaml \
    --data_dir data/jhtdb/ \
    --stats data/jhtdb/stats.npz \
    --save_dir checkpoints/pplc_64x_reproduced/ \
    --epochs 80 --patience 15
```

### 3.2 Learned baselines (~24h each, 6 total)

```bash
for m in sdvae_3d dcae_3d rae_3d wfvae_3d cosmos_3d ltx_3d; do
    bash scripts/train/${m}.sh
done
```

Each baseline reads its `configs/baselines/<name>.yaml` — these mirror
each paper's recommended config (cited in source comments).

### 3.3 Forecasters (~8h each, 4 total)

```bash
for f in latent_tx_cil latent_unet_ar pixel_tx_cil pixel_unet_ar; do
    bash scripts/train/forecasters/${f}.sh
done
```

Forecasters require a frozen PPLC encoder at `checkpoints/pplc_64x.pt`.

### 3.4 Analytic methods

POD requires fitting a basis (one-time, ~30 min CPU):

```bash
python baselines/analytic/pod/compute_basis.py \
    --data_dir data/jhtdb/ \
    --stats data/jhtdb/stats.npz \
    --K 2048 \
    --out checkpoints/pod_basis_K2048.npy
```

Wavelet, ZFP, TT-SVD have no training step — they are deterministic
analytic methods. Their compressors are called directly from
`scripts/eval/zeroshot_1024.py`.

---

## 4. Mapping from paper to code

| Paper element | Code path | CLI |
|---|---|---|
| Section 3 (PPLC architecture) | `pplc/model.py` | `python -m pplc.model --info` |
| Section 3.2 (Haar wavelet front-end) | `pplc/haar_wavelet.py` | — |
| Section 3.3 (training losses) | `pplc/losses.py` | — |
| Section 3.4 (Hann reassembly) | `pplc/reassemble.py` | — |
| Section 4.1 (analytic baselines) | `baselines/analytic/` | `--method {stride4,pod,wavelet,zfp,tt_svd}` |
| Section 4.2 (learned baselines) | `baselines/learned/` | `--method {sdvae,dcae,rae,wfvae,cosmos,ltx}_3d` |
| Section 4.3 (forecaster setup) | `forecasters/` | `--task forecast --model {latent,pixel}_{tx_cil,unet_ar}` |
| Section 5.1 (Table 1 main comparison) | `scripts/tables/table1_main.py` | |
| Section 5.2 (Table 2 component ablation) | `scripts/tables/table2_ablation.py` | |
| Section 5.3 (Table 3 compression sweep) | `scripts/tables/table3_compression_sweep.py` | |
| Section 5.4 (Table 4 forecasting) | `scripts/tables/table4_forecaster.py` | |
| Figure 1 (baseline panel) | `scripts/figures/fig1_baselines_with_error.py` | |
| Figure 2 (zero-shot vs native) | `scripts/figures/fig2_zeroshot_vs_native.py` | |
| Figure 3 (consistency spectrum) | `scripts/figures/fig3_ablation_consist_spectrum.py` | |
| Figure 4 (Pareto curve) | `scripts/figures/fig4_compression_sweep_pareto.py` | |

---

## 5. Ablation-grid hyperparameter selection

PPLC's headline hyperparameters
(`β_KL=0.01, λ_adv=0.01, λ_grad=0.5, λ_consist=0.1, consist_shift=8`)
were selected by ablation on the 256³ training split **before** the
1024³ eval frames were touched. To rerun the selection sweep:

```bash
bash scripts/train/pplc_ablation_sweep.sh
```

This trains the 2×2 cross-product (latent layout × consistency loss)
at 64× — channel-heavy (32×4³) vs spatial (4×8³), with and without
consistency. ~4× 24h GPU-days. The headline config is the
`spatial + consistency` cell.
