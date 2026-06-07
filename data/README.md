# Dataset — JHTDB `isotropic1024coarse`

Single source of truth for the data side of the paper. Satisfies the
ICDM Reproducibility checklist items Q3.1–Q3.5.

---

## Q3.1 Statistics

| Property | Value |
|---|---|
| Source | Johns Hopkins Turbulence Database (JHTDB), dataset `isotropic1024coarse` |
| Physics | Forced homogeneous isotropic turbulence, 3D direct numerical simulation (DNS) of incompressible Navier–Stokes |
| Spatial grid | `1024 × 1024 × 1024` uniform |
| Domain | `[0, 2π]³` (dimensionless) |
| Channels | 4 — velocity components `u, v, w` and pressure `p` |
| Floating point | float32 |
| Per-frame size | `4 × 1024³ × 4 B = 16 GB` |
| Frames available | 1024 |
| Frame spacing | `Δt = 0.05` (dimensionless) |
| Kinematic viscosity | `ν = 1.85 × 10⁻⁴` |
| Reynolds number | `Re_λ ≈ 433` |
| Kolmogorov time scale | `τ_η ≈ 0.044` |
| Integral time scale | `T_L ≈ 1.99` |

Total dataset size at full 1024³: **~16 TB raw**. For training we
sub-sample to 256³ (stride-4); only 3 frames are needed at full 1024³
for evaluation.

## Q3.2 Train / val / test splits

The same JHTDB trajectory is used for compressor training and forecaster
training, but with **different splits** to keep evaluation honest:

### Compressor (Table 1, 2, 3)

| Split | Frames | Resolution | Purpose |
|---|---|---|---|
| Train | 0 – 699 (700 frames) | 256³ (stride-4 from 1024³) | PPLC + 6 learned baselines |
| Val | 700 – 799 (100 frames) | 256³ | model selection (best.pt) |
| Test | **800, 900, 1000** (3 frames) | full 1024³ | zero-shot reconstruction quality |

The 1024³ test frames are **never seen during training**. PPLC is trained
exclusively at 256³ and evaluated zero-shot on 1024³; the "PPLC (native
1024)" upper-bound row trains on 1024³ patches drawn from frames 0–699
for an in-distribution comparison.

### Forecaster (Table 4)

All at 256³ — frozen 64× PPLC encoder, in-distribution forecasting:

| Split | Frames | Purpose |
|---|---|---|
| Train | 0 – 799 (800 frames) | forecaster fitting |
| Val | 800 – 899 (100 frames) | model selection |
| Test | 900 – 999 (100 frames) | reported RMSE @ τ=10/20, FSS @ τ=20 |

## Q3.3 Pre-processing and exclusions

**No data was excluded.** All available JHTDB frames in the split
ranges above are used.

Pre-processing pipeline (compressor training):

1. **Download** full 1024³ frames via `data/download_jhtdb.py` (uses
   pyJHTDB SOAP/REST).
2. **Stride-4 down-sample** to 256³ via 4³ box-averaging (the
   field-channel equivalent of strided pooling).
3. **Per-channel z-score normalize** using statistics computed over the
   training split (frames 0–699 at 256³). Statistics are saved to
   `data/jhtdb/stats.npz` and applied to all subsequent splits.
4. **32³ random patch crop** during training (one patch per frame load
   default; controlled by `--patches_per_load`).

For 1024³ zero-shot evaluation:

1. Download frames 800/900/1000 at full 1024³ via `download_jhtdb.py`.
2. Normalize using **the same statistics from step 3 above** (no
   re-fitting on the test split).
3. Tile into 32³ patches (stride-32 for naive reassembly, stride-16
   for Hann overlap-add).
4. Compress + decompress + reassemble back to 1024³.

For the 4 analytic baselines (Stride-4, POD, Wavelet, ZFP, TT-SVD):
no normalization needed — they operate on the raw field directly.

## Q3.4 Downloadable link

JHTDB landing page: https://turbulence.pha.jhu.edu/

Direct dataset page: https://turbulence.pha.jhu.edu/datasets.aspx
(scroll to "Forced Isotropic Turbulence Dataset (1024 cube)").

Access requires registration (free; instant approval for academic
emails). After registration you receive an **authentication token**;
paste it into `~/.jhtdb_token` (a single line, no quotes).

Then run:

```bash
# Download the 3 test frames (~48 GB)
python data/download_jhtdb.py --frames 800 900 1000 --out data/jhtdb/

# Or download the full training range (~11 TB; ~16 GB per frame × 700)
python data/download_jhtdb.py --frames 0-699 --out data/jhtdb/
```

The downloader skips already-downloaded frames and verifies SHA256
checksums against the JHTDB-provided metadata.

## Q3.5 New data collection

**N/A** — JHTDB is a publicly available DNS dataset. No new data was
collected for this work.

---

## Files

| File | Created by |
|---|---|
| `download_jhtdb.py` | this repo (small wrapper around pyJHTDB) |
| `jhtdb/frame_NNNNN.h5` | runtime — gitignored |
| `jhtdb/stats.npz` | runtime — per-channel z-score statistics from train split |
