# Install

Tested on Ubuntu 22.04 / 24.04 with Python 3.10–3.12 and CUDA 12.1–12.6.

## 1. Conda env (recommended)

```bash
conda create -n pplc python=3.11 -y
conda activate pplc
```

## 2. PyTorch

Pick the build matching your CUDA version. For CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 3. Project requirements

```bash
pip install -r requirements.txt
```

## 4. JHTDB account

JHTDB requires a free authentication token to download via SOAP/REST.
Register at https://turbulence.pha.jhu.edu/, then put your token in
`~/.jhtdb_token` (a single line).
`data/download_jhtdb.py` reads from this path by default.

## 5. Hardware

- **Eval (pretrained)**: 1× consumer GPU with ≥ 24 GB VRAM is enough
  to run `scripts/eval/zeroshot_1024.py` for the learned baselines.
  Analytic methods run on CPU; TT-SVD allocates ~25 GB RAM.
- **Train from scratch**: 1× RTX 5090 (or A100/A6000) per method;
  one PPLC 64× training run is ~24h; the 6 learned baselines together
  are ~6 GPU-days.

## 6. Disk

- JHTDB test frames 800/900/1000: 3 × 16 GB float32 = ~48 GB
- Training frames 0–700 at 256³: ~70 GB
- Pretrained checkpoints: ~3 GB (PPLC + 6 baselines + 4 forecasters)
- POD basis (K=2048): ~6 GB (optional download)
