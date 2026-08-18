# Baselines

Two families:

## Analytic (deterministic, no training)

| Folder | Method | Reference | 64× knob |
|---|---|---|---|
| `analytic/stride4/` | 4³ box-average + trilinear up-sample | null baseline | exact 64× by sub-sampling |
| `analytic/pod/` | POD (proper orthogonal decomposition) | Lumley 1967 | `K_modes = 2048` |
| `analytic/wavelet/` | Daubechies db4, level 3 + thresholding | Daubechies 1988 | threshold → 64× |
| `analytic/zfp/` | ZFP accuracy mode | Lindstrom 2014 IEEE TVCG | accuracy tuned to 64× |
| `analytic/tt_svd/` | Quantics MPS / TT-SVD | Oseledets 2011 SIAM | `bond_dim = 9` → 68.6× |

Each folder has a `compress.py` exposing `compress(field) → bytes` and
`decompress(bytes) → field`. POD additionally has `compute_basis.py`
(one-time fit on the training split; produces `pod_basis_K2048.npy`).

## Learned (~40 M parameter budget, all 3D ports)

| Folder | Method | Reference | Headline config |
|---|---|---|---|
| `learned/sdvae_3d/` | Stable Diffusion VAE (3D port) | Rombach 2022 CVPR | `embed_dim=4, z_channels=4` |
| `learned/dcae_3d/` | DC-AE (3D port) | Chen 2025 ICLR | `latent_channels=4, width=(64,128,256)` |
| `learned/rae_3d/` | RAE (regularized autoencoder, 3D port) | Zheng 2025 arXiv:2510.11690 | `latent_ch=4`, frozen SD-VAE-3D encoder |
| `learned/wfvae_3d/` | WF-VAE (wavelet-flow VAE, 3D port) | Li 2025 CVPR | `ch=128, z_channels=4`, energy-flow synthesized from `z` (paper would store it side-channel, breaks 64×; we keep it internal — see note in `wfvae_3d/README.md`) |
| `learned/cosmos_3d/` | NVIDIA Cosmos-CV tokenizer (factorized 3D conv pure AE) | NVIDIA 2025 arXiv:2501.03575 | `ch=96, z_channels=4` |
| `learned/ltx_3d/` | LTX-Video VAE (3D port) | Hacohen 2025 arXiv:2501.00103 | `ch=96, z_channels=4, noise_sigma=0.2`; we evaluate the VAE only (no DiT) |

Each folder has `model.py`, `train.py`, `README.md` (architecture notes
+ deviation list from the original paper if any), and a corresponding
`configs/baselines/<name>.yaml`.

All learned baselines produce a spatial-tensor latent at 1/64 the
voxel count — apples-to-apples with PPLC.

## Why these and not others

The paper's Section 4 explains the exclusions:
- **CNF** (Guo 2025) needs per-frame decoder retraining; not an
  amortized encoder. Not commensurate with the 64× tier.
- **FAE** (Bunker 2025) collapses the full 1024³ to a 256-float global
  descriptor (≈ 16.8M× by Bunker's convention). Not a spatial-tensor
  latent.
