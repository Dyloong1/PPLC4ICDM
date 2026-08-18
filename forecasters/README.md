# Forecasters (Table 4)

Downstream task: given turbulence at time `t`, forecast the state at
`t + τ Δt` (frame spacing `Δt = 0.05`, dimensionless). All experiments
in-distribution at 256³, with frozen 64× PPLC encoder for the latent
variants.

| Folder | Modality | Architecture | Strategy | Context | Headline |
|---|---|---|---|---|---|
| `latent_tx_cil/` | Latent (frozen PPLC encoder) | Transformer | direct-τ + CIL (`β_m = 0.5`) | 3 latents `{z_{t-10}, z_{t-5}, z_t}` | **best (Table 4 row 1)** |
| `latent_unet_ar/` | Latent | UNet | autoregressive (trained at `τ = 10`, AR-rolled to `τ = 20`) | single latent `z_t` | Table 4 row 2 |
| `pixel_tx_cil/` | Pixel | Transformer | direct-τ + CIL | 3 frames | Table 4 row 3 |
| `pixel_unet_ar/` | Pixel | UNet | autoregressive | single frame | Table 4 row 4 |

Each folder has `model.py`, `train.py`, `README.md`, and a YAML config
under `configs/forecasters/<name>.yaml`.

## Data split (forecaster-specific)

256³ frames at native JHTDB resolution after stride-4 down-sampling:

| Split | Frames | Use |
|---|---|---|
| Train | 0 – 799 | forecaster fitting |
| Val | 800 – 899 | model selection |
| Test | 900 – 999 | reported metrics |

## Metrics

- **RMSE @ τ ∈ {10, 20}**: pixel-space root-mean-square error against
  the DNS ground truth at `t + τ Δt` after **decoding** the latent
  forecast back to pixel space. Pixel-modality forecasters are
  evaluated directly.
- **FSS (Forecast Skill Score) @ τ = 20**: Murphy 1988 skill score
  against persistence,
  `FSS = 1 − rel_L2(forecaster) / rel_L2(persistence)`.
  `FSS > 0` means the forecaster beats naive persistence.

## CIL (Conditional Image Leakage)

Zhao NeurIPS 2024: at training time, the context latents fed to the
Transformer are perturbed by `z_t' = z_t + β_m · ε`, with `β_m = 0.5`
and `ε ∼ N(0, I)`. This prevents the network from over-relying on
exact-match context features during long-horizon direct-τ forecasting
and was the change that lifted the latent-Transformer from "worse than
persistence at τ = 20" to "best of the four".
