# Latent Transformer + CIL (headline forecaster)

Direct-tau Transformer forecaster operating on the frozen PPLC latent
space. Headline of Table 4.

## Inputs and outputs

* Context : three frozen PPLC latents at frames ``(t-10, t-5, t)``.
* Horizon : a scalar ``tau`` -- the prediction target is ``z_{t+tau}``.
* Output  : a single predicted PPLC latent at horizon ``tau``.

## CIL (Conditional Image Leakage)

Zhao NeurIPS 2024. At training time we add Gaussian noise
``z_t' = z_t + beta_m * eps`` with ``beta_m = 0.5`` to the context
tokens. This prevents the Transformer from over-relying on exact-match
context features when forecasting long horizons; in our experiments it
lifted the latent Transformer from "worse than persistence at tau=20"
to "best of the four forecasters".

## Decode + pixel-space metrics

The forecaster predicts a latent. To compute the pixel-space RMSE / FSS
columns of Table 4 we decode the predicted latent back to pixel space
using the same PPLC decoder used during compressor eval.
