# Latent U-Net (autoregressive)

Tiny 3D U-Net forecaster on the frozen PPLC latent space. Trained at a
fixed short horizon ``trained_tau`` (default 10 frames). At inference
time the model is rolled autoregressively to reach the longer test
horizons (e.g. ``tau = 20``).

Strictly weaker than the headline ``latent_tx_cil`` Transformer; included
in Table 4 as a representative AR baseline.
