# Cosmos-CV-3D

NVIDIA Cosmos Tokenizer CV (NVIDIA 2025, arXiv:2501.03575) ported to 3D
4-channel. Pure auto-encoder (no KL, no commitment loss) with
factorized 3D convolutions and spatial-only LayerNorm.

## Architecture

* Encoder: factorized 3D conv stages with the paper's channel schedule
  ``ch * [1, 2, 4]`` and ``num_res_blocks`` per stage.
* Decoder: symmetric mirror with factorized transpose convolutions.
* Latent ``(z_channels = 4, 8, 8, 8)`` for the 64x tier.

## Deviations

* The paper specifies AdamW with ``betas=(0.9, 0.99)``; we preserve that
  unusual ``beta_2``. All other hyperparameters follow the paper.
* The original Cosmos paper trains the tokenizer jointly with a
  text-to-video diffusion model; we evaluate the tokenizer in isolation.
