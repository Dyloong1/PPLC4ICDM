# DC-AE-3D

Deep-Compression Auto-Encoder (Chen et al., ICLR 2025) ported to 3D
4-channel. Pixel-unshuffle downsample + pixel-shuffle upsample stages
keep the decoder lightweight while preserving the latent shape.

## Architecture

* Width schedule ``(64, 128, 256)`` over 3 stages.
* Depth schedule ``(2, 2, 2)`` residual blocks per stage.
* Latent shape ``(latent_channels = 4, 8, 8, 8)`` for the 64x tier.

## Deviations

* No EMA decoder (the paper's quality-improvement trick that adds 10%
  inference cost). All other architectural choices match the published
  config for the smallest ``f4`` variant.
