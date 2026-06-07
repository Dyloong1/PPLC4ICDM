# SD-VAE-3D

Stable-Diffusion-style KL-VAE first stage (Rombach et al., CVPR 2022,
arXiv:2112.10752) ported from 2D RGB to 3D 4-channel.

## Architecture

* Encoder: 3-stage ladder with channels ``ch * [1, 2, 4]`` and
  ``num_res_blocks`` residual blocks per stage; mid-block has a single
  attention layer.
* Decoder: symmetric mirror.
* Latent: ``embed_dim = 4`` (paper-default f4 KL), spatial shape
  ``(4, 8, 8, 8)`` for a ``32^3`` patch -> 64x ratio.

## Deviations from the original paper

* No LPIPS perceptual loss (no perceptual model exists for 3D
  turbulence). The L1 + KL + hinge-PatchGAN combination is preserved.
* Channel width is ``ch = 64`` (vs ``ch = 128`` in the 2D RGB paper) so
  the parameter count matches the other 3D baselines within an order of
  magnitude.
