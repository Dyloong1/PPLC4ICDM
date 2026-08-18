# LTX-Video-VAE-3D

Hacohen et al. 2025, arXiv:2501.00103. Patchify-inside-VAE with
decoder noise augmentation; we evaluate the VAE in isolation (the
paper's downstream Diffusion Transformer is out of scope for the
compression-focused benchmark).

## Architecture

* Patchify-inside front-end (``2 x 2 x 2`` pixel-unshuffle) followed by
  the standard SD-VAE-style 3-stage ladder.
* Latent ``(z_channels = 4, 8, 8, 8)`` for the 64x tier.

## Deviations

* The paper's adversarial loss uses a "real pair"
  ``(noisy_decode, clean_decode)`` derived from the diffusion training
  procedure. We replace it with a same-channel temporal-neighbour pair
  ``(x_i, x_{i + 1})`` via ``torch.roll(x, 1, dim=0)``, which is the
  closest analogue that doesn't require training the DiT stack.
* The decoder noise augmentation ``sigma in [0, 0.2]`` is preserved per
  the original paper; in our compression-only setup it acts as a
  latent-space regulariser rather than as the DiT-output robustifier
  it was originally intended for.
