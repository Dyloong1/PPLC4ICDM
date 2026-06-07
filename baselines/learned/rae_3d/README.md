# RAE-3D (Regularized Auto-Encoder)

Zheng et al. 2025, arXiv:2510.11690. RAE replaces a VAE's KL term with
an explicit decoder-side denoising objective: the encoder is **frozen**
(the paper recommends a pre-trained SD-VAE KL encoder), and the decoder
is trained from scratch to invert ``z_noisy = z + tau * eps`` back to
the input field.

## Architecture

* Frozen encoder: SD-VAE-3D with the paper-default ``embed_dim = 4``.
  Path to the checkpoint is set via ``model.frozen_encoder_ckpt`` in the
  YAML.
* Trained decoder: a 3D ResNet that maps ``(4, 8, 8, 8)`` latents back to
  ``(4, 32, 32, 32)`` patches.

## Deviations

* The original RAE paper uses a much larger ViT-XL decoder; we follow
  the smaller-decoder convention from the 3D ports of every other
  baseline (~10 M parameter decoder, ~30 M total system).
