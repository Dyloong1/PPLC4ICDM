# WF-VAE-3D (Wavelet-Flow VAE)

Li et al., CVPR 2025 (arXiv:2411.17459). A KL-VAE with a Haar wavelet
front-end that gives the encoder an explicit multi-scale prior.

## Architecture

* Reversible level-1 Haar front-end (zero parameters).
* Encoder / decoder mirror of the SD-VAE-3D ladder but with two parallel
  paths -- a "main" path on the high-frequency sub-bands and an
  "energy-flow" path on the LLL low-pass band.
* Latent ``(z_channels = 4, 8, 8, 8)`` for the 64x tier.

## Deviations -- preserving the 64x ratio claim

The paper transmits the LLL low-pass band side-channel (so the encoded
representation is ``z`` + the LLL band, ~16 K extra floats per patch,
which inflates the per-frame compression ratio to ~7x). To preserve the
64x claim **without losing the architectural bias of the multi-scale
path**, we *synthesize* the energy-flow input from ``z`` via a small
learned upsample-and-conv. ``decode(z)`` takes only the latent and the
multi-scale prior is still expressed in the architecture.
