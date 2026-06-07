"""LTX-Video-VAE-3D training entry point.

We evaluate the VAE in isolation (no DiT downstream); the decoder noise
augmentation (``noise_sigma`` sampled uniformly in ``[0, 0.2]`` during
training) is preserved as a latent-space regulariser.
"""

from __future__ import annotations

from .model import LTXVideo3D
from .._common import base_parser, load_config, run_training_loop


def main():
    args = base_parser().parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    model = LTXVideo3D(
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("in_channels", 4),
        ch=mcfg.get("ch", 96),
        num_res_blocks=mcfg.get("num_res_blocks", 2),
        z_channels=mcfg.get("z_channels", 4),
        noise_sigma_max=mcfg.get("noise_sigma", 0.2),
    )

    def compute_recon(model, x):
        recon, _, _, _ = model(x, sample_posterior=True)
        return recon, None

    run_training_loop(args, cfg, model,
                      compute_recon=compute_recon,
                      uses_kl=False, uses_disc=cfg.get("uses_disc", False))


if __name__ == "__main__":
    main()
