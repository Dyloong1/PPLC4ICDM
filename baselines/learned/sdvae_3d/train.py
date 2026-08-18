"""SD-VAE-3D training entry point.

Reads architecture knobs (``ch``, ``ch_mult``, ``num_res_blocks``,
``z_channels``, ``embed_dim``) from the YAML config and dispatches to
the shared training loop in :mod:`baselines.learned._common`.
"""

from __future__ import annotations

import torch.nn.functional as F

from .model import SDVAE3D
from .._common import base_parser, load_config, run_training_loop


def main():
    args = base_parser().parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ch_mult = tuple(int(s) for s in str(mcfg.get("ch_mult", "1,2,4")).split(","))

    model = SDVAE3D(
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("in_channels", 4),
        ch=mcfg.get("ch", 64),
        ch_mult=ch_mult,
        num_res_blocks=mcfg.get("num_res_blocks", 1),
        z_channels=mcfg.get("z_channels", 4),
        embed_dim=mcfg.get("embed_dim", 4),
    )

    def compute_recon(model, x):
        recon, mean, logvar, _ = model(x, sample_posterior=True)
        # KL = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return recon, kl

    run_training_loop(args, cfg, model,
                      compute_recon=compute_recon,
                      uses_kl=True, uses_disc=cfg.get("uses_disc", True))


if __name__ == "__main__":
    main()
