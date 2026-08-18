"""RAE-3D training entry point.

RAE wraps a frozen encoder (typically the SD-VAE-3D first stage) with a
trained decoder. The frozen encoder is loaded from
``cfg['frozen_encoder_ckpt']`` and put into ``eval()`` with
``requires_grad=False``.
"""

from __future__ import annotations

import torch

from .model import RAE3D
from .._common import base_parser, load_config, run_training_loop


def main():
    args = base_parser().parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    model = RAE3D(
        latent_ch=mcfg.get("latent_ch", 4),
        out_channels=cfg.get("in_channels", 4),
        base_ch=mcfg.get("base_ch", 256),
        noise_tau=mcfg.get("noise_tau", 0.0),
        frozen_encoder_ckpt=mcfg.get("frozen_encoder_ckpt", None),
    )

    def compute_recon(model, x):
        out = model(x)
        recon = out[0] if isinstance(out, tuple) else out
        return recon, None

    run_training_loop(args, cfg, model,
                      compute_recon=compute_recon,
                      uses_kl=False, uses_disc=cfg.get("uses_disc", False))


if __name__ == "__main__":
    main()
