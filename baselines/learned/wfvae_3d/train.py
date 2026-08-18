"""WF-VAE-3D training entry point."""

from __future__ import annotations

from .model import WFVAE3D
from .._common import base_parser, load_config, run_training_loop


def main():
    args = base_parser().parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    model = WFVAE3D(
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("in_channels", 4),
        ch=mcfg.get("ch", 128),
        num_res_blocks=mcfg.get("num_res_blocks", 2),
        z_channels=mcfg.get("z_channels", 4),
        embed_dim=mcfg.get("embed_dim", 4),
    )

    def compute_recon(model, x):
        recon, mean, logvar, _ = model(x, sample_posterior=True)
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return recon, kl

    run_training_loop(args, cfg, model,
                      compute_recon=compute_recon,
                      uses_kl=True, uses_disc=cfg.get("uses_disc", False))


if __name__ == "__main__":
    main()
