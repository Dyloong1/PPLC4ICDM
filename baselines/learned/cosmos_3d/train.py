"""Cosmos-CV-3D training entry point.

Cosmos is a pure auto-encoder (no KL, no commitment), trained with L1 +
optional gradient L1. The model returns a 4-tuple for interface parity
with the VAE baselines; the ``logvar`` slot is always zeros.
"""

from __future__ import annotations

from .model import CosmosTokenizer3D
from .._common import base_parser, load_config, run_training_loop


def main():
    args = base_parser().parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    model = CosmosTokenizer3D(
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("in_channels", 4),
        ch=mcfg.get("ch", 96),
        num_res_blocks=mcfg.get("num_res_blocks", 2),
        z_channels=mcfg.get("z_channels", 4),
    )

    def compute_recon(model, x):
        recon, _, _, _ = model(x, sample_posterior=False)
        return recon, None

    run_training_loop(args, cfg, model,
                      compute_recon=compute_recon,
                      uses_kl=False, uses_disc=cfg.get("uses_disc", False))


if __name__ == "__main__":
    main()
