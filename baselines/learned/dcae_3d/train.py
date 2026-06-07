"""DC-AE-3D training entry point."""

from __future__ import annotations

import torch

from .model import DCAE3D
from .._common import base_parser, load_config, run_training_loop


def main():
    args = base_parser().parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]
    width = tuple(int(s) for s in str(mcfg.get("width", "64,128,256")).split(","))
    depth = tuple(int(s) for s in str(mcfg.get("depth", "2,2,2")).split(","))

    model = DCAE3D(
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("in_channels", 4),
        width=width,
        depth=depth,
        latent_channels=mcfg.get("latent_channels", 4),
    )

    def compute_recon(model, x):
        # DC-AE is a deterministic AE; no KL. Returns (recon,).
        out = model(x)
        recon = out[0] if isinstance(out, tuple) else out
        return recon, None

    run_training_loop(args, cfg, model,
                      compute_recon=compute_recon,
                      uses_kl=False, uses_disc=cfg.get("uses_disc", False))


if __name__ == "__main__":
    main()
