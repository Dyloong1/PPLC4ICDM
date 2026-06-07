"""Forecaster evaluation at 256^3, in-distribution.

Each forecaster predicts ``x_{t + tau}`` (pixel-modality) or
``z_{t + tau}`` (latent-modality, which is then decoded back to pixel
space through the frozen PPLC decoder). We compute the pixel-space
RMSE and the FSS-against-persistence at the user-supplied horizons.

The output is a single ``metrics_summary.json`` per forecaster, matching
the shape expected by ``scripts/tables/table4_forecaster.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--forecaster", required=True,
                   choices=["latent_tx_cil", "latent_unet_ar",
                            "pixel_tx_cil", "pixel_unet_ar"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pplc_ckpt", default=None,
                   help="required for the latent variants")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--memmap_name", default="dns_256_memmap.npy")
    p.add_argument("--tau_list", nargs="+", type=int, default=[1, 5, 10, 15, 20])
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load_pplc(ckpt_path: str, device):
    from pplc.model import PPLC
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    pplc = PPLC(latent_channels=cfg.get("latent_channels", 4)).to(device)
    pplc.load_state_dict(ckpt["model"]); pplc.eval()
    for p in pplc.parameters():
        p.requires_grad = False
    return pplc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memmap = np.load(os.path.join(args.data_dir, args.memmap_name), mmap_mode="r")

    if args.forecaster == "latent_tx_cil":
        from forecasters.latent_tx_cil.model import LatentTransformerForecaster
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = LatentTransformerForecaster(
            d_model=cfg.get("d_model", 384),
            n_heads=cfg.get("n_heads", 6),
            n_layers=cfg.get("n_layers", 6),
            beta_m=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        pplc = _load_pplc(args.pplc_ckpt, device)
        modality = "latent_tx"
    elif args.forecaster == "latent_unet_ar":
        from forecasters.latent_unet_ar.model import LatentUNetForecaster
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = LatentUNetForecaster(
            latent_channels=cfg.get("latent_channels", 4),
            ch=cfg.get("ch", 128),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        pplc = _load_pplc(args.pplc_ckpt, device)
        modality = "latent_unet"
    elif args.forecaster == "pixel_tx_cil":
        from forecasters.pixel_tx_cil.model import PixelTransformerForecaster
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = PixelTransformerForecaster(
            d_model=cfg.get("d_model", 384),
            n_heads=cfg.get("n_heads", 6),
            n_layers=cfg.get("n_layers", 6),
            beta_m=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        pplc = None
        modality = "pixel_tx"
    else:  # pixel_unet_ar
        from forecasters.pixel_unet_ar.model import PixelUNetForecaster
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = PixelUNetForecaster(
            in_channels=cfg.get("in_channels", 4),
            ch=cfg.get("ch", 64),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        pplc = None
        modality = "pixel_unet"

    test_start = cfg.get("test_start", 900)
    n_frames = memmap.shape[0]
    out = {"forecaster": args.forecaster,
           "tau_list": args.tau_list,
           "in_distribution_256": {}}

    for tau in args.tau_list:
        rmse_acc = []; rel_l2_acc = []; rel_l2_id_acc = []
        for _ in range(args.n_starts):
            t = random.randint(test_start, n_frames - tau - 1)
            x_t = torch.from_numpy(np.array(memmap[t]).astype(np.float32)).unsqueeze(0).to(device)
            x_target = torch.from_numpy(np.array(memmap[t + tau]).astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                if modality == "latent_tx":
                    ctx = torch.stack([
                        torch.from_numpy(np.array(memmap[t + off]).astype(np.float32))
                        for off in (-10, -5, 0)
                    ]).unsqueeze(0).to(device)
                    # Encode each context frame.
                    ctx_z = []
                    for i in range(3):
                        mu, _, _ = pplc.encode(ctx[:, i])
                        ctx_z.append(mu)
                    ctx_z = torch.stack(ctx_z, dim=1)
                    tau_t = torch.tensor([tau], device=device, dtype=torch.long)
                    z_pred = model(ctx_z, tau_t)
                    x_pred = pplc.decode(z_pred, mu_c=ctx[:, -1].mean(dim=(-3, -2, -1), keepdim=True))
                elif modality == "latent_unet":
                    mu, _, _ = pplc.encode(x_t)
                    tau_t = torch.tensor([tau], device=device, dtype=torch.long)
                    z_pred = model(mu, tau_t)
                    x_pred = pplc.decode(z_pred, mu_c=x_t.mean(dim=(-3, -2, -1), keepdim=True))
                elif modality == "pixel_tx":
                    ctx = torch.stack([
                        torch.from_numpy(np.array(memmap[t + off]).astype(np.float32))
                        for off in (-10, -5, 0)
                    ]).unsqueeze(0).to(device)
                    tau_t = torch.tensor([tau], device=device, dtype=torch.long)
                    x_pred = model(ctx, tau_t)
                else:  # pixel_unet
                    tau_t = torch.tensor([tau], device=device, dtype=torch.long)
                    x_pred = model(x_t, tau_t)

            diff = (x_pred - x_target).float()
            rmse = float(((diff * diff).mean()) ** 0.5)
            rl2 = float(((diff * diff).sum()) ** 0.5 / (((x_target * x_target).sum()) ** 0.5 + 1e-30))
            identity_diff = (x_t - x_target).float()
            rl2_id = float(((identity_diff * identity_diff).sum()) ** 0.5
                            / (((x_target * x_target).sum()) ** 0.5 + 1e-30))
            rmse_acc.append(rmse); rel_l2_acc.append(rl2); rel_l2_id_acc.append(rl2_id)

        out["in_distribution_256"][str(tau)] = {
            "rmse_pix_mean": float(np.mean(rmse_acc)),
            "rmse_pix_std": float(np.std(rmse_acc, ddof=1) if len(rmse_acc) > 1 else 0.0),
            "rel_l2_pix_mean": float(np.mean(rel_l2_acc)),
            "rel_l2_pix_identity_mean": float(np.mean(rel_l2_id_acc)),
            "n_starts": int(args.n_starts),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
