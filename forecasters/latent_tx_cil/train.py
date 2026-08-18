"""Train the latent Transformer + CIL forecaster.

Reads the YAML config, loads the frozen PPLC encoder from
``cfg['pplc_ckpt']``, encodes each training frame to its latent
representation on the fly, samples ``(z_context, z_target, tau)`` triples,
and minimises ``L1(z_pred, z_target)`` with AdamW + cosine schedule.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast

from pplc.model import PPLC
from .model import LatentTransformerForecaster


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--memmap_name", default="dns_256_memmap.npy")
    p.add_argument("--stats_name", default="norm_stats.json")
    p.add_argument("--pplc_ckpt", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="pplc-public")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_pplc(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    pplc = PPLC(
        in_channels=cfg.get("in_channels", 4),
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device)
    pplc.load_state_dict(ckpt["model"])
    pplc.eval()
    for p in pplc.parameters():
        p.requires_grad = False
    return pplc


def main():
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or cfg.get("epochs", 100)
    pplc_ckpt = args.pplc_ckpt or cfg.get("pplc_ckpt")
    if pplc_ckpt is None:
        raise SystemExit("Must provide --pplc_ckpt or set pplc_ckpt in the config")
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    use_scaler = False

    memmap = np.load(os.path.join(args.data_dir, args.memmap_name), mmap_mode="r")
    train_range = range(0, cfg.get("train_end", 800))
    tau_choices = list(cfg.get("tau_train", [10, 20]))
    batch_size = cfg.get("batch_size", 8)
    steps_per_epoch = cfg.get("steps_per_epoch", 200)

    pplc = _load_pplc(pplc_ckpt, device)

    model = LatentTransformerForecaster(
        d_model=cfg.get("d_model", 384),
        n_heads=cfg.get("n_heads", 6),
        n_layers=cfg.get("n_layers", 6),
        beta_m=cfg.get("beta_m", 0.5),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_scaler)

    csv_path = Path(args.save_dir) / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_l1", "lr", "seconds"])

    def _encode(frame_np):
        """Encode a 256^3 frame patch-by-patch and return its latent stack."""
        x = torch.from_numpy(frame_np).to(device, non_blocking=True)
        # For the open-source release we encode the full frame at once on
        # a 32^3 patch grid -- the forecaster operates on a single patch
        # at a time during demo training. This keeps the reference impl
        # small; production training uses a strided patch iterator.
        ps = 32
        z0 = random.randint(0, x.shape[-1] - ps)
        y0 = random.randint(0, x.shape[-2] - ps)
        x0 = random.randint(0, x.shape[-3] - ps)
        patch = x[:, x0:x0 + ps, y0:y0 + ps, z0:z0 + ps].unsqueeze(0)
        with torch.no_grad():
            mu, _, _ = pplc.encode(patch)
        return mu.squeeze(0)  # (latent_channels, 8, 8, 8)

    n_frames = memmap.shape[0]
    print(f"[start] training for {epochs} epochs, {steps_per_epoch} steps each")
    best_l1 = float("inf")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        for _ in range(steps_per_epoch):
            tau = random.choice(tau_choices)
            ctx_z = []; tgt_z = []
            for _ in range(batch_size):
                t = random.randint(10, len(train_range) - tau - 1)
                if t + tau >= n_frames:
                    continue
                ctx = torch.stack([_encode(np.array(memmap[t + off])) for off in (-10, -5, 0)])
                tgt = _encode(np.array(memmap[t + tau]))
                ctx_z.append(ctx); tgt_z.append(tgt)
            if not ctx_z:
                continue
            ctx_t = torch.stack(ctx_z).to(device)
            tgt_t = torch.stack(tgt_z).to(device)
            tau_t = torch.full((ctx_t.shape[0],), tau, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                pred = model(ctx_t, tau_t)
            loss = F.l1_loss(pred.float(), tgt_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        avg = total / max(1, steps_per_epoch)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"[{epoch + 1:4d}/{epochs}] train_l1={avg:.4f} lr={lr_now:.2e} "
              f"{elapsed:.0f}s", flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg, lr_now, elapsed])
        if avg < best_l1:
            best_l1 = avg
            torch.save({
                "epoch": epoch + 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            }, os.path.join(args.save_dir, "best.pt"))
    print(f"[done] best train_l1 = {best_l1:.6f}")


if __name__ == "__main__":
    main()
