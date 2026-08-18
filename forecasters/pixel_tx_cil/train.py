"""Train the pixel-space Transformer + CIL forecaster."""

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
from torch.amp import autocast

from .model import PixelTransformerForecaster


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--memmap_name", default="dns_256_memmap.npy")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or cfg.get("epochs", 80)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memmap = np.load(os.path.join(args.data_dir, args.memmap_name), mmap_mode="r")
    train_end = cfg.get("train_end", 800)
    tau_choices = list(cfg.get("tau_train", [10, 20]))
    batch_size = cfg.get("batch_size", 4)
    steps_per_epoch = cfg.get("steps_per_epoch", 150)

    model = PixelTransformerForecaster(
        d_model=cfg.get("d_model", 384),
        n_heads=cfg.get("n_heads", 6),
        n_layers=cfg.get("n_layers", 6),
        beta_m=cfg.get("beta_m", 0.5),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    csv_path = Path(args.save_dir) / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_l1", "lr", "seconds"])

    def crop(t):
        frame = np.array(memmap[t]).astype(np.float32)
        ps = 32
        z0 = random.randint(0, frame.shape[-1] - ps)
        y0 = random.randint(0, frame.shape[-2] - ps)
        x0 = random.randint(0, frame.shape[-3] - ps)
        return torch.from_numpy(frame[:, x0:x0 + ps, y0:y0 + ps, z0:z0 + ps])

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        for _ in range(steps_per_epoch):
            tau = random.choice(tau_choices)
            ctx_b = []; tgt_b = []
            for _ in range(batch_size):
                t = random.randint(10, train_end - tau - 1)
                ctx_b.append(torch.stack([crop(t + off) for off in (-10, -5, 0)]))
                tgt_b.append(crop(t + tau))
            ctx_t = torch.stack(ctx_b).to(device)
            tgt_t = torch.stack(tgt_b).to(device)
            tau_t = torch.full((ctx_t.shape[0],), tau, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
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
        if avg < best:
            best = avg
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "config": cfg},
                       os.path.join(args.save_dir, "best.pt"))
    print(f"[done] best train_l1 = {best:.6f}")


if __name__ == "__main__":
    main()
