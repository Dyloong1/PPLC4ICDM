"""Train the pixel-space U-Net AR forecaster."""

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

from .model import PixelUNetForecaster


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--memmap_name", default="dns_256_memmap.npy")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or cfg.get("epochs", 80)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memmap = np.load(os.path.join(args.data_dir, args.memmap_name), mmap_mode="r")
    train_end = cfg.get("train_end", 800)
    trained_tau = int(cfg.get("trained_tau", 10))
    batch_size = cfg.get("batch_size", 4)
    steps_per_epoch = cfg.get("steps_per_epoch", 150)

    model = PixelUNetForecaster(
        in_channels=cfg.get("in_channels", 4),
        ch=cfg.get("ch", 64),
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
            xs_b = []; ys_b = []
            for _ in range(batch_size):
                t = random.randint(0, train_end - trained_tau - 1)
                xs_b.append(crop(t)); ys_b.append(crop(t + trained_tau))
            xs = torch.stack(xs_b).to(device)
            ys = torch.stack(ys_b).to(device)
            tau_t = torch.full((xs.shape[0],), trained_tau, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
                pred = model(xs, tau_t)
            loss = F.l1_loss(pred.float(), ys)
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
