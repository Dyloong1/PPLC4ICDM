"""PPLC training loop.

Reads hyperparameters from a YAML config and a handful of path-style CLI
flags (data, checkpoint, optional resume). The loop:

1. Build train / val datasets from the JHTDB ``256^3`` memmap.
2. Build the PPLC generator (``spatial8`` or ``channel_heavy``) and the
   3D patch discriminator.
3. AdamW + linear-warmup + cosine schedule.
4. For each batch: G step (L1 + grad + KL + adv + optional consistency),
   then D step (hinge), with AMP + grad-clipping.
5. Track val L1 + per-channel rel-L2; save ``best.pt`` on improvement and
   stop early after ``patience`` non-improving epochs.

Optional WandB logging is **scalar metrics only** (no code / artifacts /
console capture).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import TurbulencePatchDataset, patch_collate
from .losses import (
    PatchDiscriminator3D,
    consistency_loss,
    gradient_l1,
    hinge_discriminator_loss,
    kl_divergence,
    l1_loss,
    pplc_generator_loss,
)
from .model import build_pplc


# ---------------------------------------------------------------------------
# CLI / config plumbing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a PPLC YAML config")
    p.add_argument("--data_dir", required=True,
                   help="directory containing dns_256_memmap.npy + norm_stats.json")
    p.add_argument("--save_dir", required=True, help="checkpoint output directory")
    p.add_argument("--memmap_name", default="dns_256_memmap.npy")
    p.add_argument("--stats_name", default="norm_stats.json")
    p.add_argument("--epochs", type=int, default=None, help="override config")
    p.add_argument("--patience", type=int, default=None, help="override config")
    p.add_argument("--resume", default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="pplc-public")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Train / val helpers
# ---------------------------------------------------------------------------

def build_loaders(args, cfg):
    memmap = os.path.join(args.data_dir, args.memmap_name)
    stats = os.path.join(args.data_dir, args.stats_name)
    train_idx = list(range(0, cfg.get("train_end", 700)))
    val_idx = list(range(cfg.get("train_end", 700), cfg.get("val_end", 800)))

    train_ds = TurbulencePatchDataset(
        memmap, train_idx, stats,
        patch_size=cfg["patch_size"],
        patches_per_load=cfg.get("patches_per_load", 8),
        augment=True,
    )
    val_ds = TurbulencePatchDataset(
        memmap, val_idx, stats,
        patch_size=cfg["patch_size"],
        patches_per_load=cfg.get("patches_per_load", 8),
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("frames_per_batch", 32), shuffle=True,
        num_workers=cfg.get("num_workers", 8), pin_memory=True, drop_last=True,
        persistent_workers=(cfg.get("num_workers", 8) > 0),
        collate_fn=patch_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("frames_per_batch", 32), shuffle=False,
        num_workers=cfg.get("num_workers", 8), pin_memory=True,
        persistent_workers=(cfg.get("num_workers", 8) > 0),
        collate_fn=patch_collate,
    )
    return train_loader, val_loader


@torch.no_grad()
def validate(model, val_loader, device, *, beta_kl, lambda_grad, amp_dtype):
    model.eval()
    sums = {"l1_recon": 0.0, "kl": 0.0, "gradient": 0.0}
    rel_l2 = torch.zeros(4)
    norm_sq = torch.zeros(4)
    n_batches = 0
    for data in val_loader:
        data = data.to(device, non_blocking=True)
        with autocast("cuda", dtype=amp_dtype):
            recon, mu, logvar = model(data)
        recon = recon.float(); mu = mu.float(); logvar = logvar.float()
        sums["l1_recon"] += float(l1_loss(recon, data).item())
        sums["kl"] += float(kl_divergence(mu, logvar).item())
        sums["gradient"] += float(gradient_l1(recon, data).item())
        for c in range(4):
            rel_l2[c] += (recon[:, c] - data[:, c]).pow(2).sum().item()
            norm_sq[c] += data[:, c].pow(2).sum().item()
        n_batches += 1
    avg = {k: v / max(1, n_batches) for k, v in sums.items()}
    rel_l2_err = torch.sqrt(rel_l2 / (norm_sq + 1e-12))
    model.train()
    return avg, rel_l2_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    epochs = args.epochs or cfg.get("epochs", 80)
    patience = args.patience or cfg.get("patience", 15)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "bfloat16": torch.bfloat16, "fp32": torch.float32}[
                     str(cfg.get("mixed_precision", "bf16")).lower()]
    use_scaler = amp_dtype == torch.float16

    train_loader, val_loader = build_loaders(args, cfg)

    model = build_pplc(
        arch=cfg.get("arch", "spatial8"),
        latent_channels=cfg["latent_channels"],
        in_channels=cfg.get("in_channels", 4),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    ).to(device)
    discriminator = PatchDiscriminator3D(
        in_channels=cfg.get("in_channels", 4),
        base_ch=cfg.get("d_base_ch", 64),
    ).to(device)
    n_g = sum(p.numel() for p in model.parameters()) / 1e6
    n_d = sum(p.numel() for p in discriminator.parameters()) / 1e6
    print(f"[model] G={n_g:.2f}M  D={n_d:.2f}M  arch={cfg.get('arch','spatial8')} "
          f"latent_channels={cfg['latent_channels']}")

    gen_lr = float(cfg.get("gen_lr", 1e-4))
    disc_lr = float(cfg.get("disc_lr", 5e-5))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=gen_lr,
                                  betas=betas, weight_decay=weight_decay)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=disc_lr,
                                       betas=(0.5, 0.999), weight_decay=0.0)

    warmup_epochs = cfg.get("warmup_epochs", min(10, epochs // 8))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=max(1, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    scaler = GradScaler("cuda", enabled=use_scaler)
    d_scaler = GradScaler("cuda", enabled=use_scaler)

    # Resume.
    start_epoch = 0
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        d_scaler.load_state_dict(ckpt["d_scaler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"[resume] from {args.resume} epoch {start_epoch}")

    # Optional WandB (scalar metrics only).
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, config=cfg,
                save_code=False, settings=wandb.Settings(
                    console="off", _save_requirements=False),
            )
        except Exception as e:
            print(f"[wandb] disabled: {e}")
    csv_path = Path(args.save_dir) / "training_log.csv"
    csv_fields = ["epoch", "train_l1", "val_l1", "rel_l2_u", "rel_l2_v",
                  "rel_l2_w", "rel_l2_p", "lr", "seconds"]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)

    beta_kl = float(cfg.get("beta_KL", 1e-2))
    lambda_grad = float(cfg.get("lambda_grad", 0.5))
    lambda_adv = float(cfg.get("lambda_adv", 1e-2))
    lambda_consist = float(cfg.get("lambda_consist", 0.0))
    consist_shift = int(cfg.get("consist_shift", 8))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    no_improve = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        sums = {"l1_recon": 0.0, "kl": 0.0, "gradient": 0.0,
                "g_adv": 0.0, "consist": 0.0, "d_loss": 0.0}
        n_batches = 0

        for data in train_loader:
            data = data.to(device, non_blocking=True)

            # G step.
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                recon, mu, logvar = model(data)
            recon = recon.float(); mu = mu.float(); logvar = logvar.float()
            g_loss, log, recon_det, target_det = pplc_generator_loss(
                recon, data, mu, logvar, discriminator,
                beta_kl=beta_kl, lambda_grad=lambda_grad,
                lambda_adv=lambda_adv,
            )
            if lambda_consist > 0:
                l_consist = consistency_loss(
                    model, data, shift=consist_shift, axis=-1,
                    amp_dtype=amp_dtype,
                )
                g_loss = g_loss + lambda_consist * l_consist
                log["consist"] = float(l_consist.item())
            if use_scaler:
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                g_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # D step.
            disc_optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                d_loss, _, _ = hinge_discriminator_loss(
                    discriminator, recon_det, target_det)
            if use_scaler:
                d_scaler.scale(d_loss).backward()
                d_scaler.unscale_(disc_optimizer)
                nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                d_scaler.step(disc_optimizer); d_scaler.update()
            else:
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                disc_optimizer.step()

            for k in sums:
                if k == "d_loss":
                    sums[k] += float(d_loss.item())
                elif k in log:
                    sums[k] += log[k]
            n_batches += 1

        scheduler.step()
        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_avg, rel_l2 = validate(
            model, val_loader, device,
            beta_kl=beta_kl, lambda_grad=lambda_grad,
            amp_dtype=amp_dtype,
        )
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{epoch + 1:4d}/{epochs}] "
            f"train_l1={avg['l1_recon']:.4f} val_l1={val_avg['l1_recon']:.4f} "
            f"grad={avg['gradient']:.4f} adv_g={avg['g_adv']:.3f} "
            f"adv_d={avg['d_loss']:.3f} consist={avg['consist']:.4f} "
            f"rel_l2=[{rel_l2[0]:.3f},{rel_l2[1]:.3f},{rel_l2[2]:.3f},{rel_l2[3]:.3f}] "
            f"lr={lr_now:.2e} {elapsed:.0f}s",
            flush=True,
        )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, avg["l1_recon"], val_avg["l1_recon"],
                float(rel_l2[0]), float(rel_l2[1]),
                float(rel_l2[2]), float(rel_l2[3]),
                lr_now, elapsed,
            ])
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/l1": avg["l1_recon"],
                "train/grad": avg["gradient"],
                "train/adv_g": avg["g_adv"],
                "train/adv_d": avg["d_loss"],
                "train/consist": avg["consist"],
                "val/l1": val_avg["l1_recon"],
                "val/rel_l2_u": float(rel_l2[0]),
                "val/rel_l2_v": float(rel_l2[1]),
                "val/rel_l2_w": float(rel_l2[2]),
                "val/rel_l2_p": float(rel_l2[3]),
                "lr": lr_now,
            })

        improved = val_avg["l1_recon"] < best_val - 1e-5
        ckpt_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "disc_optimizer": disc_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "d_scaler": d_scaler.state_dict(),
            "config": cfg,
            "best_val_loss": min(best_val, val_avg["l1_recon"]),
        }
        if improved:
            best_val = val_avg["l1_recon"]
            no_improve = 0
            torch.save(ckpt_state, os.path.join(args.save_dir, "best.pt"))
        else:
            no_improve += 1
        torch.save(ckpt_state, os.path.join(args.save_dir, "latest.pt"))
        if no_improve >= patience:
            print(f"[early-stop] no improvement for {patience} epochs")
            break

    print(f"[done] best val_l1 = {best_val:.6f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
