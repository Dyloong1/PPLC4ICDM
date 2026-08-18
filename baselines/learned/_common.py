"""Shared training loop for the learned baselines.

All six baselines are 3D ports of published image / video first-stage
auto-encoders. Their architectures live in each subfolder's ``model.py``;
their training loop is identical: per-channel L1 + (optionally) KL +
(optionally) a 3D patch discriminator + (optionally) a gradient L1
regulariser, AdamW + linear-warmup + cosine.

Each baseline's ``train.py`` is a thin entry point that:

    1. Parses ``--config``, ``--data_dir``, ``--save_dir`` (and optional
       ``--resume`` / ``--use_wandb`` flags).
    2. Instantiates the ``model.py`` class with the keys in the YAML
       under ``model:``.
    3. Calls :func:`run_training_loop` below.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from pplc.dataset import TurbulencePatchDataset, patch_collate
from pplc.losses import PatchDiscriminator3D, gradient_l1, hinge_discriminator_loss


def base_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--memmap_name", default="dns_256_memmap.npy")
    p.add_argument("--stats_name", default="norm_stats.json")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="pplc-public")
    return p


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
    nw = cfg.get("num_workers", 8)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("frames_per_batch", 16), shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0), collate_fn=patch_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("frames_per_batch", 16), shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0), collate_fn=patch_collate,
    )
    return train_loader, val_loader


def run_training_loop(args, cfg, model: nn.Module, *,
                      compute_recon: Callable,
                      uses_kl: bool = True,
                      uses_disc: bool = False):
    """Train ``model`` on the baseline-VAE recipe.

    Args:
        compute_recon: callable ``(model, x) -> (recon, kl_loss_or_None)``;
            keeps the per-baseline forward pass localized.
        uses_kl: if True, sums ``cfg['beta_KL'] * kl`` into the total loss.
        uses_disc: if True, also trains a 3D patch discriminator with
            ``cfg['lambda_adv']`` weight.
    """
    epochs = args.epochs or cfg.get("epochs", 60)
    patience = args.patience or cfg.get("patience", 15)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "bfloat16": torch.bfloat16, "fp32": torch.float32}[
                     str(cfg.get("mixed_precision", "bf16")).lower()]
    use_scaler = amp_dtype == torch.float16

    train_loader, val_loader = build_loaders(args, cfg)
    model = model.to(device)
    n_g = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] G={n_g:.2f}M  arch={cfg.get('arch', 'baseline')}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("gen_lr", 1e-4)),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    warmup_epochs = int(cfg.get("warmup_epochs", min(5, epochs // 8)))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=max(1, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    scaler = GradScaler("cuda", enabled=use_scaler)

    discriminator = None
    disc_optimizer = None
    d_scaler = None
    if uses_disc:
        discriminator = PatchDiscriminator3D(
            in_channels=cfg.get("in_channels", 4),
            base_ch=cfg.get("d_base_ch", 64),
        ).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(cfg.get("disc_lr", 5e-5)),
            betas=(0.5, 0.999), weight_decay=0.0,
        )
        d_scaler = GradScaler("cuda", enabled=use_scaler)

    beta_kl = float(cfg.get("beta_KL", 1e-6))
    lambda_grad = float(cfg.get("lambda_grad", 0.0))
    lambda_adv = float(cfg.get("lambda_adv", 0.0))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    csv_path = Path(args.save_dir) / "training_log.csv"
    csv_fields = ["epoch", "train_l1", "val_l1", "rel_l2_u", "rel_l2_v",
                  "rel_l2_w", "rel_l2_p", "lr", "seconds"]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)

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

    best_val = float("inf")
    no_improve = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if uses_disc and "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
            d_scaler.load_state_dict(ckpt["d_scaler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"[resume] from {args.resume} epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        sums = {"l1": 0.0, "kl": 0.0, "grad": 0.0, "adv_g": 0.0, "adv_d": 0.0}
        n_batches = 0
        for data in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                recon, kl = compute_recon(model, data)
            recon = recon.float()
            if kl is not None:
                kl = kl.float()
            l_recon = F.l1_loss(recon, data)
            l_grad = gradient_l1(recon, data) if lambda_grad > 0 else torch.zeros((), device=device)
            total = l_recon + lambda_grad * l_grad
            if uses_kl and kl is not None:
                total = total + beta_kl * kl
            if uses_disc:
                for p in discriminator.parameters():
                    p.requires_grad_(False)
                d_fake = discriminator(recon)
                for p in discriminator.parameters():
                    p.requires_grad_(True)
                l_adv = -d_fake.mean()
                total = total + lambda_adv * l_adv
                sums["adv_g"] += float(l_adv.item())
            if use_scaler:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if uses_disc:
                disc_optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", dtype=amp_dtype):
                    d_loss, _, _ = hinge_discriminator_loss(
                        discriminator, recon.detach(), data.detach())
                if use_scaler:
                    d_scaler.scale(d_loss).backward()
                    d_scaler.unscale_(disc_optimizer)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                    d_scaler.step(disc_optimizer); d_scaler.update()
                else:
                    d_loss.backward()
                    nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                    disc_optimizer.step()
                sums["adv_d"] += float(d_loss.item())
            sums["l1"] += float(l_recon.item())
            if uses_kl and kl is not None:
                sums["kl"] += float(kl.item())
            sums["grad"] += float(l_grad.item()) if lambda_grad > 0 else 0.0
            n_batches += 1
        scheduler.step()

        # Validation.
        model.eval()
        val_l1 = 0.0; val_n = 0
        rel_l2 = torch.zeros(4); norm_sq = torch.zeros(4)
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                with autocast("cuda", dtype=amp_dtype):
                    recon, _ = compute_recon(model, data)
                recon = recon.float()
                val_l1 += float(F.l1_loss(recon, data).item())
                for c in range(4):
                    rel_l2[c] += (recon[:, c] - data[:, c]).pow(2).sum().item()
                    norm_sq[c] += data[:, c].pow(2).sum().item()
                val_n += 1
        val_l1 = val_l1 / max(1, val_n)
        rel_l2_err = torch.sqrt(rel_l2 / (norm_sq + 1e-12))

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        print(
            f"[{epoch + 1:4d}/{epochs}] train_l1={avg['l1']:.4f} "
            f"val_l1={val_l1:.4f} grad={avg['grad']:.4f} "
            f"kl={avg['kl']:.4f} adv_g={avg['adv_g']:.3f} "
            f"adv_d={avg['adv_d']:.3f} "
            f"rel_l2=[{rel_l2_err[0]:.3f},{rel_l2_err[1]:.3f},"
            f"{rel_l2_err[2]:.3f},{rel_l2_err[3]:.3f}] "
            f"lr={lr_now:.2e} {elapsed:.0f}s",
            flush=True,
        )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, avg["l1"], val_l1,
                float(rel_l2_err[0]), float(rel_l2_err[1]),
                float(rel_l2_err[2]), float(rel_l2_err[3]),
                lr_now, elapsed,
            ])
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/l1": avg["l1"], "train/grad": avg["grad"],
                "train/kl": avg["kl"], "train/adv_g": avg["adv_g"],
                "train/adv_d": avg["adv_d"],
                "val/l1": val_l1,
                "lr": lr_now,
            })

        improved = val_l1 < best_val - 1e-5
        ckpt_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "best_val_loss": min(best_val, val_l1),
        }
        if uses_disc:
            ckpt_state.update({
                "discriminator": discriminator.state_dict(),
                "disc_optimizer": disc_optimizer.state_dict(),
                "d_scaler": d_scaler.state_dict(),
            })
        if improved:
            best_val = val_l1
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
