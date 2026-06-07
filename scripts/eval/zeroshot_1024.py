"""Single CLI for the 1024^3 zero-shot Table-1 eval.

Dispatches by ``--method``. For each frame ID it:

    1. Loads the ground-truth ``(C, D, H, W)`` field from
       ``<data_dir>/frame_<id>.h5`` (or ``.npy``).
    2. Compresses / decompresses through the chosen method.
    3. Computes the 14-field metric set (see ``physics`` and
       ``physics.reconstruction_metrics``) and writes the result to
       ``<out_dir>/<method_dir>/cache_frame_<id>.json``.

The cache-hit predicate is the strict schema match from the project's
contract: ``rel_l1_pix_1024`` + ``inference_time_sec`` + ``n_params``
must all be present.

For learned methods, the checkpoint location defaults to
``<ckpt_dir>/<method>.pt``. For the analytic ``pod`` baseline the
basis lives at ``<ckpt_dir>/pod_basis_K{K}.npy``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from physics import (
    NU,
    physics_on_gpu_streaming,
    rel_l1,
    rel_l2,
    mae as compute_mae,
    rmse as compute_rmse,
    psnr_db,
)
from . import _method_registry as registry


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------

def _read_frame(path: Path) -> np.ndarray:
    """Return a ``(4, D, H, W)`` float32 array."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    with h5py.File(path, "r") as f:
        vel = f["velocity"][:].astype(np.float32)
        prs = f["pressure"][:].astype(np.float32)
    return np.concatenate([vel, prs], axis=0)


def _frame_path(data_dir: Path, frame_id: int) -> Path:
    for name in (f"frame_{frame_id:05d}.h5",
                 f"frame_{frame_id:05d}.npy"):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no frame file for id {frame_id} in {data_dir}")


# ---------------------------------------------------------------------------
# Per-method dispatch
# ---------------------------------------------------------------------------

def _normalize(field: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (field - mean) / std


def _denormalize(field: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return field * std + mean


def _load_norm_stats(stats_path: Path):
    stats = json.load(open(stats_path))
    keys = ["u", "v", "w", "p"]
    mean = np.array([stats[k]["mean"] for k in keys],
                    dtype=np.float32).reshape(4, 1, 1, 1)
    std = np.array([stats[k]["std"] for k in keys],
                   dtype=np.float32).reshape(4, 1, 1, 1)
    return mean, std


def _run_analytic(method: str, field: np.ndarray, ratio: float, **knobs):
    """Returns ``(recon, payload_bytes_count)``."""
    if method == "stride4":
        from baselines.analytic.stride4.compress import compress, decompress
        blob = compress(field, stride=int(round(ratio ** (1 / 3))))
        recon = decompress(blob, shape=field.shape)
    elif method == "wavelet":
        from baselines.analytic.wavelet.compress import compress, decompress
        blob = compress(field, ratio=ratio)
        recon = decompress(blob, shape=field.shape)
    elif method == "zfp":
        from baselines.analytic.zfp.compress import compress, decompress
        blob = compress(field, ratio=ratio)
        recon = decompress(blob, shape=field.shape)
    elif method == "tt_svd":
        from baselines.analytic.tt_svd.compress import compress, decompress
        blob = compress(field, bond_dim=int(knobs.get("bond_dim", 9)))
        recon = decompress(blob, shape=field.shape)
    elif method == "pod":
        from baselines.analytic.pod.compress import compress, decompress
        basis = knobs["basis"]
        blob = compress(field, basis=basis)
        recon = decompress(blob, basis=basis, shape=field.shape)
    else:
        raise ValueError(f"unknown analytic method {method!r}")
    return recon, len(blob)


def _run_learned(method: str, field_norm_np: np.ndarray, model, *,
                 device, patch_size: int = 32, batch_patches: int = 32):
    """Patch-wise non-overlapping encode/decode."""
    from pplc.reassemble import reassemble_naive

    field_t = torch.from_numpy(field_norm_np)
    if method in ("sdvae_3d", "dcae_3d", "rae_3d", "wfvae_3d",
                  "cosmos_3d", "ltx_3d"):
        # Each baseline returns ``(recon, ...)``. Wrap with a tiny adapter
        # so reassemble_naive can call ``model(x) -> (recon, mu, logvar)``.
        class _Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__(); self.inner = inner

            def forward(self, x):
                out = self.inner(x) if method == "dcae_3d" else self.inner(x, sample_posterior=False)
                if isinstance(out, tuple):
                    return out[0], None, None
                return out, None, None

        wrapped = _Wrap(model).to(device).eval()
    else:  # pplc family already returns (recon, mu, logvar)
        wrapped = model

    recon = reassemble_naive(wrapped, field_t, patch_size=patch_size,
                              batch_patches=batch_patches, device=device)
    return recon.cpu().numpy()


# ---------------------------------------------------------------------------
# Per-frame eval
# ---------------------------------------------------------------------------

def _all_metrics(field_gt: np.ndarray, recon: np.ndarray, *,
                 mean: np.ndarray, std: np.ndarray,
                 inference_time_sec: float,
                 compression_ratio: float, n_params: int,
                 n_params_trainable: int, ckpt_size_mb: float,
                 backbone: str, frame_id: int,
                 device: torch.device):
    """Compute the 14-field locked schema dict."""
    # Pixel-space.
    rl1 = rel_l1(recon, field_gt)
    rl2 = rel_l2(recon, field_gt)
    mae_v = compute_mae(recon, field_gt)
    rmse_v = compute_rmse(recon, field_gt)
    # PSNR is reported in the normalized [-1, 1] band; rescale by std.
    rmse_norm = float(((recon - field_gt) / std).reshape(-1))
    rmse_norm = math.sqrt(np.mean(((recon - field_gt) / std) ** 2))
    psnr_v = psnr_db(rmse_norm)

    # Physics.
    def channel(c):
        return torch.from_numpy(field_gt[c].astype(np.float32)).to(device)
    phys_gt = physics_on_gpu_streaming(channel, N=field_gt.shape[-1])

    def channel_recon(c):
        return torch.from_numpy(recon[c].astype(np.float32)).to(device)
    phys_rc = physics_on_gpu_streaming(channel_recon, N=recon.shape[-1])

    eps_ratio = phys_rc["epsilon"] / phys_gt["epsilon"]
    omega_ratio = phys_rc["enstrophy"] / phys_gt["enstrophy"]
    rc_std = float(recon[0].std())
    rel_div = phys_rc["div_std"] / rc_std

    return {
        "frame_id": int(frame_id),
        "resolution": int(field_gt.shape[-1]),
        "backbone": backbone,
        "compression_ratio": float(compression_ratio),
        "n_params": int(n_params),
        "n_params_trainable": int(n_params_trainable),
        "ckpt_size_mb": float(ckpt_size_mb),
        "inference_time_sec": float(inference_time_sec),
        "eps_ratio": float(eps_ratio),
        "omega_ratio": float(omega_ratio),
        "slope_gt": float(phys_gt["inertial_slope"]),
        "slope_recon": float(phys_rc["inertial_slope"]),
        "rel_divergence": float(rel_div),
        "rel_l1_pix_1024": float(rl1),
        "rel_l2_pix_1024": float(rl2),
        "mae_pix_1024": float(mae_v),
        "rmse_pix_1024": float(rmse_v),
        "psnr_db_1024": float(psnr_v),
    }


def _cache_hit(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path) as f:
            row = json.load(f)
    except Exception:
        return False
    return (
        "rel_l1_pix_1024" in row
        and "inference_time_sec" in row
        and "n_params" in row
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   help=f"one of {sorted(registry.REGISTRY) + ['all']}")
    p.add_argument("--frames", nargs="+", type=int, default=[800, 900, 1000])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_dir", default="./checkpoints/")
    p.add_argument("--out_dir", default="./cache/zeroshot_1024/")
    p.add_argument("--stats", default=None,
                   help="optional override for the norm_stats.json path")
    p.add_argument("--ratio", type=float, default=64.0)
    p.add_argument("--batch_patches", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    methods = (sorted(registry.REGISTRY) if args.method == "all"
               else [args.method])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = Path(args.stats) if args.stats else Path(args.data_dir) / "norm_stats.json"
    mean, std = _load_norm_stats(stats_path)

    for method in methods:
        record = registry.get(method)
        out_dir = Path(args.out_dir) / record.cache_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load anything heavy (checkpoint, POD basis, etc.).
        model = None
        basis = None
        n_params = 0; n_params_trainable = 0; ckpt_size_mb = 0.0
        ckpt_path = None
        if record.requires_ckpt:
            ckpt_path = Path(args.ckpt_dir) / f"{method}.pt"
            if not ckpt_path.exists():
                print(f"[skip] {method}: ckpt missing at {ckpt_path}")
                continue
            ckpt_size_mb = float(ckpt_path.stat().st_size / (1024 * 1024))
            model = _load_model(method, ckpt_path, device)
            n_params = int(sum(p.numel() for p in model.parameters()))
            n_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if record.requires_basis:
            basis_path = Path(args.ckpt_dir) / "pod_basis_K2048.npy"
            if not basis_path.exists():
                print(f"[skip] {method}: basis missing at {basis_path}")
                continue
            basis = np.load(basis_path)

        for fid in args.frames:
            cache_path = out_dir / f"cache_frame_{fid:05d}.json"
            if _cache_hit(cache_path):
                print(f"[cache hit] {method} frame {fid}")
                continue
            frame_path = _frame_path(Path(args.data_dir), fid)
            field_gt = _read_frame(frame_path)
            field_norm = _normalize(field_gt, mean, std)

            t0 = time.time()
            if record.family == "analytic":
                if method == "pod":
                    recon_norm, _ = _run_analytic(method, field_norm,
                                                    args.ratio, basis=basis)
                else:
                    recon_norm, _ = _run_analytic(method, field_norm,
                                                    args.ratio)
            else:
                torch.cuda.synchronize() if device.type == "cuda" else None
                recon_norm = _run_learned(method, field_norm, model,
                                            device=device,
                                            batch_patches=args.batch_patches)
                torch.cuda.synchronize() if device.type == "cuda" else None
            inference_time = float(time.time() - t0)

            recon = _denormalize(recon_norm.astype(np.float32), mean, std)
            row = _all_metrics(
                field_gt, recon,
                mean=mean, std=std,
                inference_time_sec=inference_time,
                compression_ratio=args.ratio,
                n_params=n_params, n_params_trainable=n_params_trainable,
                ckpt_size_mb=ckpt_size_mb,
                backbone=method, frame_id=fid,
                device=device,
            )
            with open(cache_path, "w") as f:
                json.dump(row, f, indent=2)
            print(f"[done] {method} frame {fid}: rel_l1={row['rel_l1_pix_1024']:.4f}")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _load_model(method: str, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if method == "pplc" or method == "pplc_native":
        from pplc.model import build_pplc
        model = build_pplc(
            arch=cfg.get("arch", "spatial8"),
            latent_channels=cfg.get("latent_channels", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    if method == "sdvae_3d":
        from baselines.learned.sdvae_3d.model import SDVAE3D
        mcfg = cfg.get("model", {})
        ch_mult = tuple(int(s) for s in str(mcfg.get("ch_mult", "1,2,4")).split(","))
        model = SDVAE3D(
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("in_channels", 4),
            ch=mcfg.get("ch", 64), ch_mult=ch_mult,
            num_res_blocks=mcfg.get("num_res_blocks", 1),
            z_channels=mcfg.get("z_channels", 4),
            embed_dim=mcfg.get("embed_dim", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    if method == "dcae_3d":
        from baselines.learned.dcae_3d.model import DCAE3D
        mcfg = cfg.get("model", {})
        width = tuple(int(s) for s in str(mcfg.get("width", "64,128,256")).split(","))
        depth = tuple(int(s) for s in str(mcfg.get("depth", "2,2,2")).split(","))
        model = DCAE3D(
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("in_channels", 4),
            width=width, depth=depth,
            latent_channels=mcfg.get("latent_channels", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    if method == "rae_3d":
        from baselines.learned.rae_3d.model import RAE3D
        mcfg = cfg.get("model", {})
        model = RAE3D(
            latent_ch=mcfg.get("latent_ch", 4),
            out_channels=cfg.get("in_channels", 4),
            base_ch=mcfg.get("base_ch", 256),
            noise_tau=mcfg.get("noise_tau", 0.0),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    if method == "wfvae_3d":
        from baselines.learned.wfvae_3d.model import WFVAE3D
        mcfg = cfg.get("model", {})
        model = WFVAE3D(
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("in_channels", 4),
            ch=mcfg.get("ch", 128),
            num_res_blocks=mcfg.get("num_res_blocks", 2),
            z_channels=mcfg.get("z_channels", 4),
            embed_dim=mcfg.get("embed_dim", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    if method == "cosmos_3d":
        from baselines.learned.cosmos_3d.model import CosmosTokenizer3D
        mcfg = cfg.get("model", {})
        model = CosmosTokenizer3D(
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("in_channels", 4),
            ch=mcfg.get("ch", 96),
            num_res_blocks=mcfg.get("num_res_blocks", 2),
            z_channels=mcfg.get("z_channels", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    if method == "ltx_3d":
        from baselines.learned.ltx_3d.model import LTXVideo3D
        mcfg = cfg.get("model", {})
        model = LTXVideo3D(
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("in_channels", 4),
            ch=mcfg.get("ch", 96),
            num_res_blocks=mcfg.get("num_res_blocks", 2),
            z_channels=mcfg.get("z_channels", 4),
            noise_sigma_max=mcfg.get("noise_sigma", 0.2),
        ).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        return model
    raise ValueError(f"unsupported learned method {method!r}")


if __name__ == "__main__":
    main()
