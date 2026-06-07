"""Map ``--method`` strings to a small registry record.

Each method registers:

    * ``family``       -- ``analytic`` / ``learned`` / ``ours``.
    * ``cache_subdir`` -- folder under ``--out_dir`` for the per-frame caches.
    * ``compress``     -- callable ``(field, ctx) -> bytes`` (analytic only).
    * ``decompress``   -- callable ``(bytes, ctx) -> field``.
    * ``load_model``   -- callable ``(ckpt_path, device) -> model`` (learned).
    * ``run_frame``    -- callable ``(model, field, ctx) -> recon`` for learned
                          methods (handles patch tiling internally).

The driver script ``zeroshot_1024.py`` consumes this registry; it never
imports baselines directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class MethodRecord:
    family: str
    cache_subdir: str
    requires_ckpt: bool
    requires_basis: bool = False
    compression_ratio_per_patch: Optional[float] = None


REGISTRY: dict[str, MethodRecord] = {
    "stride4":    MethodRecord("analytic", "stride4_trilinear_64x", False),
    "pod":        MethodRecord("analytic", "pod_K2048", False, requires_basis=True),
    "wavelet":    MethodRecord("analytic", "wavelet_db4_lev3_64x_honest", False),
    "zfp":        MethodRecord("analytic", "zfp_acc_64x", False),
    "tt_svd":     MethodRecord("analytic", "frozen_tt_bond9", False),
    "sdvae_3d":   MethodRecord("learned", "sdvae_ch64_z4_64x", True),
    "dcae_3d":    MethodRecord("learned", "dcae_w96-192-384_z4_64x_large", True),
    "rae_3d":     MethodRecord("learned", "rae_3d_z4_64x", True),
    "wfvae_3d":   MethodRecord("learned", "wfvae_3d_ch256_z4_64x_large", True),
    "cosmos_3d":  MethodRecord("learned", "cosmos_tokenizer_3d_ch512_z4_64x_large", True),
    "ltx_3d":     MethodRecord("learned", "ltx_video_3d_ch160_z4_64x_large", True),
    "pplc":       MethodRecord("ours", "ours_spatial8_latent4_64x", True),
    "pplc_native": MethodRecord("ours", "ours_native1024_spatial8_lc4_64x", True),
}


def get(method: str) -> MethodRecord:
    if method not in REGISTRY:
        raise KeyError(
            f"unknown method {method!r}; choose from {sorted(REGISTRY)}"
        )
    return REGISTRY[method]
