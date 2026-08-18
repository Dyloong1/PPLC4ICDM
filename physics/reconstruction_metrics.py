"""Pixel-space reconstruction metrics shared across all methods.

The PSNR convention matches the eval driver: data are normalized to the
``[-1, 1]`` band by the per-channel z-score, so we use ``peak = 2.0``.
"""

from __future__ import annotations

import math


def rel_l1(recon, target) -> float:
    """``||recon - target||_1 / ||target||_1``."""
    num = float(abs(recon - target).sum())
    den = float(abs(target).sum()) + 1e-30
    return num / den


def rel_l2(recon, target) -> float:
    """``||recon - target||_2 / ||target||_2``."""
    diff = recon - target
    num = float((diff * diff).sum()) ** 0.5
    den = float((target * target).sum()) ** 0.5 + 1e-30
    return num / den


def mae(recon, target) -> float:
    """Mean absolute error (per voxel, channel-stacked)."""
    return float(abs(recon - target).mean())


def rmse(recon, target) -> float:
    """Root-mean-square error (per voxel, channel-stacked)."""
    diff = recon - target
    return float((diff * diff).mean()) ** 0.5


def psnr_db(rmse_normalized: float, peak: float = 2.0) -> float:
    """``20 * log10(peak / rmse)``.

    Args:
        rmse_normalized: RMSE measured in the normalized ``[-1, 1]`` band.
        peak: the peak-to-peak range of the normalized data; defaults to 2
            because data are z-scored to roughly ``[-1, 1]``.
    """
    return float(20.0 * math.log10(peak / max(rmse_normalized, 1e-30)))
