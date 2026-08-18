"""Stride-4 box average + trilinear up-sample.

The null compressor: average each ``stride^3`` block of voxels and store
the down-sampled field; decode by trilinear up-sample. For ``stride=4``
the ratio is exactly ``64x`` (``4^3``).

Stored representation is a single ``float32`` tensor of shape
``(C, D/stride, H/stride, W/stride)``. We serialize with ``numpy.save``
to a ``BytesIO`` for the byte-string interface.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F


def compress(field: np.ndarray, *, stride: int = 4) -> bytes:
    """Down-sample ``field`` by ``stride^3`` block-averaging.

    Args:
        field: ``(C, D, H, W)`` float32 array.
        stride: block side; must divide each spatial dimension.

    Returns:
        a byte string containing the down-sampled field as ``.npy``.
    """
    if field.ndim != 4:
        raise ValueError(f"expected (C, D, H, W), got {field.shape}")
    C, D, H, W = field.shape
    if D % stride or H % stride or W % stride:
        raise ValueError("spatial dims must be divisible by stride")
    x = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))
    x = x.unsqueeze(0)  # (1, C, D, H, W)
    down = F.avg_pool3d(x, kernel_size=stride, stride=stride)
    buf = io.BytesIO()
    np.save(buf, down.squeeze(0).numpy().astype(np.float32))
    return buf.getvalue()


def decompress(blob: bytes, *, shape) -> np.ndarray:
    """Trilinear-upsample the stored down-sampled field back to ``shape``."""
    down = np.load(io.BytesIO(blob))
    if down.ndim != 4:
        raise ValueError(f"expected (C, d, h, w), got {down.shape}")
    C, D, H, W = shape
    x = torch.from_numpy(down).unsqueeze(0)
    up = F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False)
    return up.squeeze(0).numpy().astype(np.float32)


def compression_ratio(field_shape, stride: int = 4) -> float:
    C, D, H, W = field_shape
    return float(stride ** 3)
