"""POD compression: store ``z = U^T x``, decompress as ``x_hat = U z``.

The basis ``U`` (shape ``(K, 4 * D * H * W)``) is fitted off-line with
:mod:`compute_basis`. The compressed payload here is just the K-vector
of coefficients.
"""

from __future__ import annotations

import io

import numpy as np


def compress(field: np.ndarray, *, basis: np.ndarray) -> bytes:
    """Project the (normalized) field onto the POD basis.

    Args:
        field: ``(C, D, H, W)`` float32 array (typically normalized).
        basis: ``(K, C * D * H * W)`` float32 matrix produced by
            :func:`compute_basis.main`.

    Returns:
        ``z`` (K floats) packed as a ``.npy`` byte string.
    """
    x = field.reshape(-1).astype(np.float32)
    z = basis @ x  # (K,)
    buf = io.BytesIO()
    np.save(buf, z.astype(np.float32))
    return buf.getvalue()


def decompress(blob: bytes, *, basis: np.ndarray, shape) -> np.ndarray:
    z = np.load(io.BytesIO(blob))
    x_hat = basis.T @ z
    return x_hat.reshape(shape).astype(np.float32)


def compression_ratio(field_shape, K: int) -> float:
    C, D, H, W = field_shape
    return float(C * D * H * W) / float(K)
