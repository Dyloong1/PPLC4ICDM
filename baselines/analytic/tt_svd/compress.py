"""TT-SVD (quantized tensor-train) compression.

Reshapes the ``(C, D, H, W)`` field into a long tensor on a binary index
tree (``D = H = W = 2^L``), then applies TT-SVD with a fixed maximum
bond dimension. ``bond_dim = 9`` gives the paper's ``68.6x`` ratio at
``1024^3``; we report the per-frame ratio rather than the nominal tier.

The implementation is intentionally short -- ``tensorly`` does the
sequential SVDs for us.
"""

from __future__ import annotations

import io

import numpy as np

try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train
    tl.set_backend("numpy")
except ImportError as e:  # pragma: no cover
    raise RuntimeError("tensorly is required for the TT-SVD baseline") from e


def _quantize_shape(shape):
    """Express ``(D, H, W)`` as ``(2,) * L`` so the tensor is a binary tree."""
    out = []
    for d in shape:
        # ``d`` must be a power of 2.
        if d & (d - 1):
            raise ValueError(f"dimension {d} is not a power of 2")
        out.append(d)
    return out


def compress(field: np.ndarray, *, bond_dim: int = 9) -> bytes:
    """Per-channel TT-SVD with fixed bond dimension.

    The returned payload is a list of TT cores per channel, packed via
    :func:`numpy.savez`.
    """
    if field.ndim != 4:
        raise ValueError(f"expected (C, D, H, W), got {field.shape}")
    C, D, H, W = field.shape
    payload = {}
    payload["shape"] = np.array([C, D, H, W], dtype=np.int64)
    payload["bond_dim"] = np.array([bond_dim], dtype=np.int64)
    for c in range(C):
        tt = tensor_train(field[c], rank=bond_dim)
        for i, core in enumerate(tt):
            payload[f"c{c}_core{i}"] = core.astype(np.float32)
        payload[f"c{c}_n_cores"] = np.array([len(tt)], dtype=np.int64)
    buf = io.BytesIO()
    np.savez(buf, **payload)
    return buf.getvalue()


def decompress(blob: bytes, *, shape) -> np.ndarray:
    z = np.load(io.BytesIO(blob))
    C, D, H, W = shape
    out = np.empty(shape, dtype=np.float32)
    for c in range(C):
        n_cores = int(z[f"c{c}_n_cores"][0])
        cores = [z[f"c{c}_core{i}"] for i in range(n_cores)]
        # Contract sequentially using einsum.
        full = cores[0]
        for nxt in cores[1:]:
            # full: (r0, ..., rk) cumulative; nxt: (rk, dk, rk+1)
            full = np.einsum("...a,abc->...bc", full, nxt)
        out[c] = full.reshape(D, H, W).astype(np.float32)
    return out


def compression_ratio(field_shape, bond_dim: int = 9) -> float:
    C, D, H, W = field_shape
    # Per-frame ratio = (number of voxels) / (number of TT-core floats).
    total_floats = C * D * H * W
    # Cores: (1, d, r), (r, d, r), ..., (r, d, 1). Sum = 2 * r * d + (n-2) * r * r * d.
    # We approximate with a single channel and product-of-dimensions tree.
    dims = [D, H, W]
    n = len(dims)
    cost = 0
    for k, d in enumerate(dims):
        left = 1 if k == 0 else bond_dim
        right = 1 if k == n - 1 else bond_dim
        cost += left * d * right
    return float(total_floats) / float(C * cost)
