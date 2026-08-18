"""Wavelet (db4, level 3) compression with hard-threshold sparsification.

Per-channel 3D ``wavedecn`` with Daubechies-4 at ``level=3``, magnitude-sort
all coefficients (across all sub-bands and channels), keep the top-K, set
the rest to zero, then ``waverecn`` back. K is chosen so the kept-count
matches the target compression ratio.

The byte representation stores the kept ``(value, index)`` pairs plus the
per-level / per-subband shape map, all packed into a numpy archive.
"""

from __future__ import annotations

import io

import numpy as np

try:
    import pywt
except ImportError as e:  # pragma: no cover
    raise RuntimeError("PyWavelets (pywt) is required for the wavelet baseline") from e


def _flatten_coeffs(coeffs):
    """``pywt.wavedecn`` returns nested dicts; flatten to a single 1D array.

    Returns ``(flat_values, shape_map)`` where ``shape_map`` records the
    list of ``(key, shape)`` tuples needed to rebuild the original
    nested structure later.
    """
    flat_parts = []
    shape_map = []
    # First entry is the coarse approximation array.
    flat_parts.append(coeffs[0].ravel())
    shape_map.append(("a", coeffs[0].shape))
    for level_dict in coeffs[1:]:
        for key, arr in level_dict.items():
            flat_parts.append(arr.ravel())
            shape_map.append((key, arr.shape))
    return np.concatenate(flat_parts), shape_map


def _unflatten_coeffs(flat, shape_map):
    coeffs = []
    cursor = 0
    # Re-build approximation.
    a_key, a_shape = shape_map[0]
    n = int(np.prod(a_shape))
    coeffs.append(flat[cursor:cursor + n].reshape(a_shape))
    cursor += n
    # Re-build per-level dicts (7 detail bands each for 3D).
    keys_per_level = 7
    n_levels = (len(shape_map) - 1) // keys_per_level
    for _ in range(n_levels):
        level_dict = {}
        for _ in range(keys_per_level):
            k, s = shape_map[1 + len(level_dict) + (_ * keys_per_level)]
            n = int(np.prod(s))
            level_dict[k] = flat[cursor:cursor + n].reshape(s)
            cursor += n
        coeffs.append(level_dict)
    return coeffs


def compress(field: np.ndarray, *, ratio: float = 64.0,
             wavelet: str = "db4", level: int = 3) -> bytes:
    """Hard-threshold a 3D wavelet decomposition to the target ratio."""
    if field.ndim != 4:
        raise ValueError(f"expected (C, D, H, W), got {field.shape}")
    C, D, H, W = field.shape
    total = C * D * H * W
    keep = int(round(total / ratio))

    per_channel_flat = []
    per_channel_shape_map = None
    for c in range(C):
        coeffs = pywt.wavedecn(field[c], wavelet=wavelet, level=level)
        flat, shape_map = _flatten_coeffs(coeffs)
        per_channel_flat.append(flat.astype(np.float32))
        if per_channel_shape_map is None:
            per_channel_shape_map = shape_map
    all_flat = np.concatenate(per_channel_flat)
    abs_sorted = np.argsort(-np.abs(all_flat), kind="stable")
    top_idx = abs_sorted[:keep]
    values = all_flat[top_idx].astype(np.float32)
    idx = top_idx.astype(np.int64)

    buf = io.BytesIO()
    np.savez(buf, values=values, idx=idx,
             channel_size=len(per_channel_flat[0]),
             shape_map=np.array(per_channel_shape_map, dtype=object))
    return buf.getvalue()


def decompress(blob: bytes, *, shape, wavelet: str = "db4",
               level: int = 3) -> np.ndarray:
    """Place the kept coefficients back into the full coefficient vector."""
    z = np.load(io.BytesIO(blob), allow_pickle=True)
    values = z["values"]; idx = z["idx"]
    channel_size = int(z["channel_size"])
    shape_map = list(z["shape_map"])

    C, D, H, W = shape
    full = np.zeros(C * channel_size, dtype=np.float32)
    full[idx] = values

    out = np.empty(shape, dtype=np.float32)
    for c in range(C):
        flat_c = full[c * channel_size:(c + 1) * channel_size]
        coeffs = _unflatten_coeffs(flat_c, shape_map)
        out[c] = pywt.waverecn(coeffs, wavelet=wavelet)[:D, :H, :W]
    return out


def compression_ratio(field_shape, ratio: float) -> float:
    return float(ratio)
