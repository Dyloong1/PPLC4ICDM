"""ZFP fixed-rate / fixed-accuracy compression (per channel).

We use the Python ``zfpy`` binding to call the C library. The 64x tier
in the paper is reached by tuning the ``rate`` parameter (bits per
voxel) -- empirically ``rate = 32 / ratio`` for ratio ``64x`` gives
roughly the target storage. For deterministic byte-counts, fixed-rate
mode is preferable to accuracy mode.
"""

from __future__ import annotations

import io

import numpy as np

try:
    import zfpy
except ImportError as e:  # pragma: no cover
    raise RuntimeError("zfpy is required for the ZFP baseline") from e


def compress(field: np.ndarray, *, ratio: float = 64.0) -> bytes:
    """Per-channel ZFP fixed-rate compression.

    Args:
        field: ``(C, D, H, W)`` float32 array.
        ratio: target compression ratio.

    Returns:
        a numpy ``.npz`` archive packing the per-channel ZFP byte streams.
    """
    if field.ndim != 4:
        raise ValueError(f"expected (C, D, H, W), got {field.shape}")
    rate = 32.0 / float(ratio)
    streams = []
    for c in range(field.shape[0]):
        streams.append(zfpy.compress_numpy(np.ascontiguousarray(field[c]),
                                            rate=rate))
    buf = io.BytesIO()
    np.savez(buf, **{f"c{i}": np.frombuffer(s, dtype=np.uint8)
                      for i, s in enumerate(streams)})
    return buf.getvalue()


def decompress(blob: bytes, *, shape) -> np.ndarray:
    z = np.load(io.BytesIO(blob))
    out = np.empty(shape, dtype=np.float32)
    for c in range(shape[0]):
        out[c] = zfpy.decompress_numpy(z[f"c{c}"].tobytes())
    return out


def compression_ratio(field_shape, ratio: float) -> float:
    return float(ratio)
