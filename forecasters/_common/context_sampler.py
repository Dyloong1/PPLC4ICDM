"""Sample three-frame contexts ``(t-10, t-5, t)`` for the Transformer forecasters.

The context is a triplet of frames spaced ``[10, 5, 0]`` Delta-t steps
behind the prediction target. Frames are drawn from a uniform random
start ``t`` so each call returns a fresh ``(context_triplet, target)``
pair within the user-supplied frame range.
"""

from __future__ import annotations

import random

import numpy as np

CONTEXT_OFFSETS = (-10, -5, 0)


def sample_context_triplet(memmap, frame_range, *, tau: int = 10):
    """Return ``(context, target)`` as numpy float32 arrays.

    Args:
        memmap: a numpy memmap of shape ``(n_frames, C, D, H, W)``.
        frame_range: iterable of frame indices the sampler is allowed to use.
        tau: horizon -- the target frame is ``t + tau``.

    Returns:
        ``context`` of shape ``(3, C, D, H, W)`` and ``target`` of shape
        ``(C, D, H, W)``.
    """
    legal = [t for t in frame_range
             if t + tau < memmap.shape[0]
             and t + min(CONTEXT_OFFSETS) >= 0]
    if not legal:
        raise ValueError("no legal frame indices for the requested tau / range")
    t = random.choice(legal)
    ctx = np.stack([np.array(memmap[t + off]) for off in CONTEXT_OFFSETS], axis=0)
    tgt = np.array(memmap[t + tau])
    return ctx.astype(np.float32), tgt.astype(np.float32)
