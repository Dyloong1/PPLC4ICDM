"""Reassemble a full field from per-patch encode/decode calls.

Two schemes:

* :func:`reassemble_naive` — stride-``patch_size`` tiling. The frame is
  partitioned into non-overlapping ``32^3`` patches that the model
  encodes/decodes independently; the patches are concatenated back. This
  is the "fast" reassembly the paper's compute-cost row refers to.

* :func:`reassemble_hann` — stride-``patch_size / 2`` tiling with a 3D
  separable Hann window applied to each patch's reconstruction, then
  weighted overlap-add. Each voxel is covered by ``2^3 = 8`` patches.
  This is the "high-quality" reassembly the paper's reconstruction-error
  row refers to.

Both functions are GPU- and ``periodic-BC``-aware: they wrap indices
with ``%`` so the reassembler treats the input field as periodic on
``[0, D) x [0, H) x [0, W)`` (consistent with JHTDB ``isotropic1024coarse``).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.amp import autocast


def hann_window_3d(size: int, *, device="cpu",
                   dtype=torch.float32) -> torch.Tensor:
    """3D separable squared-sin window of side ``size`` (always > 0)."""
    i = torch.arange(size, device=device, dtype=dtype)
    w1 = torch.sin(math.pi * (i + 0.5) / size) ** 2
    return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def _run_model(model, x, amp_dtype):
    """Forward through a PPLC-style ``(recon, mu, logvar)`` model."""
    if amp_dtype is not None:
        with autocast("cuda", dtype=amp_dtype):
            recon, _, _ = model(x)
        return recon.float()
    recon, _, _ = model(x)
    return recon


@torch.inference_mode()
def reassemble_naive(model, frame: torch.Tensor, *,
                     patch_size: int = 32,
                     batch_patches: int = 32,
                     device: str = "cuda",
                     amp_dtype=torch.float16) -> torch.Tensor:
    """Non-overlapping tiling.

    Args:
        model: PPLC-style model with ``forward(x) -> (recon, mu, logvar)``.
        frame: ``(C, D, H, W)`` tensor on ``device`` or CPU. ``D, H, W`` must
            be divisible by ``patch_size``.
        patch_size: tile side length (default 32, matching the PPLC patch).
        batch_patches: how many patches to encode/decode per forward.
        device: where the model lives.
        amp_dtype: ``torch.float16`` / ``torch.bfloat16`` / ``None`` (FP32).

    Returns:
        Reconstruction with the same shape as ``frame``, on the same device
        as the input tensor.
    """
    C, D, H, W = frame.shape
    ps = patch_size
    assert D % ps == 0 and H % ps == 0 and W % ps == 0
    nz, ny, nx = D // ps, H // ps, W // ps
    n_total = nz * ny * nx

    out_device = frame.device
    frame_dev = frame.to(device, non_blocking=True)
    out = torch.empty_like(frame_dev)

    coords = [(iz, iy, ix) for iz in range(nz)
              for iy in range(ny) for ix in range(nx)]
    done = 0
    while done < n_total:
        end = min(done + batch_patches, n_total)
        batch = torch.empty(end - done, C, ps, ps, ps,
                            device=device, dtype=frame_dev.dtype)
        for bi, (iz, iy, ix) in enumerate(coords[done:end]):
            z, y, x = iz * ps, iy * ps, ix * ps
            batch[bi] = frame_dev[:, z:z + ps, y:y + ps, x:x + ps]
        recon = _run_model(model, batch, amp_dtype)
        for bi, (iz, iy, ix) in enumerate(coords[done:end]):
            z, y, x = iz * ps, iy * ps, ix * ps
            out[:, z:z + ps, y:y + ps, x:x + ps] = recon[bi].to(frame_dev.dtype)
        done = end

    return out.to(out_device)


@torch.inference_mode()
def reassemble_hann(model, frame: torch.Tensor, *,
                    patch_size: int = 32,
                    stride: int = 16,
                    batch_patches: int = 64,
                    device: str = "cuda",
                    amp_dtype=torch.float16,
                    verbose: bool = False) -> torch.Tensor:
    """Hann-windowed overlap-add tiling.

    With ``stride = patch_size // 2`` every voxel is covered by ``2^3``
    patches; the Hann window weighting acts as a smooth blend across patch
    boundaries.
    """
    C, D, H, W = frame.shape
    ps = patch_size
    out_device = frame.device

    frame_dev = frame.to(device, non_blocking=True)
    w_window = hann_window_3d(ps, device=device, dtype=torch.float32)

    recon_acc = torch.zeros(C, D, H, W, device=device, dtype=torch.float32)
    weight_acc = torch.zeros(D, H, W, device=device, dtype=torch.float32)

    base = torch.arange(ps, device=device)
    starts = list(range(0, D, stride))
    n_total = len(starts) ** 3
    if verbose:
        n_batches = (n_total + batch_patches - 1) // batch_patches
        print(f"  reassemble_hann: {len(starts)}^3 = {n_total} patches in "
              f"{n_batches} batches", flush=True)

    patch_buffer = torch.empty(batch_patches, C, ps, ps, ps,
                               device=device, dtype=frame_dev.dtype)
    pos_buffer: list[tuple[int, int, int]] = []
    t0 = time.time()
    last_print = t0

    def _flush():
        nb = len(pos_buffer)
        if nb == 0:
            return
        recon = _run_model(model, patch_buffer[:nb], amp_dtype)
        recon_w = recon * w_window[None, None]
        for k, (z0, y0, x0) in enumerate(pos_buffer):
            zi = (base + z0) % D
            yi = (base + y0) % H
            xi = (base + x0) % W
            recon_acc[:, zi[:, None, None], yi[None, :, None], xi[None, None, :]] += recon_w[k]
            weight_acc[zi[:, None, None], yi[None, :, None], xi[None, None, :]] += w_window
        pos_buffer.clear()

    done = 0
    for z0 in starts:
        zi = (base + z0) % D
        for y0 in starts:
            yi = (base + y0) % H
            for x0 in starts:
                xi = (base + x0) % W
                patch = frame_dev[:, zi[:, None, None],
                                  yi[None, :, None],
                                  xi[None, None, :]]
                patch_buffer[len(pos_buffer)] = patch
                pos_buffer.append((z0, y0, x0))
                done += 1
                if len(pos_buffer) == batch_patches:
                    _flush()
                    if verbose:
                        now = time.time()
                        if now - last_print > 30:
                            print(f"    patch {done}/{n_total} "
                                  f"({100 * done / n_total:.1f}%) elapsed "
                                  f"{now - t0:.0f}s", flush=True)
                            last_print = now
    _flush()

    out = (recon_acc / weight_acc[None]).to(frame_dev.dtype)
    return out.to(out_device)
