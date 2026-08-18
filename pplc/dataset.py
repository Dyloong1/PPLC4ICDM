"""Random ``32^3`` patch loader for PPLC training.

The loader assumes the JHTDB ``isotropic1024coarse`` frames have been
stride-4 down-sampled to ``256^3``, stacked into a single ``numpy`` memmap
of shape ``(n_frames, 4, 256, 256, 256)``, and that the per-channel
z-score statistics (mean / std for ``u, v, w, p``) are available as a
small JSON.

Strategy: each ``__getitem__`` loads one frame into RAM (~256 MB at
``256^3``) and emits ``patches_per_load`` random ``32^3`` patches. The
``DataLoader`` batches several such mini-batches into a full SGD batch
via :func:`patch_collate`. This amortizes the per-frame disk I/O across
many patches without holding multiple full frames in RAM.

Augmentations are physics-consistent: cubic-rotation + axis-aligned
reflection with the velocity channels permuted / sign-flipped accordingly,
plus a uniform random circular shift along each axis (legal under
periodic BCs).
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

_CHANNEL_AXIS = {3: 0, 2: 1, 1: 2}  # spatial axis -> velocity channel


def load_norm_stats(path: str):
    """Load ``{u,v,w,p: {mean, std}}`` from a JSON written by ``compute_stats``."""
    with open(path) as f:
        stats = json.load(f)
    keys = ["u", "v", "w", "p"]
    mean = np.array([stats[k]["mean"] for k in keys],
                    dtype=np.float32).reshape(4, 1, 1, 1)
    std = np.array([stats[k]["std"] for k in keys],
                   dtype=np.float32).reshape(4, 1, 1, 1)
    return mean, std


class TurbulencePatchDataset(Dataset):
    """One frame in -> ``patches_per_load`` random ``patch_size^3`` patches out."""

    def __init__(self, memmap_path: str, frame_indices,
                 norm_stats_path: str, *,
                 patch_size: int = 32,
                 patches_per_load: int = 64,
                 augment: bool = True):
        self.memmap_path = memmap_path
        self.frame_indices = list(frame_indices)
        self.patch_size = patch_size
        self.patches_per_load = patches_per_load
        self.augment = augment
        self.mean, self.std = load_norm_stats(norm_stats_path)
        self._mmap = None

    def _ensure_mmap(self):
        if self._mmap is None:
            self._mmap = np.load(self.memmap_path, mmap_mode="r")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self._ensure_mmap()
        frame_idx = self.frame_indices[idx]
        frame = np.array(self._mmap[frame_idx], dtype=np.float32)
        # In-place normalize; this is the load-time hot path. Allocating a
        # temporary fails on hosts with fragmented RAM after long training.
        frame -= self.mean
        frame /= self.std

        ps = self.patch_size
        n_patch = self.patches_per_load
        res = frame.shape[1]
        out = np.empty((n_patch, 4, ps, ps, ps), dtype=np.float32)
        for n in range(n_patch):
            z0 = random.randint(0, res - ps)
            y0 = random.randint(0, res - ps)
            x0 = random.randint(0, res - ps)
            patch = frame[:, z0:z0 + ps, y0:y0 + ps, x0:x0 + ps]
            if self.augment:
                patch = self._augment(patch.copy())
            out[n] = patch
        return torch.from_numpy(out)

    @staticmethod
    def _augment(data: np.ndarray) -> np.ndarray:
        """Random circular shift + cube rotation + axis-aligned reflection.

        Velocity channels are permuted / sign-flipped to stay physically
        consistent with the spatial transform.
        """
        # Random circular shift along each spatial axis.
        for axis in (1, 2, 3):
            shift = random.randint(0, data.shape[axis] - 1)
            if shift:
                data = np.roll(data, shift, axis=axis)

        # Random cube rotation (permutation of spatial axes 1, 2, 3).
        perm = list(np.random.permutation([1, 2, 3]))
        if perm != [1, 2, 3]:
            data_spatial = np.transpose(data, [0] + perm)
            chan_perm = [0, 1, 2, 3]
            for old_ax, new_ax in zip([1, 2, 3], perm):
                chan_perm[_CHANNEL_AXIS[old_ax]] = _CHANNEL_AXIS[new_ax]
            inv_chan = [0, 0, 0]
            for i in range(3):
                inv_chan[chan_perm[i]] = i
            data_new = np.empty_like(data_spatial)
            for new_c in range(3):
                data_new[new_c] = data_spatial[inv_chan[new_c]]
            data_new[3] = data_spatial[3]
            data = data_new

        # Random axis-aligned reflection.
        for axis in (1, 2, 3):
            if random.random() < 0.5:
                data = np.flip(data, axis=axis).copy()
                data[_CHANNEL_AXIS[axis]] = -data[_CHANNEL_AXIS[axis]]
        return data


def patch_collate(batch):
    """DataLoader collate: concatenate the per-frame ``(N, 4, ps, ps, ps)`` tensors."""
    return torch.cat(batch, dim=0)
