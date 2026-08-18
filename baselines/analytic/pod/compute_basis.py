"""Fit the POD basis on a stack of stride-4 training frames.

Each ``256^3`` training frame is flattened into a row of the data matrix
``X`` of shape ``(n_frames, 4 * 256^3)``. Randomised SVD gives the top
``K`` right-singular vectors, which become the POD basis used at
encode/decode time:

    z = U^T x          (K floats)
    x_hat = U z         (decompression)

For the ``K = 2048`` paper config this yields a compression ratio of
``(4 * 1024^3) / 2048 = 2.0e6`` per frame -- which is not the per-patch
``64x``; that's a property of POD on the whole frame. We report the
per-frame ratio for honesty in the table.

This script is a one-shot. The basis tensor lives at
``checkpoints/pod_basis_K{K}.npy``.
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np

try:
    from sklearn.decomposition import TruncatedSVD
except ImportError:  # pragma: no cover - scikit-learn is in requirements.txt
    TruncatedSVD = None


def _iter_train_frames(data_dir: str, max_frames: int):
    files = sorted(glob.glob(os.path.join(data_dir, "frame_*.h5")))[:max_frames]
    for fp in files:
        with h5py.File(fp, "r") as f:
            vel = f["velocity"][:].astype(np.float32)   # (3, N, N, N)
            prs = f["pressure"][:].astype(np.float32)   # (1, N, N, N)
        yield np.concatenate([vel, prs], axis=0)        # (4, N, N, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--stats", required=True,
                    help="norm_stats.json from compute_stats.py")
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--max_frames", type=int, default=200,
                    help="how many training frames to fit the basis on")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if TruncatedSVD is None:
        raise RuntimeError("scikit-learn is required for POD basis fitting")

    import json
    stats = json.load(open(args.stats))
    mean = np.array([stats[k]["mean"] for k in "uvwp"], dtype=np.float32).reshape(4, 1, 1, 1)
    std = np.array([stats[k]["std"] for k in "uvwp"], dtype=np.float32).reshape(4, 1, 1, 1)

    rows = []
    for frame in _iter_train_frames(args.data_dir, args.max_frames):
        frame = (frame - mean) / std
        rows.append(frame.reshape(-1).astype(np.float32))
    X = np.stack(rows, axis=0)                  # (n_frames, 4 * N^3)
    print(f"POD on data matrix of shape {X.shape}; K={args.K}")

    svd = TruncatedSVD(n_components=args.K, random_state=0)
    svd.fit(X)
    basis = svd.components_.astype(np.float32)  # (K, 4 * N^3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, basis)
    print(f"Saved basis -> {args.out}  ({basis.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
