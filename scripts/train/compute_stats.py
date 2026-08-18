"""One-shot script to compute per-channel z-score statistics.

Iterates over the train-split frames and accumulates per-channel mean
and variance. Writes a JSON of the form

    {
        "u": {"mean": ..., "std": ...},
        "v": {"mean": ..., "std": ...},
        "w": {"mean": ..., "std": ...},
        "p": {"mean": ..., "std": ...}
    }

The training loop and the eval driver both read this file.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_frames", type=int, default=200,
                   help="how many frames to sample for the statistics")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "frame_*.h5")))[:args.max_frames]
    if not paths:
        raise SystemExit(f"no frames under {args.data_dir}")

    n_voxels = 0
    channel_sum = np.zeros(4, dtype=np.float64)
    channel_sum_sq = np.zeros(4, dtype=np.float64)
    for fp in paths:
        with h5py.File(fp, "r") as f:
            vel = f["velocity"][:].astype(np.float64)
            prs = f["pressure"][:].astype(np.float64)
        data = np.concatenate([vel, prs], axis=0)
        n = data[0].size
        for c in range(4):
            channel_sum[c] += data[c].sum()
            channel_sum_sq[c] += (data[c] ** 2).sum()
        n_voxels += n
    means = channel_sum / n_voxels
    stds = np.sqrt(channel_sum_sq / n_voxels - means ** 2)

    keys = ["u", "v", "w", "p"]
    out = {k: {"mean": float(means[i]), "std": float(stds[i])}
            for i, k in enumerate(keys)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} from {len(paths)} frames")
    for k in keys:
        print(f"  {k}: mean={out[k]['mean']:.6f}  std={out[k]['std']:.6f}")


if __name__ == "__main__":
    main()
