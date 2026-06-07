"""Download JHTDB ``isotropic1024coarse`` frames.

Reads an access token from ``~/.jhtdb_token`` (one line, plain text),
issues a request per frame, and saves the result as an HDF5 file at
``<out>/frame_<NNNNN>.h5`` with two datasets:

    velocity : float32 ``(3, 1024, 1024, 1024)``
    pressure : float32 ``(1, 1024, 1024, 1024)``

Skips frames whose target file already exists. The frame list is
specified as either explicit IDs (``--frames 800 900 1000``) or as a
single ``--frames 0-699`` range.

Requires ``pyJHTDB`` (https://github.com/idies/pyJHTDB).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import pyJHTDB
    from pyJHTDB import libJHTDB
except ImportError as e:  # pragma: no cover
    raise SystemExit("pyJHTDB is required; install via `pip install pyJHTDB`") from e


N = 1024
DATASET = "isotropic1024coarse"
DT = 0.002  # native sampling interval; one "frame" = step of 10 * dt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, nargs="+",
                   help="explicit ids (`800 900 1000`) or a range (`0-699`)")
    p.add_argument("--out", required=True)
    p.add_argument("--token_path", default="~/.jhtdb_token")
    return p.parse_args()


def _parse_frames(arg):
    out = []
    for s in arg:
        if "-" in s:
            a, b = s.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return out


def _load_token(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"JHTDB token file not found at {p}")
    return p.read_text().strip().splitlines()[0].strip()


def _read_frame(client: libJHTDB, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Fetch velocity (3 components) and pressure on the full N^3 grid."""
    coords = np.empty((N * N * N, 3), dtype=np.float32)
    grid = np.linspace(0.0, 2 * np.pi, N, endpoint=False, dtype=np.float32)
    Z, Y, X = np.meshgrid(grid, grid, grid, indexing="ij")
    coords[:, 0] = X.ravel(); coords[:, 1] = Y.ravel(); coords[:, 2] = Z.ravel()
    vel = client.getData(t, coords, sinterp=6, tinterp=0,
                          getFunction="getVelocity").reshape(N, N, N, 3)
    prs = client.getData(t, coords, sinterp=6, tinterp=0,
                          getFunction="getPressure").reshape(N, N, N, 1)
    velocity = np.moveaxis(vel, -1, 0).astype(np.float32)  # (3, N, N, N)
    pressure = np.moveaxis(prs, -1, 0).astype(np.float32)  # (1, N, N, N)
    return velocity, pressure


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _load_token(args.token_path)

    client = libJHTDB()
    client.initialize()
    client.add_token(token)

    frames = _parse_frames(args.frames)
    for fid in frames:
        out_path = out_dir / f"frame_{fid:05d}.h5"
        if out_path.exists():
            print(f"[skip] {out_path} exists")
            continue
        t_phys = float(fid * 10 * DT)
        try:
            velocity, pressure = _read_frame(client, t_phys)
        except Exception as e:
            print(f"[error] frame {fid}: {e}", file=sys.stderr)
            continue
        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("velocity", data=velocity, compression="gzip",
                              compression_opts=4)
            f.create_dataset("pressure", data=pressure, compression="gzip",
                              compression_opts=4)
            f.attrs["frame_id"] = fid
            f.attrs["time"] = t_phys
            f.attrs["dataset"] = DATASET
        os.replace(tmp_path, out_path)
        print(f"[done] {out_path}")

    client.finalize()


if __name__ == "__main__":
    main()
