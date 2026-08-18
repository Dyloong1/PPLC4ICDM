"""Figure 1 -- 4 row x N column baseline + error panel.

Rows (top to bottom):
    0: full ``z = 512`` slice for each method, with a red zoom box.
    1: zoom-in of that box.
    2: error map ``(recon - GT)`` for each non-DNS column, with the same box.
    3: zoom-in of the error map.

Reads the pre-extracted slices from ``assets/slices_z512.npz`` (~25 MB),
the column labelling / family colours from ``assets/meta.json``, and the
zoom-region anchor from ``assets/zoom_region.npz``. The figure is
ordered worst -> best -> DNS by the per-method slice ``rel_L2``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _viz_font_setup  # noqa: F401  Set serif font before any artist exists.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RED = "#d12"  # zoom-box + inset border colour


# In the open-source release the PPLC-native-1024 row is excluded from
# Fig 1 (it has its own dedicated comparison figure, Fig 2).
EXCLUDE = {"OursN1024"}


def _load_inputs(slices_path: str, meta_path: str | None,
                 zoom_path: str | None):
    slices = np.load(slices_path)
    meta = json.load(open(meta_path)) if meta_path else json.load(
        open(Path(slices_path).with_name("meta.json"))
    )
    if zoom_path is None:
        zoom_path = Path(slices_path).with_name("zoom_region.npz")
    z = np.load(zoom_path)
    zx, zy, zs = int(z["x"]), int(z["y"]), int(z["size"])
    return slices, meta, zx, zy, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True,
                    help="path to slices_z512.npz (e.g. assets/slices_z512.npz)")
    ap.add_argument("--meta", default=None,
                    help="path to meta.json; defaults to the file next to --slices")
    ap.add_argument("--zoom", default=None,
                    help="path to zoom_region.npz; defaults to the file next to --slices")
    ap.add_argument("--cache_dir", default=None,
                    help="unused (kept for CLI symmetry with the table builders)")
    ap.add_argument("--cell_size", type=float, default=2.6,
                    help="per-panel side length in inches")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slices, meta, zx, zy, zs = _load_inputs(args.slices, args.meta, args.zoom)
    gt = slices["u_GT"].astype(np.float32)
    vmax = float(np.percentile(np.abs(gt), 99))

    # Build the column order: per-method slice rel-L2, sorted worst -> best,
    # then DNS at the very end.
    errs = {}
    for key in meta["method_order"]:
        if key == "GT" or key in EXCLUDE:
            continue
        arr_key = f"u_{key}"
        if arr_key not in slices.files:
            continue
        u = slices[arr_key].astype(np.float32)
        errs[key] = float(np.linalg.norm(u - gt) / np.linalg.norm(gt))
    column_order = [k for k, _ in sorted(errs.items(), key=lambda kv: -kv[1])] + ["GT"]

    # Shared diff colour scale across the columns we actually draw.
    diff_vmax = max(
        float(np.percentile(np.abs(slices[f"u_{k}"].astype(np.float32) - gt), 99))
        for k in column_order if k != "GT"
    )

    n_cols, n_rows = len(column_order), 4
    cell = args.cell_size
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * cell + 0.4, n_rows * cell + 0.3),
                              constrained_layout=False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005,
                        hspace=0.015, wspace=0.015)

    box_lw = 3.0
    inset_lw = 3.0

    for col_i, key in enumerate(column_order):
        label = meta["labels"].get(key, key)
        err = errs.get(key)

        # Row 0: full slice + red zoom box.
        u = gt if key == "GT" else slices[f"u_{key}"].astype(np.float32)
        ax = axes[0, col_i]
        ax.imshow(u, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  origin="lower", interpolation="nearest", aspect="equal")
        ax.add_patch(Rectangle((zx, zy), zs, zs, linewidth=box_lw,
                               edgecolor=RED, facecolor="none", zorder=5))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#333"); s.set_linewidth(0.5)
        ax.set_title(label, fontsize=30, color="black",
                     fontweight="bold", pad=8)
        if err is not None:
            ax.text(0.035, 0.035, f"{err:.3f}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=21, color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28",
                              facecolor="black", edgecolor="none", alpha=0.80),
                    zorder=10)

        # Row 1: zoom-in of the original slice.
        ax = axes[1, col_i]
        ax.imshow(u[zy:zy + zs, zx:zx + zs],
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  origin="lower", interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(RED); s.set_linewidth(inset_lw)

        # Row 2: error map.
        ax = axes[2, col_i]
        if key == "GT":
            ax.set_xlim(0, gt.shape[1]); ax.set_ylim(0, gt.shape[0])
            ax.set_facecolor("white")
            ax.add_patch(Rectangle((zx, zy), zs, zs, linewidth=box_lw,
                                   edgecolor=RED, facecolor="none", zorder=5))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#333"); s.set_linewidth(0.5)
        else:
            d = u - gt
            ax.imshow(d, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax,
                      origin="lower", interpolation="nearest", aspect="equal")
            ax.add_patch(Rectangle((zx, zy), zs, zs, linewidth=box_lw,
                                   edgecolor=RED, facecolor="none", zorder=5))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#333"); s.set_linewidth(0.5)

        # Row 3: zoom-in of the error map.
        ax = axes[3, col_i]
        if key == "GT":
            ax.set_xlim(0, zs); ax.set_ylim(0, zs)
            ax.set_facecolor("white")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(RED); s.set_linewidth(inset_lw)
        else:
            ax.imshow(d[zy:zy + zs, zx:zx + zs],
                      cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax,
                      origin="lower", interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(RED); s.set_linewidth(inset_lw)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"Saved: {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
