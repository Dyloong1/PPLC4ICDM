"""Figure 2 -- PPLC zero-shot vs native-1024 comparison.

A 2x2 panel:
    top-left   : DNS slice with a zoom inset.
    top-right  : the two PPLC zoom crops side-by-side
                 (zero-shot on the left, native-1024 on the right).
    bottom-left:  Diff (PPLC zero-shot - DNS) with the zoom inset.
    bottom-right: Diff (PPLC native-1024 - DNS) with the same inset.

Both diff panels share a single colour bar so the visual scale is
identical. Inputs are the same shipped ``slices_z512.npz`` + ``meta.json``
used by Figure 1.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _viz_font_setup  # noqa: F401  Serif font setup.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ZS_KEY = "u_Ours"
N1_KEY = "u_OursN1024"
ZS_LABEL = "PPLC (zero-shot)"
N1_LABEL = "PPLC (native 1024$^3$)"

INSET_LOC = "upper right"


def _add_zoom_overlay(fig, ax, axins, field, vmin, vmax, cmap,
                      zx, zy, zs):
    ax.add_patch(Rectangle((zx, zy), zs, zs, linewidth=1.6,
                           edgecolor="black", facecolor="none", zorder=5))
    axins.imshow(field[zy:zy + zs, zx:zx + zs], cmap=cmap, vmin=vmin, vmax=vmax,
                 origin="lower", interpolation="nearest")
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(1.8)
    box_corners = [(zx, zy + zs), (zx, zy)]
    inset_corners = [(0, zs), (0, 0)]
    for (bx, by), (ix, iy) in zip(box_corners, inset_corners):
        fig.add_artist(ConnectionPatch(
            xyA=(bx, by), xyB=(ix, iy),
            coordsA="data", coordsB="data",
            axesA=ax, axesB=axins,
            color="black", lw=0.85, alpha=0.6, zorder=4,
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--zoom", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta_path = args.meta or Path(args.slices).with_name("meta.json")
    zoom_path = args.zoom or Path(args.slices).with_name("zoom_region.npz")
    slices = np.load(args.slices)
    meta = json.load(open(meta_path))
    z = np.load(zoom_path)
    zx, zy, zs = int(z["x"]), int(z["y"]), int(z["size"])

    gt = slices["u_GT"].astype(np.float32)
    zs_recon = slices[ZS_KEY].astype(np.float32)
    n1_recon = slices[N1_KEY].astype(np.float32)
    zs_diff = zs_recon - gt
    n1_diff = n1_recon - gt

    vmax_main = float(np.percentile(np.abs(gt), 99))
    diff_vmax = max(float(np.percentile(np.abs(zs_diff), 99)),
                    float(np.percentile(np.abs(n1_diff), 99)))

    fig = plt.figure(figsize=(11.5, 11.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2,
                          left=0.04, right=0.93, top=0.95, bottom=0.03,
                          hspace=0.10, wspace=0.10)

    # Top-left: DNS full + zoom inset.
    ax_gt = fig.add_subplot(gs[0, 0])
    im_gt = ax_gt.imshow(gt, cmap="RdBu_r", vmin=-vmax_main, vmax=vmax_main,
                         origin="lower", interpolation="nearest", aspect="equal")
    axins_gt = inset_axes(ax_gt, width="40%", height="40%",
                          loc=INSET_LOC, borderpad=0.5)
    _add_zoom_overlay(fig, ax_gt, axins_gt, gt, -vmax_main, vmax_main, "RdBu_r",
                       zx, zy, zs)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_gt.set_title("DNS 1024$^3$", fontsize=22, color="black",
                     fontweight="bold", pad=8)

    # Top-right: two zoom crops side-by-side.
    ax_rc = fig.add_subplot(gs[0, 1])
    gap_w = 6
    composite = np.full((zs, zs * 2 + gap_w), np.nan, dtype=np.float32)
    composite[:, :zs] = zs_recon[zy:zy + zs, zx:zx + zs]
    composite[:, zs + gap_w:] = n1_recon[zy:zy + zs, zx:zx + zs]
    ax_rc.imshow(composite, cmap="RdBu_r", vmin=-vmax_main, vmax=vmax_main,
                  origin="lower", interpolation="nearest", aspect="equal")
    ax_rc.set_xticks([]); ax_rc.set_yticks([])
    ax_rc.text(zs / 2, zs + 4, ZS_LABEL, ha="center", va="bottom",
                fontsize=21, fontweight="bold", color="black",
                transform=ax_rc.transData)
    ax_rc.text(zs + gap_w + zs / 2, zs + 4, N1_LABEL, ha="center", va="bottom",
                fontsize=21, fontweight="bold", color="black",
                transform=ax_rc.transData)
    ax_rc.axvline(zs + gap_w / 2, color="white", linewidth=2.0, zorder=5)
    ax_rc.axvline(zs + gap_w / 2, color="black", linewidth=0.6, zorder=6)
    for s in ax_rc.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(1.5)

    # Bottom row: two diffs (shared scale).
    im_diffs = []
    diff_axes = []
    for label, diff_field, pos in (
        (ZS_LABEL, zs_diff, gs[1, 0]),
        (N1_LABEL, n1_diff, gs[1, 1]),
    ):
        ax = fig.add_subplot(pos)
        im = ax.imshow(diff_field, cmap="RdBu_r",
                        vmin=-diff_vmax, vmax=diff_vmax,
                        origin="lower", interpolation="nearest", aspect="equal")
        axins = inset_axes(ax, width="40%", height="40%",
                            loc=INSET_LOC, borderpad=0.5)
        _add_zoom_overlay(fig, ax, axins, diff_field, -diff_vmax, diff_vmax,
                           "RdBu_r", zx, zy, zs)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Diff: {label} - DNS 1024$^3$", fontsize=22,
                      color="black", fontweight="bold", pad=8)
        im_diffs.append(im)
        diff_axes.append(ax)

    fig.canvas.draw_idle()
    cbar_gap, cbar_w = 0.022, 0.013
    gt_pos = ax_gt.get_position()
    rc_pos = ax_rc.get_position()
    cax_top = fig.add_axes([rc_pos.x1 + cbar_gap,
                              gt_pos.y0 + 0.010, cbar_w,
                              gt_pos.height - 0.020])
    fig.colorbar(im_gt, cax=cax_top).ax.tick_params(labelsize=16)
    bd_pos = diff_axes[-1].get_position()
    cax_bot = fig.add_axes([bd_pos.x1 + cbar_gap,
                              bd_pos.y0 + 0.010, cbar_w,
                              bd_pos.height - 0.020])
    fig.colorbar(im_diffs[-1], cax=cax_bot).ax.tick_params(labelsize=16)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out} ({os.path.getsize(args.out) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
