"""Figure 4 -- two-panel Pareto plot.

Panel A is a zoomed-in view of the high-quality cluster (most learned
baselines + PPLC); Panel B is the full overview that includes the
analytic methods and the null floor.

Coordinates:
    x = rel-L2 reconstruction error (lower is better)
    y = |1 - eps_ratio|             (lower is better)
    size = log10(inference time)
    colour / marker = method family (analytic / learned / ours / null)

Inputs:
    * ``assets/metrics.npz`` -- aggregated rel-L2 / eps / inference time
      per method (shipped at < 5 KB);
    * ``assets/meta.json``   -- label and family / colour mapping;
    * the ``--cache_dir`` flag is accepted for CLI symmetry but ignored
      here (the metrics NPZ already aggregates across the 3 eval frames).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _viz_font_setup  # noqa: F401  Serif font.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None,
                    help="path to metrics.npz (defaults to assets/metrics.npz "
                         "next to slices_z512.npz)")
    ap.add_argument("--meta", default=None,
                    help="path to meta.json (defaults to assets/meta.json)")
    ap.add_argument("--cache_dir", default=None,
                    help="unused -- kept for CLI symmetry with the other figures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    metrics_path = args.metrics or repo / "assets" / "metrics.npz"
    meta_path = args.meta or repo / "assets" / "meta.json"
    m = np.load(metrics_path)
    meta = json.load(open(meta_path))

    methods = list(m["methods"])
    labels = list(m["labels"])
    fams = list(m["families"])
    rel_l2 = m["rel_l2"]; eps = m["eps"]; inf = m["infer"]
    eps_dist = np.abs(1.0 - eps)

    fc = meta["family_color"]; fm = meta["family_marker"]
    log_inf = np.log10(inf)
    span = max(1e-6, log_inf.max() - log_inf.min())

    def _size(t):
        return 80 + 700 * (np.log10(t) - log_inf.min()) / span

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(14, 6.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    panels = [
        (axA, "A -- zoomed view (paper-tier methods)", (0.03, 0.10), (0, 0.30)),
        (axB, "B -- full overview (all methods)",      (0, 0.15), (0, 1.1)),
    ]
    for ax, title, xlim, ylim in panels:
        for i, _ in enumerate(methods):
            fam = fams[i]
            c, mk = fc[fam], fm[fam]
            ax.scatter([rel_l2[i]], [eps_dist[i]], s=_size(inf[i]),
                        c=c, marker=mk, alpha=0.80,
                        edgecolors="black", linewidths=1.4, zorder=3)
            ax.annotate(f"{i + 1}", xy=(rel_l2[i], eps_dist[i]),
                         ha="center", va="center", fontsize=8,
                         fontweight="bold",
                         color="white" if fam == "learned" else "black",
                         zorder=4)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("rel-L2 (reconstruction error, lower better)", fontsize=11)
        ax.set_ylabel(r"|1 $-$ $\varepsilon$ ratio| (lower better)",
                      fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.02, r"ideal" + "\n" + r"($\varepsilon$ $\approx$ 1)",
                transform=ax.transAxes, fontsize=9.5, color="#2c3e50",
                ha="left", va="bottom", style="italic")

    handles = [
        Line2D([0], [0], marker="X", color="w",
               markerfacecolor=fc["null"], markeredgecolor="black",
               markersize=11, label="null floor"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=fc["analytic"], markeredgecolor="black",
               markersize=11, label="analytic"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=fc["learned"], markeredgecolor="black",
               markersize=11, label="learned baselines"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=fc["ours"], markeredgecolor="black",
               markersize=15, label="PPLC (zero-shot)"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=fc["ours-1024"], markeredgecolor="black",
               markersize=11, label=r"PPLC (native 1024$^3$)"),
    ]
    axA.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    key_text = "Method index:  "
    for i, lbl in enumerate(labels):
        key_text += f"{i + 1}. {lbl}    "
        if (i + 1) % 5 == 0:
            key_text += "\n  "
    fig.text(0.02, -0.02, key_text, fontsize=8.5, family="monospace")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
