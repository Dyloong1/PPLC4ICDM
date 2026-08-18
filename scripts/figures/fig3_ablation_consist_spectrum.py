"""Figure 3 -- consistency-loss ablation, radial energy spectrum.

Three log-log curves:
    * DNS 1024^3                     (black solid)
    * PPLC without the consistency term (red dashed)
    * PPLC with the consistency term + Hann reassembly (blue solid)

Inputs are the per-frame cache JSONs under
``assets/cached_eval_jsons/ablation/{spatial,spatial_consist_hann}/cache_frame_00800_with_spectra.json``,
which retain the radial spectrum arrays ``wavenumbers`` /
``E_k_gt`` / ``E_k_recon`` (these were stripped from the table-1 cache
JSONs because the table builders don't need them).
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _viz_font_setup  # noqa: F401  Serif font.

import matplotlib.pyplot as plt
import numpy as np


def _load_spectra(cache_path: str):
    with open(cache_path) as f:
        d = json.load(f)
    k = np.asarray(d["wavenumbers"], dtype=np.float64)
    egt = np.asarray(d["E_k_gt"], dtype=np.float64)
    ere = np.asarray(d["E_k_recon"], dtype=np.float64)
    return k, egt, ere


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="path to assets/cached_eval_jsons/ablation/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    no_consist = os.path.join(
        args.cache_dir, "spatial", "cache_frame_00800_with_spectra.json"
    )
    pplc = os.path.join(
        args.cache_dir, "spatial_consist_hann",
        "cache_frame_00800_with_spectra.json",
    )

    k_nc, egt_nc, ere_nc = _load_spectra(no_consist)
    k_pl, egt_pl, ere_pl = _load_spectra(pplc)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    mask = k_pl >= 1
    ax.loglog(k_pl[mask], egt_pl[mask], color="black",
              linewidth=2.6, linestyle="-",
              label="DNS 1024$^3$", zorder=5)
    ax.loglog(k_nc[mask], ere_nc[mask], color="#c0392b",
              linewidth=2.0, linestyle="--",
              label="w/o consistency", alpha=0.95, zorder=3)
    ax.loglog(k_pl[mask], ere_pl[mask], color="#1e74c4",
              linewidth=2.2, linestyle="-",
              label="w/ consistency", alpha=0.95, zorder=4)

    ax.set_xlabel(r"wavenumber $k$", fontsize=20)
    ax.set_ylabel(r"$E(k)$", fontsize=20)
    ax.set_xlim(10, 520)
    ax.set_ylim(1e-8, 3e-2)
    ax.grid(True, which="both", alpha=0.30)
    ax.tick_params(labelsize=16)
    ax.axvspan(220, 510, color="#fbeaea", alpha=0.55, zorder=1)
    ax.legend(loc="lower left", fontsize=17, framealpha=0.95, ncol=1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
