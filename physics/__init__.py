"""Physics metrics shared across all 13 methods (Section 4)."""

from .metrics import (
    NU,
    epsilon_strain,
    enstrophy,
    inertial_slope,
    relative_divergence,
    physics_on_gpu_streaming,
)
from .spectra import energy_spectrum
from .reconstruction_metrics import rel_l1, rel_l2, mae, rmse, psnr_db

__all__ = [
    "NU",
    "epsilon_strain",
    "enstrophy",
    "inertial_slope",
    "relative_divergence",
    "physics_on_gpu_streaming",
    "energy_spectrum",
    "rel_l1",
    "rel_l2",
    "mae",
    "rmse",
    "psnr_db",
]
