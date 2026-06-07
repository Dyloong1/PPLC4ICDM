"""Radial energy spectrum ``E(k)`` via spherical-shell summation.

For a periodic ``(2*pi)^3`` domain on an integer ``k``-grid, the
isotropic energy spectrum is

    E(k) = sum_{||q|| in [k - 1/2, k + 1/2)}  0.5 * |u_hat(q)|^2 .

The implementation uses ``torch.fft.rfftn`` so the half-space along
the last axis carries weight 2 (except for the DC and Nyquist bins),
which preserves Parseval's identity.

NOTE on the historic bug fix:
    Earlier versions of this routine divided each shell by the number
    of points in the shell ("shell average"). For a ``k^{-5/3}`` field
    that gives an extra factor of ``k^2`` from the spherical-shell
    Jacobian and adds ``-2`` to the fitted slope. The correct
    convention is a **shell SUM** (no normalization by point count).
"""

from __future__ import annotations

import numpy as np
import torch


def energy_spectrum(velocity: torch.Tensor, *,
                    return_numpy: bool = True):
    """Shell-summed isotropic energy spectrum.

    Args:
        velocity: tensor of shape ``(B, 3, N, N, N)`` (channels: u, v, w).
        return_numpy: if True, returns ``(k, E_k)`` as numpy arrays
            of shape ``(K,)`` and ``(B, K)`` respectively.

    Returns:
        ``(wavenumbers, E_k)``.
    """
    B, C, Z, H, W = velocity.shape
    assert C == 3
    device = velocity.device

    u_hat = torch.fft.rfftn(velocity[:, 0], dim=(-3, -2, -1))
    v_hat = torch.fft.rfftn(velocity[:, 1], dim=(-3, -2, -1))
    w_hat = torch.fft.rfftn(velocity[:, 2], dim=(-3, -2, -1))

    N_total = Z * H * W
    e_density = 0.5 * (u_hat.abs() ** 2 + v_hat.abs() ** 2 + w_hat.abs() ** 2) / (N_total ** 2)
    # rfft Hermitian-symmetry weight: 2 everywhere except DC + Nyquist on
    # the truncated last axis.
    weight = torch.ones_like(e_density)
    weight[:, :, :, 1:-1] = 2.0
    e_density = e_density * weight

    kz = torch.fft.fftfreq(Z, d=1.0 / Z, device=device)
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=device)
    kx = torch.fft.rfftfreq(W, d=1.0 / W, device=device)
    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
    k_mag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    k_max = max(Z, H, W) // 2
    wavenumbers = torch.arange(1, k_max + 1, device=device, dtype=torch.float32)
    E_k = torch.zeros(B, k_max, device=device)
    for i in range(k_max):
        kappa = float(i + 1)
        mask = (k_mag >= kappa - 0.5) & (k_mag < kappa + 0.5)
        for b in range(B):
            E_k[b, i] = e_density[b][mask].sum()

    if return_numpy:
        return wavenumbers.cpu().numpy(), E_k.cpu().numpy()
    return wavenumbers, E_k
