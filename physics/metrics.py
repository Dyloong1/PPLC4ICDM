"""Physics metrics for the paper's tables (Sections 4 + 5).

Implements the four scalar fidelity diagnostics:

* ``epsilon_strain`` -- pseudo-dissipation ``eps = nu * <|grad u|^2>``
  computed by integrating ``|ik * u_hat|^2`` in spectral space.
* ``enstrophy``      -- ``Omega = 0.5 * <|omega|^2>`` with vorticity
  ``omega = curl u`` computed in spectral space.
* ``inertial_slope`` -- log-log least-squares fit of ``E(k)`` on
  ``k in [4, 40]``.
* ``relative_divergence`` -- ``std(div u) / std(u_x)``; target -> 0.

The function ``physics_on_gpu_streaming`` is the streaming routine the
eval driver uses on full ``1024^3`` frames; it computes all four scalars
plus the radial spectrum without ever holding more than two complex
spectra at once.

Three convention fixes documented for reviewers:
    1. ``E(k)`` is a shell SUM (not a shell mean). See ``spectra.py``.
    2. The spectral derivative multiplier is ``ik`` with the integer
       wavenumber returned by ``torch.fft.fftfreq(N, d=1/N)`` (i.e. no
       ``2 * pi`` factor; the domain is normalized to ``[0, N)``).
    3. The divergence sum ``div u = du_x/dx + du_y/dy + du_z/dz`` pairs
       velocity-component ``c`` with the wavenumber along the matching
       physical axis (``c=0 <-> kx``, ``c=1 <-> ky``, ``c=2 <-> kz``),
       which is the opposite of the ``torch.meshgrid(indexing='ij')``
       default if you reuse the meshgrid output blindly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .spectra import energy_spectrum

# JHTDB isotropic1024coarse kinematic viscosity.
NU = 1.85e-4


# ---------------------------------------------------------------------------
# Standalone scalar metrics (small-grid: 256^3 or smaller)
# ---------------------------------------------------------------------------

def _velocity_gradient_tensor(velocity: torch.Tensor) -> torch.Tensor:
    """Spectral velocity gradient ``A[i, j] = d u_i / d x_j``.

    Args:
        velocity: ``(B, 3, Z, H, W)``.

    Returns:
        ``(B, 3, 3, Z, H, W)``.
    """
    B, C, Z, H, W = velocity.shape
    device = velocity.device
    assert C == 3

    kz = torch.fft.fftfreq(Z, d=1.0 / Z, device=device)
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=device)
    kx = torch.fft.rfftfreq(W, d=1.0 / W, device=device)
    if Z % 2 == 0: kz[Z // 2] = 0
    if H % 2 == 0: ky[H // 2] = 0
    if W % 2 == 0: kx[W // 2] = 0
    KZ = kz[None, :, None, None]
    KY = ky[None, None, :, None]
    KX = kx[None, None, None, :]
    k_for_dir = [KX, KY, KZ]  # j=0 -> x, j=1 -> y, j=2 -> z

    grad = torch.zeros(B, 3, 3, Z, H, W, device=device, dtype=velocity.dtype)
    for i in range(3):
        u_hat = torch.fft.rfftn(velocity[:, i], dim=(-3, -2, -1))
        for j in range(3):
            d_hat = 1j * k_for_dir[j] * u_hat
            grad[:, i, j] = torch.fft.irfftn(d_hat, s=(Z, H, W), dim=(-3, -2, -1))
    return grad


def epsilon_strain(velocity: torch.Tensor, *, nu: float = NU):
    """Pseudo-dissipation ``eps = nu * <|grad u|^2>``."""
    grad = _velocity_gradient_tensor(velocity)
    grad_sq = (grad * grad).sum(dim=(1, 2))  # (B, Z, H, W)
    return nu * grad_sq.mean(dim=(-3, -2, -1))


def enstrophy(velocity: torch.Tensor):
    """``0.5 * <|omega|^2>`` with ``omega = curl u``."""
    grad = _velocity_gradient_tensor(velocity)
    # omega_x = du_z/dy - du_y/dz, etc.
    wx = grad[:, 2, 1] - grad[:, 1, 2]
    wy = grad[:, 0, 2] - grad[:, 2, 0]
    wz = grad[:, 1, 0] - grad[:, 0, 1]
    omega_sq = wx ** 2 + wy ** 2 + wz ** 2
    return 0.5 * omega_sq.mean(dim=(-3, -2, -1))


def relative_divergence(velocity: torch.Tensor) -> torch.Tensor:
    """``std(div u) / std(u_x)``. Target -> 0 (incompressibility)."""
    grad = _velocity_gradient_tensor(velocity)
    div = grad[:, 0, 0] + grad[:, 1, 1] + grad[:, 2, 2]
    div_std = div.flatten(1).std(dim=1)
    u_std = velocity[:, 0].flatten(1).std(dim=1)
    return div_std / (u_std + 1e-30)


def inertial_slope(wavenumbers, E_k, *, k_min: int = 4,
                   k_max: int = 40) -> float:
    """Log-log least-squares slope of ``E(k)`` on ``k in [k_min, k_max]``."""
    k = np.asarray(wavenumbers, dtype=np.float64)
    if E_k.ndim == 2:
        E = np.asarray(E_k.mean(axis=0), dtype=np.float64)
    else:
        E = np.asarray(E_k, dtype=np.float64)
    mask = (k >= k_min) & (k <= k_max) & (E > 0)
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[mask]), np.log(E[mask]), 1)[0])


# ---------------------------------------------------------------------------
# Streaming version for 1024^3 evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def physics_on_gpu_streaming(velocity_callable: Callable[[int], torch.Tensor],
                             *, nu: float = NU, N: int = 1024,
                             slope_k_min: int = 4, slope_k_max: int = 40):
    """Streaming physics metrics for ``1024^3`` fields.

    ``velocity_callable(c)`` should return the channel-c velocity field
    as a real ``(N, N, N)`` tensor on the GPU. We never hold more than
    two ``rfftn``-shaped complex tensors at once.

    Returns a dict with:
        ``epsilon``       -- ``nu * <|grad u|^2>``
        ``enstrophy``     -- ``0.5 * <|omega|^2>`` via curl in spectral space
        ``div_std``       -- ``std(div u)``
        ``inertial_slope``-- log-log slope of E(k) on [k_min, k_max]
        ``wavenumbers``   -- integer ``k = 0 .. N/2 - 1``
        ``E_k``           -- shell-summed radial spectrum
    """
    device = torch.device("cuda")
    k_max = N // 2
    E_k = torch.zeros(k_max, device=device)
    div_field = torch.zeros((N, N, N), device=device)
    grad_norm_sq = torch.zeros((N, N, N), device=device)

    k_full = torch.fft.fftfreq(N, d=1.0 / N).to(device)
    k_half = torch.fft.rfftfreq(N, d=1.0 / N).to(device)
    Nh = N // 2 + 1
    kz_b = k_full.view(N, 1, 1)
    ky_b = k_full.view(1, N, 1)
    kx_b = k_half.view(1, 1, Nh)
    # Channel c -> matching spectral axis (see module docstring, fix #3).
    k_for_channel = [kx_b, ky_b, kz_b]
    ik = 1j

    # Pass 1: gradient-norm + divergence + spectrum.
    for c in range(3):
        u = velocity_callable(c).float()
        u_hat = torch.fft.rfftn(u, norm="forward")
        del u; torch.cuda.empty_cache()
        for k_grad_b in (kx_b, ky_b, kz_b):
            tmp = u_hat * (ik * k_grad_b)
            u_grad = torch.fft.irfftn(tmp, s=(N, N, N), norm="forward")
            del tmp
            grad_norm_sq += u_grad ** 2
            del u_grad; torch.cuda.empty_cache()
        tmp = u_hat * (ik * k_for_channel[c])
        u_grad = torch.fft.irfftn(tmp, s=(N, N, N), norm="forward")
        del tmp
        div_field += u_grad
        del u_grad; torch.cuda.empty_cache()

        amp_sq_half = u_hat.real ** 2 + u_hat.imag ** 2
        weight = torch.ones((1, 1, Nh), device=device)
        if Nh >= 2:
            weight[..., 1:-1] = 2.0
        amp_sq_half = amp_sq_half * weight
        k_mag = torch.sqrt(kx_b ** 2 + ky_b ** 2 + kz_b ** 2)
        k_bin = torch.round(k_mag).long().clamp(0, k_max - 1)
        del k_mag, weight
        E_k.scatter_add_(0, k_bin.flatten(),
                         (amp_sq_half * 0.5).flatten())
        del amp_sq_half, k_bin, u_hat; torch.cuda.empty_cache()

    # Pass 2: curl pairs for enstrophy.
    enstrophy_acc = torch.zeros((N, N, N), device=device)
    # For w_i, the term is ``ik_a * u_hat_b - ik_b * u_hat_a`` where
    # ``(b, a)`` follows the right-hand-rule cyclic order:
    curl_specs = [
        (2, ky_b, 1, kz_b),   # w_x = ik_y * u_hat_z - ik_z * u_hat_y
        (0, kz_b, 2, kx_b),   # w_y = ik_z * u_hat_x - ik_x * u_hat_z
        (1, kx_b, 0, ky_b),   # w_z = ik_x * u_hat_y - ik_y * u_hat_x
    ]
    for a_chan, a_k, b_chan, b_k in curl_specs:
        u_a = velocity_callable(a_chan).float()
        u_hat_a = torch.fft.rfftn(u_a, norm="forward")
        del u_a; torch.cuda.empty_cache()
        u_b = velocity_callable(b_chan).float()
        u_hat_b = torch.fft.rfftn(u_b, norm="forward")
        del u_b; torch.cuda.empty_cache()
        tmp_a = u_hat_a * (ik * a_k); del u_hat_a
        tmp_b = u_hat_b * (ik * b_k); del u_hat_b
        w_hat = tmp_a - tmp_b
        del tmp_a, tmp_b
        w_real = torch.fft.irfftn(w_hat, s=(N, N, N), norm="forward")
        del w_hat
        enstrophy_acc += w_real ** 2
        del w_real; torch.cuda.empty_cache()

    eps = float((nu * grad_norm_sq.mean()).cpu())
    Omega = float(0.5 * enstrophy_acc.mean())
    div_std = float(div_field.std().cpu())
    del grad_norm_sq, enstrophy_acc, div_field; torch.cuda.empty_cache()

    E_k_np = E_k.cpu().numpy()
    wavenumbers = np.arange(k_max)
    valid = (wavenumbers >= slope_k_min) & (wavenumbers <= slope_k_max) & (E_k_np > 0)
    slope = (float(np.polyfit(np.log(wavenumbers[valid]),
                              np.log(E_k_np[valid]), 1)[0])
             if valid.sum() >= 2 else float("nan"))

    return {
        "epsilon": eps,
        "enstrophy": Omega,
        "div_std": div_std,
        "inertial_slope": slope,
        "wavenumbers": wavenumbers,
        "E_k": E_k_np,
    }
