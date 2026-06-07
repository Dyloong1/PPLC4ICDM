"""PPLC training losses.

Generator-side total: ``L_G = L1 + lambda_grad * L_grad + beta_KL * L_KL
+ lambda_adv * L_adv + lambda_consist * L_consist``.

* ``L1`` — voxel L1 between recon and target (channels: u, v, w, p).
* ``L_grad`` — L1 of first-order spatial finite-differences on the velocity
  channels (Sobolev-H1 prior; up-weights high-k content).
* ``L_KL`` — KL divergence of the encoder posterior against the standard
  normal prior.
* ``L_adv`` — non-saturating generator loss against a 3D patch
  discriminator (hinge formulation in :class:`PatchDiscriminator3D`).
* ``L_consist`` — circular-shift consistency: with shift ``k=8`` voxels,
  encourages ``D(model(x))`` to match ``D(model(shift_k(x)))``. Mitigates
  patch-boundary spectral leakage when the trained codec is later applied
  to whole frames at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base VAE terms
# ---------------------------------------------------------------------------

def l1_loss(recon, target):
    return F.l1_loss(recon, target)


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)) per element (FP32-safe cast)."""
    mu = mu.float()
    logvar = logvar.float()
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _periodic_diff(x, dim):
    """Forward first difference with periodic wrap: ``x[i+1] - x[i]``."""
    return torch.roll(x, shifts=-1, dims=dim) - x


def gradient_l1(recon, target):
    """L1 of first-order finite-difference gradients on velocity channels.

    Operates on channels 0..2 (u, v, w). Pressure (channel 3) is ignored
    here because the spectral leakage we care about lives in the velocity
    derivatives.
    """
    vel_r = recon[:, :3]
    vel_t = target[:, :3]
    loss = 0.0
    for dim in (-3, -2, -1):
        loss = loss + F.l1_loss(_periodic_diff(vel_r, dim),
                                _periodic_diff(vel_t, dim))
    return loss / 3.0


# ---------------------------------------------------------------------------
# 3D patch discriminator (hinge GAN)
# ---------------------------------------------------------------------------

class PatchDiscriminator3D(nn.Module):
    """Small 3D conv stack with three stride-2 stages and a 1x1x1 readout.

    Input  ``(B, in_channels, 32, 32, 32)``.
    Output ``(B,)`` mean-pooled scalar logit per patch.
    """

    def __init__(self, in_channels: int = 4, base_ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, base_ch, 4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, base_ch)
        self.conv2 = nn.Conv3d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, base_ch * 2)
        self.conv3 = nn.Conv3d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, base_ch * 4)
        self.head = nn.Conv3d(base_ch * 4, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.act(self.gn2(self.conv2(h)))
        h = self.act(self.gn3(self.conv3(h)))
        h = self.head(h)
        return h.mean(dim=(-3, -2, -1)).squeeze(1)


def hinge_discriminator_loss(discriminator, recon_detached, target_detached):
    """Standard hinge loss: ``D`` step."""
    d_real = discriminator(target_detached)
    d_fake = discriminator(recon_detached)
    d_loss = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()) / 2.0
    return d_loss, d_real, d_fake


# ---------------------------------------------------------------------------
# Consistency (shift-equivariance) loss
# ---------------------------------------------------------------------------

def consistency_loss(model, x, shift: int = 8, axis: int = -1,
                     amp_dtype=None):
    """``L1(model(shift_k(x)), shift_k(model(x).detach()))``.

    The detached target removes the trivial gradient path through the
    unshifted forward; only the shifted forward is constrained.
    """
    x_shift = torch.roll(x, shifts=shift, dims=axis)
    if amp_dtype is not None:
        with torch.autocast("cuda", dtype=amp_dtype):
            recon_shift, _, _ = model(x_shift)
        recon_shift = recon_shift.float()
        with torch.autocast("cuda", dtype=amp_dtype):
            recon_unshift, _, _ = model(x)
        recon_unshift = recon_unshift.float()
    else:
        recon_shift, _, _ = model(x_shift)
        recon_unshift, _, _ = model(x)
    target = torch.roll(recon_unshift.detach(), shifts=shift, dims=axis)
    return F.l1_loss(recon_shift, target)


# ---------------------------------------------------------------------------
# Generator total
# ---------------------------------------------------------------------------

def pplc_generator_loss(recon, target, mu, logvar, discriminator,
                        *,
                        beta_kl: float = 1e-2,
                        lambda_grad: float = 0.5,
                        lambda_adv: float = 1e-2):
    """Returns ``(total, log_dict, recon_detached, target_detached)``.

    Note: ``lambda_consist`` is applied separately by the training loop
    via :func:`consistency_loss` because the consistency term needs a
    second forward pass through the model itself, not just through
    ``recon`` / ``target`` tensors.
    """
    l_recon = l1_loss(recon, target)
    l_kl = kl_divergence(mu, logvar)
    l_grad = gradient_l1(recon, target)

    # Generator-side adv: freeze D's params for this forward, then re-enable.
    for p in discriminator.parameters():
        p.requires_grad_(False)
    d_fake = discriminator(recon)
    for p in discriminator.parameters():
        p.requires_grad_(True)
    l_adv = -d_fake.mean()

    total = l_recon + lambda_grad * l_grad + beta_kl * l_kl + lambda_adv * l_adv

    log = {
        "l1_recon": float(l_recon.item()),
        "kl": float(l_kl.item()),
        "gradient": float(l_grad.item()),
        "g_adv": float(l_adv.item()),
        "d_fake_mean": float(d_fake.detach().mean().item()),
        "total_g": float(total.item()),
    }
    return total, log, recon.detach(), target.detach()
