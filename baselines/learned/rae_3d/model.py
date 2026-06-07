"""Paper-faithful 3D RAE (Regularized Auto-Encoder) port.

Zheng et al. 2025 (arXiv:2510.11690). RAE replaces a VAE's KL term with
an explicit decoder-side denoising objective: the encoder is frozen
(here we use a pre-trained SD-VAE-3D first stage), the decoder is
trained from scratch with input noise

    z_noisy = z + tau * eps,    eps ~ N(0, I)

so the decoder learns to invert a noisy latent back to the input field.

Architecture:
    Encoder: ``baselines.learned.sdvae_3d.SDVAE3D.encode`` (frozen).
    Decoder: a 3D CNN that upsamples the (latent_ch, 8, 8, 8) latent
             back to (4, 32, 32, 32).
    Loss:    L1 only; the noise schedule replaces the KL regulariser.

Returns ``(recon, z, None, None)`` so the four-tuple interface matches
the other learned baselines (the two trailing slots are ``mean / logvar``
in the VAE family, both undefined for an AE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sdvae_3d.model import SDVAE3D


def _normalize(c, num_groups: int = 32):
    return nn.GroupNorm(min(num_groups, c), c, eps=1e-6, affine=True)


def _silu(x):
    return F.silu(x)


class _ResBlock3d(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm1 = _normalize(in_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, 1, 1)
        self.norm2 = _normalize(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, 1)
        self.shortcut = (
            nn.Conv3d(in_c, out_c, 1, 1, 0) if in_c != out_c else None
        )

    def forward(self, x):
        h = _silu(self.norm1(x)); h = self.conv1(h)
        h = _silu(self.norm2(h)); h = self.conv2(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


class RAE3DDecoder(nn.Module):
    """``(B, latent_ch, 8, 8, 8) -> (B, out_channels, 32, 32, 32)``.

    Two ConvTranspose3d upsamples (8 -> 16 -> 32) sandwiched between
    residual blocks. Roughly 10 M parameters.
    """

    def __init__(self, latent_ch: int = 4, out_channels: int = 4,
                 base_ch: int = 256):
        super().__init__()
        self.in_conv = nn.Conv3d(latent_ch, base_ch, 3, 1, 1)
        self.mid_blocks = nn.Sequential(
            _ResBlock3d(base_ch, base_ch),
            _ResBlock3d(base_ch, base_ch),
        )
        self.up1 = nn.ConvTranspose3d(base_ch, base_ch // 2, 4, stride=2, padding=1)
        self.res1 = nn.Sequential(
            _ResBlock3d(base_ch // 2, base_ch // 2),
            _ResBlock3d(base_ch // 2, base_ch // 2),
        )
        self.up2 = nn.ConvTranspose3d(base_ch // 2, base_ch // 4, 4, stride=2, padding=1)
        self.res2 = nn.Sequential(
            _ResBlock3d(base_ch // 4, base_ch // 4),
            _ResBlock3d(base_ch // 4, base_ch // 4),
        )
        self.out_norm = _normalize(base_ch // 4)
        self.out_conv = nn.Conv3d(base_ch // 4, out_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        h = self.mid_blocks(h)
        h = self.up1(h); h = self.res1(h)
        h = self.up2(h); h = self.res2(h)
        return self.out_conv(_silu(self.out_norm(h)))


class RAE3D(nn.Module):
    """Frozen SD-VAE encoder + trained CNN decoder + noise augmentation.

    Args:
        latent_ch: number of latent channels (must match the frozen encoder).
        out_channels: number of output channels (u, v, w, p = 4).
        base_ch: decoder width.
        noise_tau: standard deviation of the latent-space Gaussian noise
            added at training time. Set to 0 for evaluation.
        frozen_encoder_ckpt: optional path to a pre-trained SD-VAE-3D
            checkpoint. If ``None``, the encoder is randomly initialized
            (this is fine for smoke tests, but the paper-faithful RAE
            *requires* a pre-trained frozen encoder).
    """

    def __init__(self, *, latent_ch: int = 4, out_channels: int = 4,
                 base_ch: int = 256, noise_tau: float = 0.0,
                 frozen_encoder_ckpt: str | None = None,
                 sdvae_ch: int = 64, sdvae_ch_mult=(1, 2, 4),
                 sdvae_num_res_blocks: int = 1):
        super().__init__()
        self.latent_ch = latent_ch
        self.noise_tau = noise_tau

        self.encoder = SDVAE3D(
            in_channels=out_channels, out_channels=out_channels,
            ch=sdvae_ch, ch_mult=sdvae_ch_mult,
            num_res_blocks=sdvae_num_res_blocks,
            z_channels=latent_ch, embed_dim=latent_ch,
        )
        if frozen_encoder_ckpt is not None:
            ckpt = torch.load(frozen_encoder_ckpt, map_location="cpu",
                              weights_only=False)
            state = None
            for key in ("model_state", "model", "state_dict", "net"):
                if isinstance(ckpt, dict) and key in ckpt and isinstance(
                        ckpt[key], dict):
                    state = ckpt[key]; break
            if state is None:
                state = ckpt
            self.encoder.load_state_dict(state, strict=False)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.decoder = RAE3DDecoder(
            latent_ch=latent_ch, out_channels=out_channels, base_ch=base_ch,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.encoder.eval()
            mean, _ = self.encoder.encode(x)
        return mean.detach()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        if self.training and self.noise_tau > 0:
            z_noisy = z + self.noise_tau * torch.randn_like(z)
        else:
            z_noisy = z
        recon = self.decode(z_noisy)
        return recon, z, None, None

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self


def rae_loss(recon, target, recon_weight: float = 1.0):
    """Plain L1 reconstruction (paper-faithful)."""
    return recon_weight * F.l1_loss(recon, target)
