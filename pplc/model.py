"""PPLC (Physics-Preserving Latent Compressor) — Section 3 of the paper.

Three pieces:

1. ``AEResBlock3D`` — the residual block used throughout the encoder /
   decoder ladder.
2. ``PPLCSpatial8`` — the headline 64x tier: ``latent_channels=4`` with a
   spatial ``8^3`` latent. Per-patch mean is stored exactly as a 4-vector
   (the "content"), so the latent budget per ``32^3`` patch is
   ``4 + 4 * 8^3 = 2052 floats`` -> ratio ``131072 / 2052 = 63.9x``.
3. ``PPLCChannelHeavy`` — the ablation variant: ``latent_channels`` is
   larger but the spatial latent collapses to ``4^3``. The headline 64x
   tier of this variant uses ``latent_channels=32`` -> budget
   ``4 + 32 * 4^3 = 2052 floats``, also ``63.9x``.

Both variants share the per-patch mean / fluctuation split. The encoder
operates on the zero-mean fluctuation; the mean bypasses the codec and
is added back inside ``decode``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .haar_wavelet import haar_forward_3d, haar_inverse_3d


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

class AEResBlock3D(nn.Module):
    """GroupNorm + SiLU + Conv3d * 2 + identity residual."""

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# Spatial-8 (headline 64x) encoder + decoder
# ---------------------------------------------------------------------------

class _Spatial8Encoder(nn.Module):
    """4 -> Haar(32) -> 128 -> ResBlock -> down to 8^3 -> 4 ResBlocks -> 2 * Cl."""

    def __init__(self, in_channels: int = 4, latent_channels: int = 4,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        haar_ch = in_channels * 8  # 4 * 8 = 32

        self.input_conv = nn.Conv3d(haar_ch, 128, 3, padding=1)
        self.input_res = AEResBlock3D(128, 128)
        self.down1 = nn.Conv3d(128, 128, 4, stride=2, padding=1)
        self.res1 = nn.Sequential(
            AEResBlock3D(128, 256), AEResBlock3D(256, 256),
            AEResBlock3D(256, 384), AEResBlock3D(384, 384),
        )
        self.output_conv = nn.Sequential(
            nn.GroupNorm(1, 384), nn.SiLU(),
            nn.Conv3d(384, 2 * latent_channels, 1),
        )

    def _maybe_ckpt(self, module, x):
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, fluct: torch.Tensor) -> torch.Tensor:
        # fluct: (B, 4, 32, 32, 32) -> (B, 32, 16, 16, 16) after Haar
        x = haar_forward_3d(fluct)
        x = self.input_conv(x)
        x = self._maybe_ckpt(self.input_res, x)
        x = self.down1(x)                       # (B, 128, 8, 8, 8)
        x = self._maybe_ckpt(self.res1, x)
        return self.output_conv(x)              # (B, 2*Cl, 8, 8, 8)


class _Spatial8Decoder(nn.Module):
    """Mirror of _Spatial8Encoder."""

    def __init__(self, out_channels: int = 4, latent_channels: int = 4,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        haar_ch = out_channels * 8

        self.input_conv = nn.Sequential(
            nn.Conv3d(latent_channels, 384, 3, padding=1), nn.SiLU(),
        )
        self.res1 = nn.Sequential(
            AEResBlock3D(384, 384), AEResBlock3D(384, 256),
            AEResBlock3D(256, 256), AEResBlock3D(256, 128),
        )
        self.up1 = nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1)
        self.res_pre = AEResBlock3D(128, 128)
        self.output_conv = nn.Conv3d(128, haar_ch, 3, padding=1)
        self.output_res = AEResBlock3D(haar_ch, haar_ch)

    def _maybe_ckpt(self, module, x):
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(z)
        x = self._maybe_ckpt(self.res1, x)
        x = self.up1(x)                         # (B, 128, 16, 16, 16)
        x = self.res_pre(x)
        x = self.output_conv(x)
        x = self.output_res(x)
        return haar_inverse_3d(x)               # (B, 4, 32, 32, 32)


# ---------------------------------------------------------------------------
# Channel-heavy (ablation) encoder + decoder
# ---------------------------------------------------------------------------

class _ChannelHeavyEncoder(nn.Module):
    """Two-stride-2 ladder ending at a 4^3 spatial latent."""

    def __init__(self, in_channels: int = 4, latent_channels: int = 32,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        haar_ch = in_channels * 8

        self.input_conv = nn.Conv3d(haar_ch, 128, 3, padding=1)
        self.input_res = AEResBlock3D(128, 128)
        self.down1 = nn.Conv3d(128, 128, 4, stride=2, padding=1)  # 16 -> 8
        self.res1 = nn.Sequential(AEResBlock3D(128, 256), AEResBlock3D(256, 256))
        self.down2 = nn.Conv3d(256, 256, 4, stride=2, padding=1)  # 8 -> 4
        self.res2 = nn.Sequential(AEResBlock3D(256, 384), AEResBlock3D(384, 384))
        self.output_conv = nn.Sequential(
            nn.GroupNorm(1, 384), nn.SiLU(),
            nn.Conv3d(384, 2 * latent_channels, 1),
        )

    def _maybe_ckpt(self, module, x):
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, fluct: torch.Tensor) -> torch.Tensor:
        x = haar_forward_3d(fluct)
        x = self.input_conv(x)
        x = self._maybe_ckpt(self.input_res, x)
        x = self.down1(x)
        x = self._maybe_ckpt(self.res1, x)
        x = self.down2(x)
        x = self._maybe_ckpt(self.res2, x)
        return self.output_conv(x)              # (B, 2*Cl, 4, 4, 4)


class _ChannelHeavyDecoder(nn.Module):
    """Mirror of _ChannelHeavyEncoder."""

    def __init__(self, out_channels: int = 4, latent_channels: int = 32,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        haar_ch = out_channels * 8

        self.input_conv = nn.Sequential(
            nn.Conv3d(latent_channels, 384, 3, padding=1), nn.SiLU(),
        )
        self.res2 = nn.Sequential(AEResBlock3D(384, 384), AEResBlock3D(384, 256))
        self.up2 = nn.ConvTranspose3d(256, 256, 4, stride=2, padding=1)  # 4 -> 8
        self.res1 = nn.Sequential(AEResBlock3D(256, 256), AEResBlock3D(256, 128))
        self.up1 = nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1)  # 8 -> 16
        self.res_pre = AEResBlock3D(128, 128)
        self.output_conv = nn.Conv3d(128, haar_ch, 3, padding=1)
        self.output_res = AEResBlock3D(haar_ch, haar_ch)

    def _maybe_ckpt(self, module, x):
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(z)
        x = self._maybe_ckpt(self.res2, x)
        x = self.up2(x)
        x = self._maybe_ckpt(self.res1, x)
        x = self.up1(x)
        x = self.res_pre(x)
        x = self.output_conv(x)
        x = self.output_res(x)
        return haar_inverse_3d(x)


# ---------------------------------------------------------------------------
# Top-level PPLC model with mean-fluctuation split
# ---------------------------------------------------------------------------

class _PPLCBase(nn.Module):
    """Shared mean / fluctuation split + KL reparameterisation."""

    def __init__(self, in_channels: int, latent_channels: int,
                 encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def _split_mean(x):
        mu_c = x.mean(dim=(-3, -2, -1), keepdim=True)
        return mu_c, x - mu_c

    def encode(self, x: torch.Tensor):
        """Encode a patch into (mu, logvar, mu_c)."""
        mu_c, fluct = self._split_mean(x)
        h = self.encoder(fluct)
        mu, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)
        return mu, logvar, mu_c

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z, mu_c):
        fluct_recon = self.decoder(z)
        return fluct_recon + mu_c

    def forward(self, x):
        mu, logvar, mu_c = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, mu_c)
        return recon, mu, logvar


class PPLCSpatial8(_PPLCBase):
    """Headline 64x PPLC. ``latent_channels=4`` -> (4, 8, 8, 8) latent."""

    def __init__(self, in_channels: int = 4, latent_channels: int = 4,
                 gradient_checkpointing: bool = False):
        encoder = _Spatial8Encoder(in_channels, latent_channels, gradient_checkpointing)
        decoder = _Spatial8Decoder(in_channels, latent_channels, gradient_checkpointing)
        super().__init__(in_channels, latent_channels, encoder, decoder)


class PPLCChannelHeavy(_PPLCBase):
    """Channel-heavy ablation. ``latent_channels=32`` -> (32, 4, 4, 4) latent."""

    def __init__(self, in_channels: int = 4, latent_channels: int = 32,
                 gradient_checkpointing: bool = False):
        encoder = _ChannelHeavyEncoder(in_channels, latent_channels, gradient_checkpointing)
        decoder = _ChannelHeavyDecoder(in_channels, latent_channels, gradient_checkpointing)
        super().__init__(in_channels, latent_channels, encoder, decoder)


# Default alias used by the README + configs.
PPLC = PPLCSpatial8


def build_pplc(arch: str = "spatial8", latent_channels: int = 4,
               in_channels: int = 4,
               gradient_checkpointing: bool = False) -> _PPLCBase:
    """Factory used by training scripts / configs."""
    if arch == "spatial8":
        return PPLCSpatial8(in_channels, latent_channels, gradient_checkpointing)
    if arch == "channel_heavy":
        return PPLCChannelHeavy(in_channels, latent_channels, gradient_checkpointing)
    raise ValueError(f"unknown arch {arch!r}; expected 'spatial8' or 'channel_heavy'")


def _cli_info():
    """`python -m pplc.model --info` — print param counts and compression ratios."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--arch", default="spatial8", choices=["spatial8", "channel_heavy"])
    ap.add_argument("--latent_channels", type=int, default=4)
    args = ap.parse_args()
    m = build_pplc(args.arch, args.latent_channels)
    n_params = sum(p.numel() for p in m.parameters())
    if args.arch == "spatial8":
        latent_elems = 4 + args.latent_channels * 8 ** 3
    else:
        latent_elems = 4 + args.latent_channels * 4 ** 3
    patch_elems = 4 * 32 ** 3
    print(f"arch={args.arch} latent_channels={args.latent_channels}")
    print(f"params (M): {n_params / 1e6:.2f}")
    print(f"latent floats / patch: {latent_elems}")
    print(f"compression ratio: {patch_elems / latent_elems:.2f}x")


if __name__ == "__main__":
    _cli_info()
