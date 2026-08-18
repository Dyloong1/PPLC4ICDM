"""Paper-faithful DC-AE 3D (Deep Compression Autoencoder).

Reference: Chen et al. ICLR 2025, "Deep Compression Autoencoder for
Efficient High-Resolution Diffusion Models" (arXiv:2410.10733).

Official 2D code: github.com/mit-han-lab/efficientvit
  efficientvit/models/efficientvit/dc_ae.py
  Key components:
    - ConvPixelUnshuffleDownSampleLayer  (space-to-channel downsample)
    - ConvPixelShuffleUpSampleLayer       (channel-to-space upsample)
    - PixelUnshuffleChannelAveragingDownSampleLayer (averaging shortcut)
    - ChannelDuplicatingPixelUnshuffleUpSampleLayer (duplicating shortcut)
    - ResBlock (3x3 conv + norm + SiLU)
    - Residual connection (key technique: "Residual Autoencoding")

Paper recommended configs:
    dc_ae_f32c32:   width=(128,256,512,512,1024,1024), depth=(2,2,2,2,2,2), f=32, c=32
    dc_ae_f64c128:  same width/depth, f=64, c=128
    dc_ae_f128c512: f=128, c=512

For 32^3 turbulence patches we use a scaled-down config:
    width=(64,128,256,256), depth=(2,2,2,2), f=4 (32->8), c=32
    Latent: (32, 8, 8, 8) = 16384 floats
    Compression: 4*32^3 / (32*8^3) = 131072/16384 = 8x  -- too low for paper
    -> we use c=4 -> ratio = 131072/(4*512) = 64x  (matches headline)

Loss: L1 (paper uses similar reconstruction-focused loss). The official
DC-AE for diffusion pretraining uses L1+LPIPS+GAN as in SD-VAE; for our
ICDM compression-only comparison we use just L1 (no LPIPS/GAN for fair
fast convergence on 3D turbulence).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x): return F.silu(x)
def Normalize(c, num_groups=32):
    return nn.GroupNorm(num_groups=min(num_groups, c), num_channels=c, eps=1e-6, affine=True)


class ResBlock3d(nn.Module):
    """3D ResBlock matching DC-AE official ResBlock (3x3 conv + norm + act)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = nonlinearity(self.norm2(h))
        h = self.conv2(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


def pixel_unshuffle_3d(x, factor=2):
    """3D PixelUnshuffle: (B, C, fD, fH, fW) -> (B, C*f^3, D, H, W).

    Channel ordering matches torch.nn.PixelUnshuffle convention extended to 3D.
    """
    B, C, D, H, W = x.shape
    f = factor
    x = x.view(B, C, D//f, f, H//f, f, W//f, f)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return x.view(B, C * f**3, D//f, H//f, W//f)


def pixel_shuffle_3d(x, factor=2):
    """Inverse: (B, C*f^3, D, H, W) -> (B, C, fD, fH, fW)."""
    B, Cf, D, H, W = x.shape
    f = factor
    C = Cf // (f**3)
    x = x.view(B, C, f, f, f, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(B, C, D*f, H*f, W*f)


class ConvPixelUnshuffleDownsample3d(nn.Module):
    """DC-AE downsample: PixelUnshuffle (2x) + 1x1 conv to merge channels.

    From efficientvit:
      pixel_unshuffle(2) increases C by 8 (in 3D, by 2^3=8 since each dim halves)
      Then 1x1 conv reduces C to the target out_channels.
    """

    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv3d(in_channels * factor**3, out_channels, 1, 1, 0)

    def forward(self, x):
        x = pixel_unshuffle_3d(x, self.factor)
        return self.conv(x)


class ConvPixelShuffleUpsample3d(nn.Module):
    """DC-AE upsample: 1x1 conv expands C, then PixelShuffle (2x)."""

    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv3d(in_channels, out_channels * factor**3, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return pixel_shuffle_3d(x, self.factor)


class PixelUnshuffleAveragingShortcut3d(nn.Module):
    """DC-AE downsample shortcut: PixelUnshuffle + channel averaging.
    Used in encoder's residual path to preserve information without learnable weights."""

    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        # after pixel_unshuffle: in*f^3 channels; we average groups to get out_channels
        total = in_channels * factor**3
        assert total % out_channels == 0, f"can't avg {total} -> {out_channels}"
        self.group_size = total // out_channels

    def forward(self, x):
        x = pixel_unshuffle_3d(x, self.factor)  # (B, in*f^3, D, H, W)
        B, C, D, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, D, H, W).mean(dim=2)
        return x


class ChannelDuplicatingPixelShuffleShortcut3d(nn.Module):
    """DC-AE upsample shortcut: channel duplication + PixelShuffle."""

    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        total = out_channels * factor**3
        assert total % in_channels == 0
        self.repeat = total // in_channels

    def forward(self, x):
        x = x.repeat_interleave(self.repeat, dim=1)  # (B, out*f^3, D, H, W)
        return pixel_shuffle_3d(x, self.factor)


class DCAEStage3d(nn.Module):
    """One stage: depth ResBlocks. Includes residual shortcut for residual-autoencoding."""

    def __init__(self, channels, depth):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock3d(channels, channels) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class DCAEEncoder3d(nn.Module):
    def __init__(self, in_channels=4, latent_channels=4,
                 width=(64, 128, 256, 256), depth=(2, 2, 2, 2)):
        super().__init__()
        assert len(width) == len(depth)
        # project_in: 1x1 conv to first width
        self.project_in = nn.Conv3d(in_channels, width[0], 3, 1, 1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        n_stages = len(width)
        for i in range(n_stages):
            self.stages.append(DCAEStage3d(width[i], depth[i]))
            if i < n_stages - 1:
                self.downsamples.append(
                    ConvPixelUnshuffleDownsample3d(width[i], width[i+1], factor=2))
                self.shortcuts.append(
                    PixelUnshuffleAveragingShortcut3d(width[i], width[i+1], factor=2))
        # project_out
        self.project_out_norm = Normalize(width[-1])
        self.project_out = nn.Conv3d(width[-1], latent_channels, 3, 1, 1)

    def forward(self, x):
        x = self.project_in(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                # residual autoencoding: shortcut + main downsample
                shortcut = self.shortcuts[i](x)
                main = self.downsamples[i](x)
                x = main + shortcut
        x = nonlinearity(self.project_out_norm(x))
        return self.project_out(x)


class DCAEDecoder3d(nn.Module):
    def __init__(self, out_channels=4, latent_channels=4,
                 width=(64, 128, 256, 256), depth=(2, 2, 2, 2)):
        super().__init__()
        assert len(width) == len(depth)
        n_stages = len(width)
        # project_in: 1x1 conv from latent to last-stage width
        self.project_in = nn.Conv3d(latent_channels, width[-1], 3, 1, 1)
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        # decoder: reverse the order
        for i in range(n_stages - 1, -1, -1):
            self.stages.append(DCAEStage3d(width[i], depth[i]))
            if i > 0:
                self.upsamples.append(
                    ConvPixelShuffleUpsample3d(width[i], width[i-1], factor=2))
                self.shortcuts.append(
                    ChannelDuplicatingPixelShuffleShortcut3d(width[i], width[i-1], factor=2))
        # project_out
        self.project_out_norm = Normalize(width[0])
        self.project_out = nn.Conv3d(width[0], out_channels, 3, 1, 1)

    def forward(self, z):
        x = self.project_in(z)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.upsamples):
                shortcut = self.shortcuts[i](x)
                main = self.upsamples[i](x)
                x = main + shortcut
        x = nonlinearity(self.project_out_norm(x))
        return self.project_out(x)


class DCAE3D(nn.Module):
    """Paper-faithful 3D DC-AE port.

    Default config (32^3 patches, headline 64x ratio):
      width=(64,128,256,256), depth=(2,2,2,2), latent_channels=4
      Spatial 32 -> 4 (3 downsamples), latent (4, 4, 4, 4) = 256 elements? wait
      Actually we have n_stages=4, n_downsamples=3 → spatial /2^3 = 32/8 = 4
      Latent (latent_channels=4, 4, 4, 4) = 256 elements
      Compression = 4*32^3 / 256 = 512x  -- too aggressive!
      For 64x tier: 4*32^3 / 64 = 2048 elements
        -> use n_downsamples=2 (3 stages), latent (4, 8, 8, 8) = 2048

    So default config we use: width=(64,128,256), depth=(2,2,2), latent_channels=4
    Spatial 32 -> 8 (2 downsamples)
    Compression = 4*32^3 / (4*8^3) = 131072/2048 = 64x  ✓
    """

    def __init__(self, in_channels=4, out_channels=4, latent_channels=4,
                 width=(64, 128, 256), depth=(2, 2, 2)):
        super().__init__()
        self.encoder = DCAEEncoder3d(in_channels, latent_channels, width, depth)
        self.decoder = DCAEDecoder3d(out_channels, latent_channels, width, depth)
        self.latent_channels = latent_channels

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def dcae_loss(recon, target, recon_weight=1.0):
    """L1 reconstruction loss (paper-faithful for autoencoder pretraining)."""
    return recon_weight * F.l1_loss(recon, target)
