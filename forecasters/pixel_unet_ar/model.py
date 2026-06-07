"""Pixel-space 3D U-Net forecaster (autoregressive).

A tiny 3D U-Net mapping ``x_t`` -> ``x_{t + trained_tau}`` directly on
``32^3`` patches, with FiLM-style timestep conditioning. Rolled
autoregressively for longer horizons.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoid(t, d, max_period: float = 1e4):
    half = d // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Block(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c)
        self.shortcut = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[..., None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class PixelUNetForecaster(nn.Module):
    def __init__(self, *, in_channels: int = 4, ch: int = 64,
                 ch_mult=(1, 2, 4), t_dim: int = 256):
        super().__init__()
        self.t_dim = t_dim
        self.t_embed = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        chs = [ch * m for m in ch_mult]
        self.in_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        self.downs = nn.ModuleList([
            _Block(chs[i], chs[i + 1], t_dim) for i in range(len(chs) - 1)
        ])
        self.mid = _Block(chs[-1], chs[-1], t_dim)
        self.ups = nn.ModuleList([
            _Block(chs[i + 1] + chs[i], chs[i], t_dim)
            for i in reversed(range(len(chs) - 1))
        ])
        self.out_conv = nn.Conv3d(chs[0], in_channels, 3, padding=1)

    def forward(self, x, tau):
        t_emb = self.t_embed(_sinusoid(tau.float(), self.t_dim))
        h = self.in_conv(x)
        skips = [h]
        for blk in self.downs:
            h = blk(h, t_emb); skips.append(h); h = F.avg_pool3d(h, 2)
        h = self.mid(h, t_emb)
        for blk in self.ups:
            h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
            h = blk(torch.cat([h, skips.pop()], dim=1), t_emb)
        return self.out_conv(h)
