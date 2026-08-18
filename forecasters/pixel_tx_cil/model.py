"""Pixel-space Transformer + CIL forecaster.

Same idea as ``latent_tx_cil`` but the Transformer operates directly on
``32^3`` voxel patches (no compressor in the loop). Token features are
patch-flattened pixels; otherwise identical to the latent variant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_LEN = 3
PATCH_NUMEL = 4 * 32 * 32 * 32


class _SinusoidalEmbed(nn.Module):
    def __init__(self, d_model: int, max_period: float = 1.0e4):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, t):
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class PixelTransformerForecaster(nn.Module):
    """Direct-tau Transformer forecaster on 32^3 patches."""

    def __init__(self, *, d_model: int = 384, n_heads: int = 6,
                 n_layers: int = 6, beta_m: float = 0.5,
                 patch_numel: int = PATCH_NUMEL):
        super().__init__()
        self.beta_m = beta_m
        self.patch_numel = patch_numel
        self.in_proj = nn.Linear(patch_numel, d_model)
        self.context_pos = nn.Parameter(torch.zeros(CONTEXT_LEN, d_model))
        self.tau_embed = _SinusoidalEmbed(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, patch_numel)

    def forward(self, x_context: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """``x_context`` has shape ``(B, 3, 4, 32, 32, 32)``."""
        B = x_context.shape[0]
        x = x_context.reshape(B, CONTEXT_LEN, -1)
        if self.training and self.beta_m > 0:
            x = x + self.beta_m * torch.randn_like(x)
        x = self.in_proj(x) + self.context_pos[None]
        tau_token = self.tau_embed(tau).unsqueeze(1)
        h = self.encoder(torch.cat([x, tau_token], dim=1))
        pred = self.out_proj(h[:, -1])
        return pred.view(B, 4, 32, 32, 32)
