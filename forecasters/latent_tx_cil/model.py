"""Latent-space Transformer with direct-tau + CIL (headline forecaster).

Pipeline:

    context = (z_{t-10}, z_{t-5}, z_t)   # 3 frozen PPLC latents
    target  = z_{t+tau}                  # ground-truth latent at horizon tau

A Transformer encoder operates on the concatenated context tokens; a
linear head reads out the predicted target latent at the requested
horizon ``tau``.

CIL (Conditional Image Leakage, Zhao NeurIPS 2024) perturbs the context
latents at training time by ``z_t' = z_t + beta_m * eps`` with
``beta_m = 0.5`` and ``eps ~ N(0, I)`` to prevent the network from
over-fitting to exact-match context features when forecasting long
horizons.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_LEN = 3
LATENT_SHAPE = (4, 8, 8, 8)
LATENT_NUMEL = 4 * 8 * 8 * 8


class _SinusoidalEmbed(nn.Module):
    """Standard sinusoidal embedding of a scalar tau."""

    def __init__(self, d_model: int, max_period: float = 1.0e4):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
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


class LatentTransformerForecaster(nn.Module):
    """Direct-tau Transformer forecaster over PPLC latents.

    Args:
        d_model: hidden width.
        n_heads: number of attention heads.
        n_layers: encoder depth.
        beta_m: CIL noise standard deviation applied to the context tokens
            at training time (set to 0 to disable CIL).
    """

    def __init__(self, *, d_model: int = 384, n_heads: int = 6,
                 n_layers: int = 6, beta_m: float = 0.5):
        super().__init__()
        self.beta_m = beta_m
        self.in_proj = nn.Linear(LATENT_NUMEL, d_model)
        self.context_pos = nn.Parameter(torch.zeros(CONTEXT_LEN, d_model))
        self.tau_embed = _SinusoidalEmbed(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, LATENT_NUMEL)

    def forward(self, z_context: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Args:
            z_context: ``(B, 3, *LATENT_SHAPE)``.
            tau: ``(B,)`` integer horizon.

        Returns:
            ``(B, *LATENT_SHAPE)`` predicted latent at horizon ``tau``.
        """
        B = z_context.shape[0]
        x = z_context.reshape(B, CONTEXT_LEN, -1)
        if self.training and self.beta_m > 0:
            x = x + self.beta_m * torch.randn_like(x)
        x = self.in_proj(x) + self.context_pos[None]
        tau_token = self.tau_embed(tau).unsqueeze(1)        # (B, 1, d_model)
        tokens = torch.cat([x, tau_token], dim=1)             # (B, 4, d_model)
        h = self.encoder(tokens)
        # Read out from the last (tau) token.
        pred = self.out_proj(h[:, -1])
        return pred.view(B, *LATENT_SHAPE)
