"""Paper-faithful Cosmos Tokenizer CV 3D (NVIDIA 2025).

Reference: NVIDIA, "Cosmos World Foundation Model Platform for Physical AI"
(arXiv:2501.03575, Jan 2025). Code: github.com/NVIDIA/Cosmos-Tokenizer.

The CV ("Continuous Video") tokenizer is described in the paper as:
  - **Pure deterministic AE** — NO KL, NO commitment loss
    ("we do not use auxiliary losses tapped into the latent spaces")
  - 2-level wavelet input front-end (×4 spatial reduction before backbone)
  - **Factorized 3D conv ResBlocks**: 1×k×k spatial then k×1×1 axial
  - **Swish (SiLU) + LayerNorm** (not GroupNorm)
  - **Causal temporal attention** with global support (we drop this
    in the 3D port; turbulence has no time-causal axis)

Architecture port to 3D 4-channel (32^3 input patches), 64x tier:
  Front-end:
    - 3D Haar DWT level 2: (B, 4, 32, 32, 32) -> 2 levels of Haar gives
      (B, 64*4, 8, 8, 8) = 256 channels at 8^3 spatial. BUT 2 full levels
      at ×8 spatial is too aggressive for 32^3 -> would land at 4^3.
      Instead: **1 Haar level (×2)** for our smaller 32^3 patch — same as
      WF-VAE 3D port, which lets the 32^3 → 8^3 (×4) target hit cleanly.
      So front-end = 1-level Haar producing (B, 32, 16, 16, 16).
  Encoder backbone (paper-style, no KL):
    - in_conv: Conv3d 32 -> ch (default 96)
    - 1 factorized ResBlock at 16^3
    - Downsample (16 -> 8): strided 3D conv
    - 1 factorized ResBlock at 8^3
    - end_conv: norm + Swish + Conv3d -> z_channels (NO doubling — no KL!)
  Decoder mirror.
  Back-end: inverse Haar.

Default for 32^3 patches:
  ch=96, num_res_blocks=2, z_channels=4
  Spatial: 32 -> [Haar /2] -> 16 -> [conv /2] -> 8 (total /4)
  Latent: (4, 8, 8, 8) per patch = 2048 elements
  Compression ratio = 4*32^3 / (4*8^3) = 64x EXACT

Loss (paper Stage 1 + selective Stage 2, no KL, no LPIPS):
  L_G = L1(recon, target)
      + disc_weight * 3D PatchGAN hinge loss (after disc_start steps)
      + gram_weight * gram_matrix_loss(recon, target)   [Stage 2, optional]
  L_D = ½(relu(1 - D(real)) + relu(1 + D(fake)))
  Defaults: disc_weight=0.5, disc_start=6000, gram_weight=0.0 (off by default
            for first run; Stage 2 enhancement is per-paper optional)

Optimizer (paper-specific): AdamW β=(0.9, 0.99) — only knob differs from sdvae's
  Adam β=(0.5, 0.9).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse Haar implementation from the shared PPLC module.
from pplc.haar_wavelet import haar_forward_3d as haar_dwt3d, \
    haar_inverse_3d as haar_idwt3d


def swish(x):
    return F.silu(x)


class LayerNorm3d(nn.Module):
    """Spatial-only LayerNorm — matches official Cosmos tokenizer convention.

    HONEST-LN FIX (2026-06-02): normalize over spatial dims (D, H, W) per
    sample-per-channel, then apply per-channel affine. The previous
    implementation normalized over (C, D, H, W) which is non-standard:
    it collapses cross-channel statistics into a single scalar per
    sample and renders the per-channel affine partially redundant. The
    spatial-only version recovers the per-channel mean/var structure
    that the affine can then rescale, matching how the official Cosmos
    repo's LayerNorm operates.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, D, H, W) — normalize over spatial (D, H, W) per sample per channel
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


class FactorizedConv3d(nn.Module):
    """1×k×k spatial conv followed by k×1×1 axial conv.

    Cosmos paper: this factorization halves params and matches the
    causal-3D structure used in the official tokenizer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super().__init__()
        # Spatial 1xkxk (axes 3, 4)
        self.spatial = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=(1, kernel_size, kernel_size),
                                   stride=(1, stride, stride),
                                   padding=(0, padding, padding))
        # Axial kx1x1 (axis 2 = depth/temporal in video, spatial here)
        self.axial = nn.Conv3d(out_channels, out_channels,
                                kernel_size=(kernel_size, 1, 1),
                                stride=(stride, 1, 1),
                                padding=(padding, 0, 0))

    def forward(self, x):
        return self.axial(self.spatial(x))


class FactorizedResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = LayerNorm3d(in_channels)
        self.conv1 = FactorizedConv3d(in_channels, out_channels, kernel_size=3,
                                        stride=1, padding=1)
        self.norm2 = LayerNorm3d(out_channels)
        self.conv2 = FactorizedConv3d(out_channels, out_channels, kernel_size=3,
                                        stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels,
                                            kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = x
        h = self.norm1(h); h = swish(h); h = self.conv1(h)
        h = self.norm2(h); h = swish(h); h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class CosmosEncoder3d(nn.Module):
    def __init__(self, in_channels=4, ch=96, num_res_blocks=2, z_channels=4):
        super().__init__()
        self.in_channels = in_channels
        # Haar produces 8 * in_channels = 32 channels after level 1
        self.conv_in = nn.Conv3d(8 * in_channels, ch, kernel_size=3,
                                   stride=1, padding=1)
        # 1 factorized resblock at 16^3
        self.block_16 = FactorizedResBlock3d(ch, ch)
        # Downsample (16 -> 8) via strided 3D conv
        self.downsample = nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)
        # 1 factorized resblock at 8^3
        self.block_8 = FactorizedResBlock3d(ch, ch)
        # end
        self.norm_out = LayerNorm3d(ch)
        # NO double_z — pure AE, no KL
        self.conv_out = nn.Conv3d(ch, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_wavelet):
        # x_wavelet: (B, 8*in_channels, 16, 16, 16)
        h = self.conv_in(x_wavelet)
        h = self.block_16(h)
        h = self.downsample(h)  # 16 -> 8
        h = self.block_8(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)  # (B, z_channels, 8, 8, 8)
        return h


class CosmosDecoder3d(nn.Module):
    def __init__(self, out_channels=4, ch=96, num_res_blocks=2, z_channels=4):
        super().__init__()
        self.out_channels = out_channels
        self.conv_in = nn.Conv3d(z_channels, ch, kernel_size=3, stride=1, padding=1)
        self.block_8 = FactorizedResBlock3d(ch, ch)
        # Upsample (8 -> 16): nearest + conv (matches sdvae upsample style)
        self.upsample_conv = nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.block_16 = FactorizedResBlock3d(ch, ch)
        self.norm_out = LayerNorm3d(ch)
        # Produce 8 * out_channels for inverse Haar
        self.conv_out = nn.Conv3d(ch, 8 * out_channels, kernel_size=3,
                                    stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.block_8(h)
        h = F.interpolate(h, scale_factor=2.0, mode='nearest')  # 8 -> 16
        h = self.upsample_conv(h)
        h = self.block_16(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class CosmosTokenizer3D(nn.Module):
    """Paper-faithful 3D Cosmos Tokenizer CV (continuous AE, no KL).

    Forward signature matches SDVAE3D for eval-script compatibility:
      forward(x, sample_posterior=True) -> (recon, mean, logvar, z)
    Since this is a pure AE, mean == z and logvar is zeros (returned for
    interface compatibility; KL loss is 0 and the reparam is identity).
    """

    def __init__(self, in_channels=4, out_channels=4,
                 ch=96, num_res_blocks=2,
                 z_channels=4, embed_dim=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = CosmosEncoder3d(in_channels, ch, num_res_blocks, z_channels)
        self.decoder = CosmosDecoder3d(out_channels, ch, num_res_blocks, z_channels)
        # Optional 1x1x1 channel proj (matches SD-VAE quant_conv layout for API parity)
        # For pure AE these are identity-init-ish 1x1 convs.
        self.quant_conv = nn.Conv3d(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        self.z_channels = z_channels

    def encode(self, x):
        coeffs = haar_dwt3d(x)  # (B, 8*in_channels, 16, 16, 16)
        h = self.encoder(coeffs)  # (B, z_channels, 8, 8, 8)
        z = self.quant_conv(h)  # (B, embed_dim, 8, 8, 8)
        # Return (mean=z, logvar=zeros) for API parity with SDVAE3D
        logvar = torch.zeros_like(z)
        return z, logvar

    def decode(self, z):
        z = self.post_quant_conv(z)
        coeffs_recon = self.decoder(z)
        x_recon = haar_idwt3d(coeffs_recon, self.out_channels)
        return x_recon

    def forward(self, x, sample_posterior=True):
        # sample_posterior is ignored — pure AE, no sampling
        mean, logvar = self.encode(x)
        z = mean  # deterministic
        recon = self.decode(z)
        return recon, mean, logvar, z


def cosmos_loss(recon, target, mean, logvar, z, disc=None,
                disc_logits_real=None, disc_logits_fake=None,
                step=0, disc_start=6000,
                kl_weight=0.0, disc_weight=0.5, recon_weight=1.0,
                gram_weight=0.0):
    """Paper-faithful Cosmos loss: L1 + (optional Gram) + adversarial.

    NO KL — pure AE per paper.
    Returns (loss_g, loss_d, components_dict).
    """
    recon_l1 = F.l1_loss(recon, target)
    loss_g = recon_weight * recon_l1
    components = {"recon_l1": recon_l1.detach(),
                  "kl": torch.tensor(0.0, device=recon.device)}
    if gram_weight > 0.0:
        # Gram-matrix loss on raw (4-channel) features, channel-wise
        B, C = recon.shape[:2]
        r_flat = recon.reshape(B, C, -1)
        t_flat = target.reshape(B, C, -1)
        gram_r = torch.bmm(r_flat, r_flat.transpose(1, 2)) / r_flat.shape[-1]
        gram_t = torch.bmm(t_flat, t_flat.transpose(1, 2)) / t_flat.shape[-1]
        gram_l1 = F.l1_loss(gram_r, gram_t)
        loss_g = loss_g + gram_weight * gram_l1
        components["gram_l1"] = gram_l1.detach()
    loss_d = None
    if disc is not None and disc_logits_fake is not None and step >= disc_start:
        g_adv = -disc_logits_fake.mean()
        loss_g = loss_g + disc_weight * g_adv
        components["g_adv"] = g_adv.detach()
        if disc_logits_real is not None:
            loss_d = 0.5 * (F.relu(1.0 - disc_logits_real).mean()
                            + F.relu(1.0 + disc_logits_fake.detach()).mean())
            components["loss_d"] = loss_d.detach()
    return loss_g, loss_d, components
