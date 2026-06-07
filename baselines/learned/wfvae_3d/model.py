"""Paper-faithful WF-VAE 3D (Wavelet-Flow VAE).

Reference: Li et al. CVPR 2025, "WF-VAE: Enhancing Video VAE by
Wavelet-Driven Energy Flow" (arXiv:2411.17459).

Official 2D+T code: github.com/PKU-YuanGroup/WF-VAE
  - The model uses a 3D Haar wavelet front-end (3 levels in the paper)
    with a multi-scale "energy-flow" path: lower-frequency sub-bands
    are injected into the decoder at matching depths via concat,
    bypassing the conv-only path and preserving multi-scale structure.

Architecture port to 3D 4-channel (32^3 input patches), 64x tier:
  Front-end:
    - 3D Haar DWT level 1: input (B, 4, 32, 32, 32) -> 8 sub-bands of
      (B, 4, 16, 16, 16) each (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH).
      Concatenated along channel: (B, 32, 16, 16, 16).
    - Energy-flow extraction: keep low-freq band (LLL only, 4 ch) as
      energy_low for decoder injection.
  Encoder backbone:
    - in_conv: Conv3d 32 -> ch (default 128)
    - 1 strided-conv stage (16 -> 8 spatial): 2 ResBlock3d + Downsample3d
    - middle: ResBlock3d + ResBlock3d
    - end_conv: norm + SiLU + Conv3d -> 2 * z_channels (KL mean + logvar)
  Decoder mirror, with energy-flow injection (concat LLL low-freq band
  into decoder at matching depth).
  Back-end:
    - Inverse 3D Haar to recover (B, 4, 32, 32, 32).

Default for 32^3 patches:
  ch=128, z_channels=4, embed_dim=4, num_res_blocks=2
  Spatial: 32 -> [Haar /2] -> 16 -> [conv /2] -> 8 (total /4)
  Latent: (4, 8, 8, 8) per patch = 2048 elements
  Compression ratio = 4*32^3 / (4*8^3) = 131072/2048 = 64x EXACT

Loss (paper-faithful, no LPIPS since no perceptual model for turbulence):
  L = L1(recon, target) + kl_weight * KL(q(z|x) || N(0,I))
    + wavelet_weight * L1(wavelet_coeffs(recon) - wavelet_coeffs(target))
    + disc_weight * 3D PatchGAN hinge loss (after disc_start steps)
  Defaults: kl_weight=1e-6, wavelet_weight=0.1 (paper-faithful λ_WL),
            disc_weight=0.5, disc_start=6000 (matches sdvae)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
    # Allow num_groups to shrink if in_channels is small (e.g., 4)
    if in_channels < num_groups:
        num_groups = max(1, in_channels // 4) if in_channels >= 4 else 1
    if in_channels % num_groups != 0:
        # find largest divisor <= 32
        for g in (32, 16, 8, 4, 2, 1):
            if in_channels % g == 0:
                num_groups = g
                break
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels,
                         eps=1e-6, affine=True)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels,
                                            kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = x
        h = self.norm1(h); h = nonlinearity(h); h = self.conv1(h)
        h = self.norm2(h); h = nonlinearity(h); h = self.dropout(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class Downsample3d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                               stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                               stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


def haar_dwt3d(x):
    """3D Haar wavelet forward transform, 1 level.

    Input: (B, C, D, H, W) — D, H, W must each be even.
    Output: (B, 8*C, D/2, H/2, W/2) — 8 sub-bands concatenated in channel.

    Sub-band order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
    (axis order: depth, height, width; L=low/sum, H=high/diff).

    Normalization: 1/sqrt(8) per sub-band so inverse is exact.
    """
    # Pair-wise add/sub along each axis
    # Step 1: along depth
    xL = (x[:, :, 0::2] + x[:, :, 1::2]) / math.sqrt(2.0)  # (B,C,D/2,H,W)
    xH = (x[:, :, 0::2] - x[:, :, 1::2]) / math.sqrt(2.0)
    # Step 2: along height
    xLL = (xL[:, :, :, 0::2] + xL[:, :, :, 1::2]) / math.sqrt(2.0)
    xLH = (xL[:, :, :, 0::2] - xL[:, :, :, 1::2]) / math.sqrt(2.0)
    xHL = (xH[:, :, :, 0::2] + xH[:, :, :, 1::2]) / math.sqrt(2.0)
    xHH = (xH[:, :, :, 0::2] - xH[:, :, :, 1::2]) / math.sqrt(2.0)
    # Step 3: along width
    xLLL = (xLL[:, :, :, :, 0::2] + xLL[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xLLH = (xLL[:, :, :, :, 0::2] - xLL[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xLHL = (xLH[:, :, :, :, 0::2] + xLH[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xLHH = (xLH[:, :, :, :, 0::2] - xLH[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xHLL = (xHL[:, :, :, :, 0::2] + xHL[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xHLH = (xHL[:, :, :, :, 0::2] - xHL[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xHHL = (xHH[:, :, :, :, 0::2] + xHH[:, :, :, :, 1::2]) / math.sqrt(2.0)
    xHHH = (xHH[:, :, :, :, 0::2] - xHH[:, :, :, :, 1::2]) / math.sqrt(2.0)
    return torch.cat([xLLL, xLLH, xLHL, xLHH, xHLL, xHLH, xHHL, xHHH], dim=1)


def haar_idwt3d(coeffs, in_channels):
    """3D Haar wavelet inverse transform, 1 level.

    Input: (B, 8*C, D/2, H/2, W/2) — concatenated sub-bands.
    Output: (B, C, D, H, W).
    """
    C = in_channels
    xLLL = coeffs[:, 0*C:1*C]; xLLH = coeffs[:, 1*C:2*C]
    xLHL = coeffs[:, 2*C:3*C]; xLHH = coeffs[:, 3*C:4*C]
    xHLL = coeffs[:, 4*C:5*C]; xHLH = coeffs[:, 5*C:6*C]
    xHHL = coeffs[:, 6*C:7*C]; xHHH = coeffs[:, 7*C:8*C]

    # Inverse along width
    xLL = torch.stack([
        (xLLL + xLLH) / math.sqrt(2.0),
        (xLLL - xLLH) / math.sqrt(2.0)], dim=-1)
    xLL = xLL.reshape(xLLL.shape[0], C, xLLL.shape[2], xLLL.shape[3], xLLL.shape[4]*2)
    xLH = torch.stack([
        (xLHL + xLHH) / math.sqrt(2.0),
        (xLHL - xLHH) / math.sqrt(2.0)], dim=-1)
    xLH = xLH.reshape(xLHL.shape[0], C, xLHL.shape[2], xLHL.shape[3], xLHL.shape[4]*2)
    xHL = torch.stack([
        (xHLL + xHLH) / math.sqrt(2.0),
        (xHLL - xHLH) / math.sqrt(2.0)], dim=-1)
    xHL = xHL.reshape(xHLL.shape[0], C, xHLL.shape[2], xHLL.shape[3], xHLL.shape[4]*2)
    xHH = torch.stack([
        (xHHL + xHHH) / math.sqrt(2.0),
        (xHHL - xHHH) / math.sqrt(2.0)], dim=-1)
    xHH = xHH.reshape(xHHL.shape[0], C, xHHL.shape[2], xHHL.shape[3], xHHL.shape[4]*2)

    # Inverse along height
    xL = torch.stack([
        (xLL + xLH) / math.sqrt(2.0),
        (xLL - xLH) / math.sqrt(2.0)], dim=-2)
    xL = xL.reshape(xLL.shape[0], C, xLL.shape[2], xLL.shape[3]*2, xLL.shape[4])
    xH = torch.stack([
        (xHL + xHH) / math.sqrt(2.0),
        (xHL - xHH) / math.sqrt(2.0)], dim=-2)
    xH = xH.reshape(xHL.shape[0], C, xHL.shape[2], xHL.shape[3]*2, xHL.shape[4])

    # Inverse along depth
    x = torch.stack([
        (xL + xH) / math.sqrt(2.0),
        (xL - xH) / math.sqrt(2.0)], dim=-3)
    x = x.reshape(xL.shape[0], C, xL.shape[2]*2, xL.shape[3], xL.shape[4])
    return x


class WFEncoder3d(nn.Module):
    def __init__(self, in_channels=4, ch=128, num_res_blocks=2, z_channels=4):
        super().__init__()
        self.in_channels = in_channels
        # Haar produces 8 * in_channels = 32 channels after level 1
        self.conv_in = nn.Conv3d(8 * in_channels, ch, kernel_size=3, stride=1, padding=1)
        # One stage of ResBlocks at 16^3 spatial
        block = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks)])
        self.block = block
        self.downsample = Downsample3d(ch)  # 16 -> 8
        # middle
        self.mid_block_1 = ResnetBlock3d(ch, ch)
        self.mid_block_2 = ResnetBlock3d(ch, ch)
        # end
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv3d(ch, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_wavelet):
        # x_wavelet: (B, 8*in_channels, 16, 16, 16) — concatenated Haar sub-bands
        h = self.conv_in(x_wavelet)
        for blk in self.block:
            h = blk(h)
        h = self.downsample(h)  # 16 -> 8
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  # (B, 2*z_channels, 8, 8, 8)
        return h


class WFDecoder3d(nn.Module):
    """Decoder with internally-derived energy-flow path.

    HONEST-64x FIX (2026-06-02): the original WF-VAE paper injects the
    low-frequency wavelet sub-band (LLL) of the INPUT into the decoder.
    Doing so requires storing the LLL band on disk in addition to the
    latent, which destroys the claimed compression ratio (latent 2048
    + LLL 16384 = 18432 floats per 32^3 patch → 7.1x real ratio, NOT
    64x). For an apples-to-apples 64x baseline we instead derive a
    "synthetic energy-flow" path INTERNALLY from the latent z via a
    small learned upsample+conv branch. The multi-scale architectural
    bias (paper's signature) is preserved — the only difference is
    that the low-freq side-info is RECOVERED from z, not transmitted.
    Decode only needs z.
    """

    def __init__(self, out_channels=4, ch=128, num_res_blocks=2, z_channels=4):
        super().__init__()
        self.out_channels = out_channels
        self.ch = ch
        self.conv_in = nn.Conv3d(z_channels, ch, kernel_size=3, stride=1, padding=1)
        self.mid_block_1 = ResnetBlock3d(ch, ch)
        self.mid_block_2 = ResnetBlock3d(ch, ch)
        self.upsample = Upsample3d(ch)  # 8 -> 16
        # Synthetic energy-flow branch (derived from z):
        #   z (z_channels, 8^3) -> nearest x2 -> Conv3d -> (out_channels, 16^3)
        # This gives the decoder a low-freq prior at 16^3 spatial resolution
        # without requiring side-info storage.
        self.energy_synth = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv3d(z_channels, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, stride=1, padding=1),
        )
        # After upsample: concat with synthetic energy-flow low-freq band
        self.flow_conv = nn.Conv3d(ch + out_channels, ch, kernel_size=3, stride=1, padding=1)
        # ResBlocks at 16^3
        self.block = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks + 1)])
        # End: produce 8 * out_channels for inverse Haar
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv3d(ch, 8 * out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z: (B, z_channels, 8, 8, 8)
        # Synthesize energy-flow path internally — no external side-info needed.
        energy_low = self.energy_synth(z)  # (B, out_channels, 16, 16, 16)
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.upsample(h)  # 8 -> 16
        # Energy-flow injection (now from synthesized branch, not input bypass)
        h = torch.cat([h, energy_low], dim=1)
        h = self.flow_conv(h)
        for blk in self.block:
            h = blk(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  # (B, 8*out_channels, 16, 16, 16)
        return h


class WFVAE3D(nn.Module):
    """Paper-faithful 3D WF-VAE: Haar front-end + energy-flow VAE.

    Forward signature matches SDVAE3D for eval-script compatibility:
      forward(x, sample_posterior=True) -> (recon, mean, logvar, z)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 ch=128, num_res_blocks=2,
                 z_channels=4, embed_dim=4, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = WFEncoder3d(in_channels, ch, num_res_blocks, z_channels)
        self.decoder = WFDecoder3d(out_channels, ch, num_res_blocks, z_channels)
        self.quant_conv = nn.Conv3d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        self.z_channels = z_channels

    def encode(self, x):
        # Haar wavelet front-end (architectural prior; only the latent is stored)
        coeffs = haar_dwt3d(x)  # (B, 8*in_channels, 16, 16, 16)
        h = self.encoder(coeffs)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z):
        # Honest-64x: decode takes ONLY the latent. Energy-flow path is
        # synthesized internally inside WFDecoder3d (no side-info storage).
        z = self.post_quant_conv(z)
        coeffs_recon = self.decoder(z)
        # Inverse Haar
        x_recon = haar_idwt3d(coeffs_recon, self.out_channels)
        return x_recon

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        recon = self.decode(z)
        return recon, mean, logvar, z


def wfvae_loss(recon, target, mean, logvar, z, disc=None,
               disc_logits_real=None, disc_logits_fake=None,
               step=0, disc_start=6000,
               kl_weight=1e-6, disc_weight=0.5, recon_weight=1.0,
               wavelet_weight=0.1):
    """Paper-faithful WF-VAE loss: L1 + KL + wavelet-coeff match + adversarial.

    Returns (loss_g, loss_d, components_dict).
    """
    recon_l1 = F.l1_loss(recon, target)
    kl = 0.5 * torch.sum(mean ** 2 + logvar.exp() - 1.0 - logvar,
                          dim=[1, 2, 3, 4]).mean()
    # Wavelet-coefficient match (paper signature term: L1 in wavelet domain)
    coeffs_recon = haar_dwt3d(recon)
    coeffs_target = haar_dwt3d(target)
    wavelet_l1 = F.l1_loss(coeffs_recon, coeffs_target)
    loss_g = recon_weight * recon_l1 + kl_weight * kl + wavelet_weight * wavelet_l1
    components = {"recon_l1": recon_l1.detach(), "kl": kl.detach(),
                  "wavelet_l1": wavelet_l1.detach()}
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
