"""Paper-faithful LTX-Video VAE 3D (Lightricks 2025).

Reference: Hacohen et al., "LTX-Video: Realtime Video Latent Diffusion"
(arXiv:2501.00103, Jan 2025). Code:
github.com/Lightricks/LTX-Video.

LTX-Video's two signature design choices:
  1. **Patchify INSIDE the VAE** (not at the downstream DiT input).
     A 1×1×1 1x1x1 patchify operates as the first encoder layer.
  2. **Decoder doubles as the final denoising step**. During training,
     Gaussian noise σ ∈ [0, 0.2] is added to the latent before decoding,
     and the decoder is trained to recover the clean recon. This makes
     the decoder a noise-robust learned prior.

Architecture port to 3D 4-channel (32^3 input patches), 64x tier:
  Encoder:
    - patchify: identity 1×1×1 conv (32^3 patch already at our patch scale)
    - in_conv: Conv3d 4 -> ch (default 96)
    - 2 strided-conv stages: 32 -> 16 -> 8 (each ×2 with ResBlock)
    - middle: ResBlock + ResBlock
    - end_conv: norm + SiLU + Conv3d -> z_channels
  Decoder:
    - conv_in: z_channels -> ch
    - 2 upsample stages: 8 -> 16 -> 32 (each ×2 with ResBlock)
    - end_conv: -> out_channels=4

Default for 32^3 patches:
  ch=96, num_res_blocks=2, z_channels=4
  Spatial: 32 -> 16 -> 8 (2 downsamples, ×4 total)
  Latent: (4, 8, 8, 8) per patch = 2048 elements
  Compression ratio = 4*32^3 / (4*8^3) = 64x EXACT

Loss (paper-faithful, LPIPS dropped):
  L_G = MSE(recon, target)
      + wavelet_weight * L1(haar_dwt3d(recon), haar_dwt3d(target))  [Video-DWT L1]
      + disc_weight * 3D PatchGAN hinge loss
      + rgan_weight * Reconstruction-GAN loss (rGAN: disc sees pairs)
  L_D = standard hinge for both adversaries
  Defaults: wavelet_weight=0.1, disc_weight=0.5, rgan_weight=0.5,
            disc_start=6000

Decoder noise augmentation:
  During training (sample_posterior=True), latent z is perturbed with
  Gaussian noise σ ∈ [0, 0.2] before decoding. At inference
  (sample_posterior=False), no noise is added.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse Haar implementation for wavelet-coefficient match loss.
from pplc.haar_wavelet import haar_forward_3d as haar_dwt3d
# Reuse standard ResBlock + Downsample + Upsample blocks from SD-VAE-3D.
from ..sdvae_3d.model import (
    ResnetBlock3d, Downsample3d, Upsample3d, Normalize, nonlinearity,
)


class LTXEncoder3d(nn.Module):
    """Patchify-inside-VAE encoder. Patchify = identity 1×1×1 at our patch scale.

    For 32³ input, 2 strided downsamples → 8³ latent (×4 spatial).
    """

    def __init__(self, in_channels=4, ch=96, num_res_blocks=2, z_channels=4):
        super().__init__()
        # Patchify layer: 1x1x1 conv (identity-like). Paper's signature: the
        # patchify operation is moved from DiT input into the VAE input.
        # At our 32^3 patch scale, this is just a learned 1x1x1 projection.
        self.patchify = nn.Conv3d(in_channels, ch, kernel_size=1, stride=1, padding=0)
        # First ResBlocks at 32^3
        self.block_32 = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks)])
        # Downsample 32 -> 16
        self.down1 = Downsample3d(ch)
        # ResBlocks at 16^3
        self.block_16 = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks)])
        # Downsample 16 -> 8
        self.down2 = Downsample3d(ch)
        # ResBlocks at 8^3
        self.block_8 = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks)])
        # middle
        self.mid_block_1 = ResnetBlock3d(ch, ch)
        self.mid_block_2 = ResnetBlock3d(ch, ch)
        # end
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv3d(ch, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.patchify(x)
        for blk in self.block_32:
            h = blk(h)
        h = self.down1(h)
        for blk in self.block_16:
            h = blk(h)
        h = self.down2(h)
        for blk in self.block_8:
            h = blk(h)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  # (B, z_channels, 8, 8, 8)
        return h


class LTXDecoder3d(nn.Module):
    """Decoder-as-denoiser. Receives optionally-noisy latent."""

    def __init__(self, out_channels=4, ch=96, num_res_blocks=2, z_channels=4):
        super().__init__()
        self.conv_in = nn.Conv3d(z_channels, ch, kernel_size=3, stride=1, padding=1)
        self.mid_block_1 = ResnetBlock3d(ch, ch)
        self.mid_block_2 = ResnetBlock3d(ch, ch)
        # Upsample 8 -> 16
        self.block_8 = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks + 1)])
        self.up1 = Upsample3d(ch)
        # ResBlocks at 16^3
        self.block_16 = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks + 1)])
        self.up2 = Upsample3d(ch)
        # ResBlocks at 32^3
        self.block_32 = nn.ModuleList([ResnetBlock3d(ch, ch) for _ in range(num_res_blocks + 1)])
        # End
        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv3d(ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        for blk in self.block_8:
            h = blk(h)
        h = self.up1(h)  # 8 -> 16
        for blk in self.block_16:
            h = blk(h)
        h = self.up2(h)  # 16 -> 32
        for blk in self.block_32:
            h = blk(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LTXVideo3D(nn.Module):
    """Paper-faithful 3D LTX-Video VAE: patchify-inside + noise-augmented decoder.

    Forward signature matches SDVAE3D for eval-script compatibility:
      forward(x, sample_posterior=True) -> (recon, mean, logvar, z)

    During training (sample_posterior=True): adds Gaussian noise σ ∈ [0, 0.2]
    to latent z before decoding. Logvar returned as zeros (no KL).
    At inference (sample_posterior=False): no noise; deterministic recon.
    """

    def __init__(self, in_channels=4, out_channels=4,
                 ch=96, num_res_blocks=2,
                 z_channels=4, embed_dim=4,
                 noise_sigma_max=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = LTXEncoder3d(in_channels, ch, num_res_blocks, z_channels)
        self.decoder = LTXDecoder3d(out_channels, ch, num_res_blocks, z_channels)
        self.quant_conv = nn.Conv3d(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        self.z_channels = z_channels
        self.noise_sigma_max = noise_sigma_max

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        logvar = torch.zeros_like(z)  # no KL, return zeros for API parity
        return z, logvar

    def decode(self, z, noise_sigma=None):
        if noise_sigma is not None and noise_sigma > 0:
            z = z + noise_sigma * torch.randn_like(z)
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        z = mean
        if sample_posterior:
            # Sample noise σ uniformly in [0, noise_sigma_max]
            sigma = float(torch.rand(1).item()) * self.noise_sigma_max
        else:
            sigma = 0.0
        recon = self.decode(z, noise_sigma=sigma)
        return recon, mean, logvar, z


class RGANDiscriminator3d(nn.Module):
    """Reconstruction-GAN discriminator: sees BOTH (recon, GT) as a pair.

    Paper-faithful rGAN concept: concat recon and GT along channel dim,
    discriminator predicts whether the first half is real or recon.
    """

    def __init__(self, in_channels=4, ndf=32, n_layers=3):
        super().__init__()
        # Input: 2 * in_channels (concat of pair)
        layers = [nn.Conv3d(2 * in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.GroupNorm(8, ndf * nf_mult), nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            nn.GroupNorm(8, ndf * nf_mult), nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * nf_mult, 1, 4, 1, 1),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, a, b):
        # a, b: each (B, C, D, H, W). Concat on channel dim.
        pair = torch.cat([a, b], dim=1)
        return self.main(pair)


def ltx_loss(recon, target, mean, logvar, z, disc=None, rgan_disc=None,
              disc_logits_real=None, disc_logits_fake=None,
              rgan_logits_real=None, rgan_logits_fake=None,
              step=0, disc_start=6000,
              kl_weight=0.0, disc_weight=0.5, rgan_weight=0.5, recon_weight=1.0,
              wavelet_weight=0.1):
    """Paper-faithful LTX-Video VAE loss: MSE + Wavelet-L1 + PatchGAN + rGAN.

    NO KL — pure AE per paper.
    Returns (loss_g, loss_d_patch, loss_d_rgan, components_dict).
    """
    # Paper: MSE pixel loss (not L1 like sdvae)
    recon_mse = F.mse_loss(recon, target)
    # For monitoring parity also compute L1
    recon_l1 = F.l1_loss(recon, target)
    # Wavelet-coefficient L1 match (paper: "Video-DWT L1")
    coeffs_recon = haar_dwt3d(recon)
    coeffs_target = haar_dwt3d(target)
    wavelet_l1 = F.l1_loss(coeffs_recon, coeffs_target)
    loss_g = recon_weight * recon_mse + wavelet_weight * wavelet_l1
    components = {"recon_l1": recon_l1.detach(),
                  "recon_mse": recon_mse.detach(),
                  "wavelet_l1": wavelet_l1.detach(),
                  "kl": torch.tensor(0.0, device=recon.device)}
    loss_d_patch = None
    loss_d_rgan = None
    if step >= disc_start:
        if disc is not None and disc_logits_fake is not None:
            g_adv = -disc_logits_fake.mean()
            loss_g = loss_g + disc_weight * g_adv
            components["g_adv"] = g_adv.detach()
            if disc_logits_real is not None:
                loss_d_patch = 0.5 * (F.relu(1.0 - disc_logits_real).mean()
                                       + F.relu(1.0 + disc_logits_fake.detach()).mean())
                components["loss_d_patch"] = loss_d_patch.detach()
        if rgan_disc is not None and rgan_logits_fake is not None:
            g_radv = -rgan_logits_fake.mean()
            loss_g = loss_g + rgan_weight * g_radv
            components["g_radv"] = g_radv.detach()
            if rgan_logits_real is not None:
                loss_d_rgan = 0.5 * (F.relu(1.0 - rgan_logits_real).mean()
                                      + F.relu(1.0 + rgan_logits_fake.detach()).mean())
                components["loss_d_rgan"] = loss_d_rgan.detach()
    return loss_g, loss_d_patch, loss_d_rgan, components
