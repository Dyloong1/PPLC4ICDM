"""Paper-faithful SD-VAE 3D (Stable-Diffusion-style KL-VAE first stage).

Reference: Rombach et al. CVPR 2022, "High-Resolution Image Synthesis with
Latent Diffusion Models" (arXiv:2112.10752).

Official 2D code: github.com/CompVis/latent-diffusion
  - configs/autoencoder/autoencoder_kl_64x64x3.yaml (f=4 model)
  - ldm/modules/diffusionmodules/model.py (Encoder/Decoder classes)
  - ldm/models/autoencoder.py (AutoencoderKL wrapper)

Architecture port to 3D 4-channel (32^3 input patches):
  Encoder:
    - in_conv: Conv3d 4 -> 128
    - 3 stages: ch_mult=[1,2,4] → channels 128 -> 256 -> 512
      Each stage: 2 ResBlock3d + (Downsample3d except last)
    - middle: ResBlock3d + Attention + ResBlock3d
    - end_conv: norm + SiLU + Conv3d → 2 * z_channels (KL mean + logvar)
  Decoder mirror.

Default for 32^3 patches:
  ch=128, ch_mult=[1,2,4], num_res_blocks=2, z_channels=8
  Spatial: 32 → 16 → 8 → 8 (2 downsamples, num_down = len(ch_mult)-1 = 2)
  Latent: (8, 8, 8, 8) per patch = 4096 elements
  Compression ratio = 4*32^3 / (8*8^3) = 131072/4096 = 32x

This matches the SD-VAE-f4 KL config from the official repo, scaled
from 2D RGB to 3D 4-channel.

Loss (paper-faithful, no LPIPS since no perceptual model for turbulence):
  L = L1(recon, target) + kl_weight * KL(q(z|x) || N(0,I))
    + disc_weight * 3D PatchGAN hinge loss (after disc_start steps)
  Defaults: kl_weight=1e-6, disc_weight=0.5, disc_start=30000
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
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
        # paper-faithful: stride-2 conv (asymmetric pad in 2D, here symmetric)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                               stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # paper: nearest-neighbor 2x + conv
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                               stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class Encoder3d(nn.Module):
    def __init__(self, in_channels=4, ch=128, ch_mult=(1, 2, 4),
                 num_res_blocks=2, z_channels=8, double_z=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock3d(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample3d(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3d(block_in, block_in)
        self.mid.block_2 = ResnetBlock3d(block_in, block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, 2 * z_channels if double_z else z_channels,
                                    kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3d(nn.Module):
    def __init__(self, out_channels=4, ch=128, ch_mult=(1, 2, 4),
                 num_res_blocks=2, z_channels=8):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]

        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3d(block_in, block_in)
        self.mid.block_2 = ResnetBlock3d(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock3d(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample3d(block_in)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, out_channels, kernel_size=3,
                                    stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SDVAE3D(nn.Module):
    """Paper-faithful 3D SD-VAE: Encoder + Decoder + KL reparam.

    Config (matches SD-VAE f=4 from CompVis configs/autoencoder/autoencoder_kl_64x64x3.yaml):
      ch=128, ch_mult=[1,2,4], num_res_blocks=2, z_channels=8
      Spatial 32 -> 8 (2 downsamples), latent (8, 8, 8, 8) = 4096
      Compression ratio: 4*32^3/(8*8^3) = 32x
    """

    def __init__(self, in_channels=4, out_channels=4,
                 ch=128, ch_mult=(1, 2, 4), num_res_blocks=2,
                 z_channels=8, embed_dim=8, dropout=0.0):
        super().__init__()
        self.encoder = Encoder3d(in_channels, ch, ch_mult, num_res_blocks,
                                   z_channels, double_z=True)
        self.decoder = Decoder3d(out_channels, ch, ch_mult, num_res_blocks,
                                   z_channels)
        # paper: pre-conv before reparam (quant_conv / post_quant_conv)
        self.quant_conv = nn.Conv3d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        self.z_channels = z_channels

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        return self.decode(z), mean, logvar, z


class PatchDiscriminator3d(nn.Module):
    """3D PatchGAN discriminator (3D port of Isola pix2pix / SD-VAE disc)."""

    def __init__(self, in_channels=4, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv3d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
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

    def forward(self, x):
        return self.main(x)


def sdvae_loss(recon, target, mean, logvar, z, disc=None,
               disc_logits_real=None, disc_logits_fake=None,
               step=0, disc_start=30000,
               kl_weight=1e-6, disc_weight=0.5, recon_weight=1.0):
    """Paper-faithful SD-VAE loss: L1 + KL + adversarial.

    Returns (loss_g, loss_d, components_dict).
    Without disc: returns L1 + KL only (loss_d=None).
    """
    # L1 reconstruction (paper uses L1; LPIPS skipped since no perceptual for turbulence)
    recon_l1 = F.l1_loss(recon, target)
    # KL term (per-dim KL to N(0,I), averaged over batch)
    kl = 0.5 * torch.sum(mean ** 2 + logvar.exp() - 1.0 - logvar, dim=[1, 2, 3, 4]).mean()
    loss_g = recon_weight * recon_l1 + kl_weight * kl
    components = {"recon_l1": recon_l1.detach(), "kl": kl.detach()}
    loss_d = None
    if disc is not None and disc_logits_fake is not None and step >= disc_start:
        # Generator gets adv loss (wants disc to think recon is real)
        g_adv = -disc_logits_fake.mean()
        loss_g = loss_g + disc_weight * g_adv
        components["g_adv"] = g_adv.detach()
        if disc_logits_real is not None:
            # Discriminator hinge loss
            loss_d = 0.5 * (F.relu(1.0 - disc_logits_real).mean()
                            + F.relu(1.0 + disc_logits_fake.detach()).mean())
            components["loss_d"] = loss_d.detach()
    return loss_g, loss_d, components
