"""PPLC — Physics-Preserving Latent Compressor."""

from .model import PPLC, PPLCSpatial8, PPLCChannelHeavy
from .haar_wavelet import haar_forward_3d, haar_inverse_3d
from .reassemble import reassemble_naive, reassemble_hann, hann_window_3d
from .losses import (
    PatchDiscriminator3D,
    pplc_generator_loss,
    hinge_discriminator_loss,
    consistency_loss,
)

__all__ = [
    "PPLC",
    "PPLCSpatial8",
    "PPLCChannelHeavy",
    "haar_forward_3d",
    "haar_inverse_3d",
    "reassemble_naive",
    "reassemble_hann",
    "hann_window_3d",
    "PatchDiscriminator3D",
    "pplc_generator_loss",
    "hinge_discriminator_loss",
    "consistency_loss",
]
