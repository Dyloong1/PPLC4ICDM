"""3D Haar wavelet transform (forward and inverse).

Zero parameters, fully invertible, differentiable. Used as the reversible
front-end inside the PPLC encoder/decoder.

Conventions:
    forward:  [B, C, D, H, W]   -> [B, 8*C, D/2, H/2, W/2]
    inverse:  [B, 8*C, D/2, H/2, W/2] -> [B, C, D, H, W]

The 8 sub-bands per input channel are ordered:
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
where L = low-pass average and H = high-pass detail, with axes (D, H, W).
"""

import torch


def haar_forward_3d(x: torch.Tensor) -> torch.Tensor:
    """3D Haar wavelet forward transform.

    Args:
        x: tensor of shape [B, C, D, H, W]. D, H, W must be even.

    Returns:
        tensor of shape [B, 8*C, D/2, H/2, W/2].
    """
    # Split along D.
    x_e = x[:, :, 0::2, :, :]
    x_o = x[:, :, 1::2, :, :]
    ld = (x_e + x_o) / 2.0
    hd = (x_e - x_o) / 2.0

    # Split along H.
    ld_le = ld[:, :, :, 0::2, :]
    ld_lo = ld[:, :, :, 1::2, :]
    hd_le = hd[:, :, :, 0::2, :]
    hd_lo = hd[:, :, :, 1::2, :]
    ll = (ld_le + ld_lo) / 2.0
    lh = (ld_le - ld_lo) / 2.0
    hl = (hd_le + hd_lo) / 2.0
    hh = (hd_le - hd_lo) / 2.0

    # Split along W.
    def split_w(t):
        te = t[:, :, :, :, 0::2]
        to = t[:, :, :, :, 1::2]
        return (te + to) / 2.0, (te - to) / 2.0

    lll, llh = split_w(ll)
    lhl, lhh = split_w(lh)
    hll, hlh = split_w(hl)
    hhl, hhh = split_w(hh)

    return torch.cat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=1)


def haar_inverse_3d(x: torch.Tensor) -> torch.Tensor:
    """3D Haar wavelet inverse transform.

    Args:
        x: tensor of shape [B, 8*C, D/2, H/2, W/2].

    Returns:
        tensor of shape [B, C, D, H, W].
    """
    B, C8, Dh, Hh, Wh = x.shape
    C = C8 // 8

    lll, llh, lhl, lhh, hll, hlh, hhl, hhh = x.chunk(8, dim=1)

    def merge_w(low, high):
        Bw, Cw, Dw, Hw, Ww = low.shape
        out = torch.empty(Bw, Cw, Dw, Hw, Ww * 2, device=low.device, dtype=low.dtype)
        out[:, :, :, :, 0::2] = low + high
        out[:, :, :, :, 1::2] = low - high
        return out

    ll = merge_w(lll, llh)
    lh = merge_w(lhl, lhh)
    hl = merge_w(hll, hlh)
    hh = merge_w(hhl, hhh)

    def merge_h(low, high):
        Bm, Cm, Dm, Hm, Wm = low.shape
        out = torch.empty(Bm, Cm, Dm, Hm * 2, Wm, device=low.device, dtype=low.dtype)
        out[:, :, :, 0::2, :] = low + high
        out[:, :, :, 1::2, :] = low - high
        return out

    ld = merge_h(ll, lh)
    hd = merge_h(hl, hh)

    out = torch.empty(B, C, Dh * 2, Hh * 2, Wh * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, :, :] = ld + hd
    out[:, :, 1::2, :, :] = ld - hd
    return out
