"""Differentiable color-space utilities (PyTorch).

Conversions and metrics needed by the new composite loss + Lightroom-correct
renderer:

  sRGB <-> linear-RGB      (IEC 61966-2-1 EOTF / OETF)
  linear-RGB <-> XYZ       (sRGB primaries, D65)
  XYZ <-> CIELAB           (D65 reference white)
  sRGB <-> LAB             (composite of the above)
  delta_e_2000(lab1, lab2) (full CIEDE2000)
  bradford_cat(...)        (chromatic adaptation in LMS)
  temp_tint_to_white_xyz   (Lightroom Kelvin/tint -> XYZ white point)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ============================================================
# sRGB <-> linear-RGB (IEC 61966-2-1)
# ============================================================

_SRGB_THRESH_FWD = 0.04045
_SRGB_THRESH_INV = 0.0031308


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB -> linear-RGB.  x in [0,1]; output may slightly exceed [0,1]."""
    a = 0.055
    low = x / 12.92
    high = ((x.clamp(min=1e-12) + a) / (1 + a)).pow(2.4)
    return torch.where(x <= _SRGB_THRESH_FWD, low, high)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """linear-RGB -> sRGB."""
    a = 0.055
    low = 12.92 * x
    high = (1 + a) * x.clamp(min=1e-12).pow(1.0 / 2.4) - a
    return torch.where(x <= _SRGB_THRESH_INV, low, high)


# ============================================================
# linear-RGB <-> XYZ (sRGB primaries, D65)
# ============================================================

# Bruce Lindbloom matrices, sRGB / D65
_RGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)
_XYZ_TO_RGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

D65_WHITE = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)


def _matmul_chw(M: torch.Tensor, img_bchw: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 matrix to a (B,3,H,W) tensor."""
    M = M.to(img_bchw.dtype).to(img_bchw.device)
    return torch.einsum('ij,bjhw->bihw', M, img_bchw)


def linear_rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    return _matmul_chw(_RGB_TO_XYZ, rgb)


def xyz_to_linear_rgb(xyz: torch.Tensor) -> torch.Tensor:
    return _matmul_chw(_XYZ_TO_RGB, xyz)


# ============================================================
# XYZ <-> CIELAB (D65)
# ============================================================

_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_3 = _LAB_DELTA ** 3


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    cube = t.clamp(min=1e-12).pow(1.0 / 3.0)
    linear = t / (3.0 * _LAB_DELTA ** 2) + 4.0 / 29.0
    return torch.where(t > _LAB_DELTA_3, cube, linear)


def _f_lab_inv(t: torch.Tensor) -> torch.Tensor:
    cube = t.pow(3.0)
    linear = 3.0 * _LAB_DELTA ** 2 * (t - 4.0 / 29.0)
    return torch.where(t > _LAB_DELTA, cube, linear)


def xyz_to_lab(xyz: torch.Tensor, white: torch.Tensor | None = None) -> torch.Tensor:
    if white is None:
        white = D65_WHITE.to(xyz.device).to(xyz.dtype)
    w = white.view(1, 3, 1, 1)
    fxyz = _f_lab(xyz / w.clamp(min=1e-12))
    fx, fy, fz = fxyz[:, 0:1], fxyz[:, 1:2], fxyz[:, 2:3]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def lab_to_xyz(lab: torch.Tensor, white: torch.Tensor | None = None) -> torch.Tensor:
    if white is None:
        white = D65_WHITE.to(lab.device).to(lab.dtype)
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = torch.cat([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], dim=1)
    return xyz * white.view(1, 3, 1, 1)


# ============================================================
# sRGB <-> LAB shorthand
# ============================================================

def srgb_to_lab(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB[0,1] (B,3,H,W) -> LAB."""
    return xyz_to_lab(linear_rgb_to_xyz(srgb_to_linear(srgb.clamp(0.0, 1.0))))


def lab_to_srgb(lab: torch.Tensor) -> torch.Tensor:
    return linear_to_srgb(xyz_to_linear_rgb(lab_to_xyz(lab))).clamp(0.0, 1.0)


# ============================================================
# CIEDE2000  (Sharma, Wu, Dalal — implemented per the original paper)
# ============================================================

def delta_e_2000(lab1: torch.Tensor, lab2: torch.Tensor,
                 kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> torch.Tensor:
    """Per-pixel CIEDE2000 between two (B,3,H,W) Lab tensors.  Returns (B,1,H,W)."""
    L1, a1, b1 = lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3]
    L2, a2, b2 = lab2[:, 0:1], lab2[:, 1:2], lab2[:, 2:3]

    eps = 1e-12
    C1 = torch.sqrt(a1 * a1 + b1 * b1 + eps)
    C2 = torch.sqrt(a2 * a2 + b2 * b2 + eps)
    Cb = (C1 + C2) / 2.0
    Cb7 = Cb.pow(7)
    # +eps inside sqrt: at zero chroma (Cb7→0) the bare sqrt has an infinite
    # derivative which, chained with Cb7's zero derivative, yields a NaN
    # gradient (inf×0).  The eps floors the sqrt argument so the gradient stays
    # finite on achromatic pixels; forward change is ~1e-12.
    G = 0.5 * (1.0 - torch.sqrt(Cb7 / (Cb7 + 25.0 ** 7) + eps))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1 + eps)
    C2p = torch.sqrt(a2p * a2p + b2 * b2 + eps)

    # Hue angles in radians, in [0, 2pi).
    # torch.atan2(0, 0) has a NaN gradient, and pure-neutral pixels (R=G=B →
    # a=b=0) land exactly on that point.  Their hue is masked out below via
    # ``zero_chroma``, but torch.where still back-propagates the NaN through
    # the *unselected* branch, poisoning every parameter's gradient — which
    # silently collapses fits of achromatic / dark targets to identity.
    # Nudging the real axis off zero keeps the gradient finite; the forward
    # shift (<1e-7 in a*) is far below uint8 quantisation.
    def _atan2_pos(y, x):
        h = torch.atan2(y, x + 1e-7)
        return torch.where(h < 0, h + 2.0 * math.pi, h)

    h1p = _atan2_pos(b1, a1p)
    h2p = _atan2_pos(b2, a2p)
    # Where C is nearly zero, h is undefined; force equal hue so delta_h vanishes
    zero_chroma = (C1p < 1e-6) | (C2p < 1e-6)
    h1p = torch.where(zero_chroma, h2p, h1p)
    h2p = torch.where(zero_chroma, h1p, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    # delta-h handling per paper
    dhp = h2p - h1p
    dhp = torch.where(dhp >  math.pi, dhp - 2.0 * math.pi, dhp)
    dhp = torch.where(dhp < -math.pi, dhp + 2.0 * math.pi, dhp)
    dHp = 2.0 * torch.sqrt(C1p * C2p + eps) * torch.sin(dhp / 2.0)

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    hdiff = (h1p - h2p).abs()
    Hbp = torch.where(
        hdiff <= math.pi,
        hsum / 2.0,
        torch.where(hsum < 2.0 * math.pi, (hsum + 2.0 * math.pi) / 2.0,
                                          (hsum - 2.0 * math.pi) / 2.0)
    )

    T = (1.0
         - 0.17 * torch.cos(Hbp - math.radians(30))
         + 0.24 * torch.cos(2.0 * Hbp)
         + 0.32 * torch.cos(3.0 * Hbp + math.radians(6))
         - 0.20 * torch.cos(4.0 * Hbp - math.radians(63)))

    dTheta = math.radians(30) * torch.exp(-((Hbp - math.radians(275)) / math.radians(25)).pow(2))
    Cbp7 = Cbp.pow(7)
    Rc = 2.0 * torch.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7) + eps)   # +eps: see G above
    Sl = 1.0 + (0.015 * (Lbp - 50.0).pow(2)) / torch.sqrt(20.0 + (Lbp - 50.0).pow(2) + eps)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -torch.sin(2.0 * dTheta) * Rc

    de = torch.sqrt(
        (dLp / (kL * Sl)).pow(2)
        + (dCp / (kC * Sc)).pow(2)
        + (dHp / (kH * Sh)).pow(2)
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
        + eps
    )
    return de


# ============================================================
# Bradford CAT (chromatic adaptation)
# ============================================================

_BRADFORD = torch.tensor([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
], dtype=torch.float32)
_BRADFORD_INV = torch.tensor([
    [ 0.9869929, -0.1470543,  0.1599627],
    [ 0.4323053,  0.5183603,  0.0492912],
    [-0.0085287,  0.0400428,  0.9684867],
], dtype=torch.float32)


def bradford_cat(linear_rgb: torch.Tensor,
                 src_white_xyz: torch.Tensor,
                 dst_white_xyz: torch.Tensor) -> torch.Tensor:
    """Apply Bradford chromatic adaptation in LMS (cone) space.

    Args
    ----
    linear_rgb     : (B,3,H,W) linear sRGB.
    src_white_xyz  : (3,) source white point in XYZ.
    dst_white_xyz  : (3,) target white point in XYZ.
    """
    M = _BRADFORD.to(linear_rgb.device).to(linear_rgb.dtype)
    Minv = _BRADFORD_INV.to(linear_rgb.device).to(linear_rgb.dtype)
    src_lms = M @ src_white_xyz.to(linear_rgb.device).to(linear_rgb.dtype)
    dst_lms = M @ dst_white_xyz.to(linear_rgb.device).to(linear_rgb.dtype)
    diag = (dst_lms / src_lms.clamp(min=1e-8))
    # Full 3x3 CAT: M_inv * diag(dst/src) * M * RGB_to_XYZ * rgb
    A_xyz = Minv @ torch.diag(diag) @ M
    rgb2xyz = _RGB_TO_XYZ.to(linear_rgb.device).to(linear_rgb.dtype)
    xyz2rgb = _XYZ_TO_RGB.to(linear_rgb.device).to(linear_rgb.dtype)
    full = xyz2rgb @ A_xyz @ rgb2xyz
    return _matmul_chw(full, linear_rgb)


# ============================================================
# Temperature / tint -> XYZ white point
# ============================================================

def cct_to_xy(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """McCamy-style approximation of CIE D-illuminant chromaticity from CCT.

    Valid for ~4000K..25000K (Lightroom range 2000..50000 is extrapolated).
    """
    T = T.clamp(min=1500.0)
    # CIE D-illuminant locus piecewise polynomial (Wyszecki & Stiles)
    inv_T = 1000.0 / T
    inv_T2 = inv_T * inv_T
    inv_T3 = inv_T2 * inv_T
    # x_d
    x_lo = -4.6070 * inv_T3 + 2.9678 * inv_T2 + 0.09911 * inv_T + 0.244063
    x_hi = -2.0064 * inv_T3 + 1.9018 * inv_T2 + 0.24748 * inv_T + 0.237040
    x = torch.where(T <= 7000.0, x_lo, x_hi)
    y = -3.000 * x * x + 2.870 * x - 0.275
    return x, y


def temp_tint_to_white_xyz(temp_K: torch.Tensor, tint: torch.Tensor) -> torch.Tensor:
    """Lightroom-style temp/tint -> XYZ white point.

    *temp_K*  scalar tensor, target colour temperature in Kelvin.
    *tint*    scalar tensor in roughly [-150, 150]; positive shifts towards
              magenta (+y_offset), negative towards green.

    Returns a (3,) XYZ tensor (Y normalised to 1).
    """
    x, y = cct_to_xy(temp_K)
    # Tint perturbs the y chromaticity perpendicular to the daylight locus.
    # Per LR convention positive tint -> magenta -> reduce y.
    y = y - tint / 1500.0
    y = y.clamp(min=0.05)
    Y = torch.ones_like(x)
    X = x * Y / y
    Z = (1.0 - x - y) * Y / y
    return torch.stack([X.squeeze(), Y.squeeze(), Z.squeeze()], dim=0)


# ============================================================
# Convenience: hue (in degrees) of an sRGB image
# ============================================================

def rgb_hue_deg(rgb: torch.Tensor) -> torch.Tensor:
    """Return per-pixel hue in degrees [0,360) without going through HSL."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    max_c = torch.max(rgb, dim=1, keepdim=True).values
    min_c = torch.min(rgb, dim=1, keepdim=True).values
    d = (max_c - min_c).clamp(min=1e-7)
    hr = ((g - b) / d) % 6.0
    hg = (b - r) / d + 2.0
    hb = (r - g) / d + 4.0
    H = torch.where(max_c == r, hr,
        torch.where(max_c == g, hg,
        torch.where(max_c == b, hb, torch.zeros_like(d))))
    H = (H * 60.0) % 360.0
    return H
