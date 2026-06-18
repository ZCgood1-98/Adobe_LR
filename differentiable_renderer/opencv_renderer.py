#!/usr/bin/env python3
"""OpenCV/NumPy renderer that mirrors ``DifferentiableRenderer`` exactly.

Read parameters from ``params_v0.json`` (produced by ``fit.py``) and apply
them to an input PNG using only NumPy + OpenCV (with ``opencv-contrib-python``
for ``cv2.ximgproc.guidedFilter``).

The pipeline order, constants, and formulae here are kept **bit-by-bit
identical** to ``differentiable_renderer/renderer.py``; the *only* legitimate
divergence is that PyTorch ops run on GPU float32, while this version runs on
CPU float64 inside NumPy and then casts back to uint8 for saving.

Usage
-----
    python opencv_renderer.py original.png params_v0.json output.png
    python opencv_renderer.py --batch original_dir/ fit_results_demo/ rendered/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Tuple

import cv2
import numpy as np


# ============================================================
# Constants  (must match renderer.py)
# ============================================================

TEMP_NEUTRAL = 5500

MIXER_CENTERS = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 280.0, 320.0],
                         dtype=np.float64)
MIXER_WIDTHS  = np.array([40.0, 30.0, 35.0, 50.0, 35.0, 40.0, 35.0, 35.0],
                         dtype=np.float64)

# Bruce Lindbloom matrices, sRGB / D65
RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)
D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

BRADFORD = np.array([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
], dtype=np.float64)
BRADFORD_INV = np.array([
    [ 0.9869929, -0.1470543,  0.1599627],
    [ 0.4323053,  0.5183603,  0.0492912],
    [-0.0085287,  0.0400428,  0.9684867],
], dtype=np.float64)

# Guided-filter hyper-parameters (match renderer.py)
GF_HL = (8,  0.01 ** 2)
GF_SH = (8,  0.01 ** 2)
GF_CL = (16, 0.04 ** 2)
GF_TX = (4,  0.01 ** 2)


# ============================================================
# Helpers
# ============================================================

def _luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 luminance.  img: (H, W, 3) → (H, W)."""
    return (0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2])


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    low = x / 12.92
    high = np.power(np.maximum(x + a, 1e-12) / (1 + a), 2.4)
    return np.where(x <= 0.04045, low, high)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    low = 12.92 * x
    high = (1 + a) * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - a
    return np.where(x <= 0.0031308, low, high)


def cct_to_xy(T: float) -> Tuple[float, float]:
    T = max(T, 1500.0)
    inv_T = 1000.0 / T
    inv_T2 = inv_T * inv_T
    inv_T3 = inv_T2 * inv_T
    if T <= 7000.0:
        x = -4.6070 * inv_T3 + 2.9678 * inv_T2 + 0.09911 * inv_T + 0.244063
    else:
        x = -2.0064 * inv_T3 + 1.9018 * inv_T2 + 0.24748 * inv_T + 0.237040
    y = -3.000 * x * x + 2.870 * x - 0.275
    return x, y


def temp_tint_to_white_xyz(temp_K: float, tint: float) -> np.ndarray:
    x, y = cct_to_xy(temp_K)
    y = max(y - tint / 1500.0, 0.05)
    Y = 1.0
    X = x * Y / y
    Z = (1.0 - x - y) * Y / y
    return np.array([X, Y, Z], dtype=np.float64)


def bradford_cat(linear_rgb: np.ndarray,
                 src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    src_lms = BRADFORD @ src_xyz
    dst_lms = BRADFORD @ dst_xyz
    diag = dst_lms / np.maximum(src_lms, 1e-8)
    A_xyz = BRADFORD_INV @ np.diag(diag) @ BRADFORD
    full = XYZ_TO_RGB @ A_xyz @ RGB_TO_XYZ
    return linear_rgb @ full.T


def gf(guide: np.ndarray, src: np.ndarray,
       radius: int, eps: float) -> np.ndarray:
    """Guided filter wrapper.  guide is single-channel float32."""
    g = guide.astype(np.float32)
    s = src.astype(np.float32)
    return cv2.ximgproc.guidedFilter(guide=g, src=s,
                                      radius=int(radius), eps=float(eps))


# ============================================================
# Stage operators  (same names / order as renderer.py)
# ============================================================

def op_exposure(x: np.ndarray, ev: float) -> np.ndarray:
    if ev == 0.0:
        return x
    gain = 2.0 ** ev
    lin = srgb_to_linear(np.clip(x, 0.0, 1.0)) * gain
    # Clamp at 32 (= 5 EV) — must match renderer.py's _exposure to keep the
    # PyTorch and OpenCV outputs bit-identical.
    return np.clip(linear_to_srgb(np.clip(lin, 0.0, 32.0)), 0.0, 1.0)


def _pchip_curve(dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (xs, ys, m) for an 8-point monotonic cubic-Hermite curve.

    Reproduces the Fritsch-Carlson clipping in renderer._tone_curve.
    """
    n = dy.shape[0]
    xs = np.linspace(0.0, 1.0, n)
    ys = np.clip(xs + dy / 100.0, 0.0, 1.0)
    h = xs[1:] - xs[:-1]
    eps = 1e-8
    d = (ys[1:] - ys[:-1]) / np.maximum(h, eps)
    zero_seg = np.abs(d) < eps

    m_int = 0.5 * (d[:-1] + d[1:])
    m = np.concatenate([[d[0]], m_int, [d[-1]]])
    sign_flip = (d[:-1] * d[1:]) <= 0
    mask_zero_int = sign_flip | zero_seg[:-1] | zero_seg[1:]
    m_int = np.where(mask_zero_int, 0.0, m_int)
    m = np.concatenate([[m[0]], m_int, [m[-1]]])

    d_safe = np.where(zero_seg, 1.0, d)
    alpha = m[:-1] / d_safe
    beta  = m[1:]  / d_safe
    s = np.sqrt(np.maximum(alpha * alpha + beta * beta, eps))
    scale = np.minimum(3.0 / s, 1.0)
    scale_left  = np.concatenate([[1.0], scale])
    scale_right = np.concatenate([scale, [1.0]])
    m = m * np.minimum(scale_left, scale_right)
    zero_left  = np.concatenate([zero_seg, [False]])
    zero_right = np.concatenate([[False], zero_seg])
    m = np.where(zero_left | zero_right, 0.0, m)
    return xs, ys, m


def op_tone_curve(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    if np.allclose(dy, 0.0):
        return x
    n = dy.shape[0]
    xs, ys, m = _pchip_curve(dy)
    h = xs[1:] - xs[:-1]

    lum = np.maximum(_luminance(x), 1e-6)
    u = np.clip(lum, 0.0, 1.0)
    seg = np.clip(u * (n - 1), 0.0, n - 1.001)
    idx = np.clip(seg.astype(np.int64), 0, n - 2)
    t = seg - idx.astype(np.float64)

    y0 = ys[idx];  y1 = ys[idx + 1]
    m0 = m[idx];   m1 = m[idx + 1]
    h_seg = h[np.minimum(idx, n - 2)]

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    new_lum = np.clip(h00 * y0 + h10 * h_seg * m0
                      + h01 * y1 + h11 * h_seg * m1, 0.0, 1.0)

    ratio = np.clip(new_lum / lum, 0.05, 20.0)[..., None]
    return np.clip(x * ratio, 0.0, 1.0)


def op_whites(x: np.ndarray, w: float) -> np.ndarray:
    if w == 0.0:
        return x
    lum = _luminance(x)
    mask = np.clip((lum - 0.75) / 0.25, 0.0, 1.0)[..., None]
    return np.clip(x + mask * (w / 100.0) * 0.40, 0.0, 1.0)


def op_blacks(x: np.ndarray, b: float) -> np.ndarray:
    if b == 0.0:
        return x
    lum = _luminance(x)
    mask = np.clip((0.25 - lum) / 0.25, 0.0, 1.0)[..., None]
    return np.clip(x + mask * (b / 100.0) * 0.40, 0.0, 1.0)


def op_contrast(x: np.ndarray, c: float) -> np.ndarray:
    if c == 0.0:
        return x
    cc = c * 2.0
    factor = (259.0 * (cc + 255.0)) / (255.0 * (259.0 - cc) + 1e-8)
    return np.clip(factor * (x - 0.5) + 0.5, 0.0, 1.0)


def _scale_radius(base: int, short_side: int, ref: int = 512) -> int:
    """Mirror of DifferentiableRenderer._scaled_radius.  Must produce the
    same value at the same image size to keep PyTorch ↔ OpenCV outputs
    bit-identical."""
    return max(2, int(round(base * short_side / ref)))


def op_highlights(x: np.ndarray, h: float) -> np.ndarray:
    if h == 0.0:
        return x
    lum = _luminance(x).astype(np.float32)
    mask_in = np.clip((lum - 0.5) / 0.5, 0.0, 1.0).astype(np.float32)
    r = _scale_radius(GF_HL[0], min(x.shape[:2]))
    mask = gf(lum, mask_in, r, GF_HL[1]).astype(np.float64)[..., None]
    return np.clip(x + mask * (h / 100.0) * 0.50, 0.0, 1.0)


def op_shadows(x: np.ndarray, s: float) -> np.ndarray:
    if s == 0.0:
        return x
    lum = _luminance(x).astype(np.float32)
    mask_in = np.clip((0.5 - lum) / 0.5, 0.0, 1.0).astype(np.float32)
    r = _scale_radius(GF_SH[0], min(x.shape[:2]))
    mask = gf(lum, mask_in, r, GF_SH[1]).astype(np.float64)[..., None]
    return np.clip(x + mask * (s / 100.0) * 0.50, 0.0, 1.0)


def op_clarity(x: np.ndarray, c: float) -> np.ndarray:
    """Gaussian unsharp σ=4 (calibrated at 512 px short-side, scales to 16 at
    2K).  Smaller σ than the failed σ=8 attempt to keep halo width below
    the visible threshold."""
    if c == 0.0:
        return x
    sigma = max(1.0, 4.0 * min(x.shape[:2]) / 512.0)
    ks = max(3, int(6 * sigma + 1) | 1)
    base = cv2.GaussianBlur(x.astype(np.float32), (ks, ks),
                             sigmaX=sigma).astype(np.float64)
    return np.clip(x + (c / 100.0) * 1.20 * (x - base), 0.0, 1.0)


def op_texture(x: np.ndarray, t: float) -> np.ndarray:
    """Gaussian unsharp σ=2 (calibrated at 512 px → 8 at 2K)."""
    if t == 0.0:
        return x
    sigma = max(0.5, 2.0 * min(x.shape[:2]) / 512.0)
    ks = max(3, int(6 * sigma + 1) | 1)
    base = cv2.GaussianBlur(x.astype(np.float32), (ks, ks),
                             sigmaX=sigma).astype(np.float64)
    return np.clip(x + (t / 100.0) * 1.20 * (x - base), 0.0, 1.0)


def op_white_balance(x: np.ndarray, temp_K: float, tint: float) -> np.ndarray:
    if temp_K == TEMP_NEUTRAL and tint == 0.0:
        return x   # src == dst at the neutral point ⇒ exact identity
    # Source white = the neutral point on the same D-illuminant locus as the
    # target (mirror of renderer._white_balance).  Using the neutral white
    # rather than the D65 constant makes temp_K==TEMP_NEUTRAL a true no-op and
    # keeps the shortcut above consistent with the full CAT path.
    src = temp_tint_to_white_xyz(TEMP_NEUTRAL, 0.0)
    dst = temp_tint_to_white_xyz(temp_K, tint)
    lin = srgb_to_linear(np.clip(x, 0.0, 1.0))
    H, W = lin.shape[:2]
    flat = lin.reshape(-1, 3)
    out = bradford_cat(flat, src, dst).reshape(H, W, 3)
    return np.clip(linear_to_srgb(np.clip(out, 0.0, 8.0)), 0.0, 1.0)


def op_vibrance(x: np.ndarray, v: float) -> np.ndarray:
    if v == 0.0:
        return x
    max_c = np.max(x, axis=-1, keepdims=True)
    min_c = np.min(x, axis=-1, keepdims=True)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-6)
    boost = (v / 100.0) * 1.5 * (1.0 - sat)
    lum = _luminance(x)[..., None]
    return np.clip(x + boost * (x - lum), 0.0, 1.0)


def op_saturation(x: np.ndarray, s: float) -> np.ndarray:
    if s == 0.0:
        return x
    lum = _luminance(x)[..., None]
    factor = max(0.0, 1.0 + s / 100.0 * 2.0)   # clamp(min=0) → no inversion
    return np.clip(lum + factor * (x - lum), 0.0, 1.0)


def _rgb_to_hsl(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    max_c = np.max(img, axis=-1)
    min_c = np.min(img, axis=-1)
    delta = max_c - min_c
    eps = 1e-7
    L = (max_c + min_c) / 2.0
    denom = np.where(L < 0.5, np.maximum(max_c + min_c, eps),
                              np.maximum(2.0 - max_c - min_c, eps))
    S = np.where(delta < eps, 0.0, delta / denom)
    d = np.maximum(delta, eps)
    h_r = ((g - b) / d) % 6.0
    h_g = (b - r) / d + 2.0
    h_b = (r - g) / d + 4.0
    mask_r = (delta > eps) & (max_c == r)
    mask_g = (delta > eps) & (max_c == g) & ~mask_r
    mask_b = (delta > eps) & ~mask_r & ~mask_g
    H = np.where(mask_r, h_r, np.where(mask_g, h_g,
        np.where(mask_b, h_b, 0.0)))
    H = (H * 60.0) % 360.0
    return H, S, L


def _hsl_to_rgb(H: np.ndarray, S: np.ndarray, L: np.ndarray) -> np.ndarray:
    C = (1.0 - np.abs(2.0 * L - 1.0)) * S
    Hp = H / 60.0
    X = C * (1.0 - np.abs(Hp % 2.0 - 1.0))
    m = L - C / 2.0
    Z = np.zeros_like(C)
    seg = (Hp.astype(np.int64) % 6)

    R = np.where(seg == 0, C, np.where(seg == 1, X,
        np.where(seg == 4, X, np.where(seg == 5, C, Z)))) + m
    G = np.where(seg == 0, X, np.where(seg == 1, C,
        np.where(seg == 2, C, np.where(seg == 3, X, Z)))) + m
    B = np.where(seg == 2, X, np.where(seg == 3, C,
        np.where(seg == 4, C, np.where(seg == 5, X, Z)))) + m
    return np.stack([R, G, B], axis=-1)


def op_color_mixer(x: np.ndarray, mixer: list) -> np.ndarray:
    """mixer: list of 8 dicts with keys H/S/L (degrees, ±100 normalised)."""
    if not mixer or all(m.get('H', 0) == 0 and m.get('S', 0) == 0
                        and m.get('L', 0) == 0 for m in mixer):
        return x
    H, S, L = _rgb_to_hsl(x)
    sat_guard = np.clip(S / 0.10, 0.0, 1.0)
    H_adj, S_adj, L_adj = H, S, L

    for i in range(min(8, len(mixer))):
        center, halfwidth = MIXER_CENTERS[i], MIXER_WIDTHS[i]
        dist = np.abs(H - center)
        dist = np.minimum(dist, 360.0 - dist)
        inside = (dist <= halfwidth).astype(np.float64)
        hue_mask = 0.5 * (1.0 + np.cos(np.pi * dist / halfwidth)) * inside
        eff = hue_mask * sat_guard

        h_v = float(mixer[i].get('H', 0.0))
        s_v = float(mixer[i].get('S', 0.0))
        l_v = float(mixer[i].get('L', 0.0))

        H_adj = (H_adj + eff * h_v * 0.30) % 360.0
        s_shift = eff * (s_v / 100.0)
        S_adj = np.where(s_shift >= 0,
                         np.clip(S_adj + s_shift * (1.0 - S_adj), 0.0, 1.0),
                         np.clip(S_adj + s_shift * S_adj, 0.0, 1.0))
        l_shift = eff * (l_v / 200.0)
        L_adj = np.where(l_shift >= 0,
                         np.clip(L_adj + l_shift * (1.0 - L_adj), 0.0, 1.0),
                         np.clip(L_adj + l_shift * L_adj, 0.0, 1.0))
    return np.clip(_hsl_to_rgb(H_adj, S_adj, L_adj), 0.0, 1.0)


def op_color_grading(x: np.ndarray, sh: list, mid: list, hi: list) -> np.ndarray:
    if (sh == [0, 0, 0] and mid == [0, 0, 0] and hi == [0, 0, 0]):
        return x
    lum = _luminance(x)
    sh_mask  = np.clip(1.0 - lum * 2.0, 0.0, 1.0)[..., None]
    hi_mask  = np.clip(lum * 2.0 - 1.0, 0.0, 1.0)[..., None]
    mid_mask = 1.0 - sh_mask - hi_mask
    sh_rgb  = np.array(sh,  dtype=np.float64).reshape(1, 1, 3) / 100.0
    mid_rgb = np.array(mid, dtype=np.float64).reshape(1, 1, 3) / 100.0
    hi_rgb  = np.array(hi,  dtype=np.float64).reshape(1, 1, 3) / 100.0
    return np.clip(x + sh_mask * sh_rgb + mid_mask * mid_rgb
                   + hi_mask * hi_rgb, 0.0, 1.0)


def op_vignette(x: np.ndarray, v: float) -> np.ndarray:
    if v == 0.0:
        return x
    H, W = x.shape[:2]
    yy = np.linspace(-1, 1, H).reshape(H, 1)
    xx = np.linspace(-1, 1, W).reshape(1, W)
    dist = np.sqrt(xx ** 2 + yy ** 2)[..., None]
    mask = np.maximum(1.0 + (v / 100.0) * dist * 0.6, 0.0)
    return np.clip(x * mask, 0.0, 1.0)


def op_dehaze(x: np.ndarray, d: float) -> np.ndarray:
    if d == 0.0:
        return x
    mean = x.mean()
    return np.clip(mean + (x - mean) * (1.0 + d / 80.0), 0.0, 1.0)


def op_sharpen(x: np.ndarray, s: float) -> np.ndarray:
    if s == 0.0:
        return x
    blurred = cv2.GaussianBlur(x.astype(np.float32), (3, 3), sigmaX=0.5)
    blurred = blurred.astype(np.float64)
    return np.clip(x + (s / 150.0) * 1.5 * (x - blurred), 0.0, 1.0)


def op_noise_reduction(x: np.ndarray, n: float) -> np.ndarray:
    if n == 0.0:
        return x
    blurred = cv2.GaussianBlur(x.astype(np.float32), (13, 13), sigmaX=2.0)
    blurred = blurred.astype(np.float64)
    alpha = max(0.0, min(1.0, n / 100.0))
    return np.clip(x * (1.0 - alpha) + blurred * alpha, 0.0, 1.0)


# ============================================================
# Top-level pipeline
# ============================================================

# CSS rgb() → channel index, mirrors renderer._CSS_TO_CHANNEL_IDX
_CSS_TO_IDX = {
    'rgb(170, 0, 0)':     0,
    'rgb(187, 102, 0)':   1,
    'rgb(170, 170, 0)':   2,
    'rgb(0, 153, 0)':     3,
    'rgb(0, 153, 153)':   4,
    'rgb(51, 102, 221)':  5,
    'rgb(102, 51, 204)':  6,
    'rgb(170, 51, 170)':  7,
}


def _params_to_mixer_list(p: dict) -> list:
    """Convert params['color_mixer'] to a stable 8-element list ordered by
    channel index (Red → Magenta).  Accepts both list-of-dict (renderer
    format) and the dataset's CSS-keyed format.
    """
    raw = p.get('color_mixer') or []
    out = [{'H': 0.0, 'S': 0.0, 'L': 0.0} for _ in range(8)]
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # renderer.get_params_dict format: {'channel': 'Red', H, S, L}
        ch = entry.get('channel')
        if ch in ('Red', 'Orange', 'Yellow', 'Green',
                  'Aqua', 'Blue', 'Purple', 'Magenta'):
            idx = ['Red', 'Orange', 'Yellow', 'Green',
                   'Aqua', 'Blue', 'Purple', 'Magenta'].index(ch)
        else:
            css = entry.get('color', '')
            idx = _CSS_TO_IDX.get(css)
            if idx is None:
                continue
        out[idx] = {
            'H': float(entry.get('H', 0.0) or 0.0),
            'S': float(entry.get('S', 0.0) or 0.0),
            'L': float(entry.get('L', 0.0) or 0.0),
        }
    return out


def _cg_zone(p: dict, key_zh: str, key_en: str) -> list:
    cg = p.get('颜色分级') or {}
    z = cg.get(key_en) or {}
    if z:
        return [float(z.get('R', 0.0)), float(z.get('G', 0.0)),
                float(z.get('B', 0.0))]
    v = p.get(key_zh)
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return [float(v[0]), float(v[1]), float(v[2])]
    return [0.0, 0.0, 0.0]


def detail_transfer(rendered: np.ndarray, orig: np.ndarray,
                    clarity: float, texture: float) -> np.ndarray:
    """Luminance-only detail transfer (additive).

    Mirror of ``DifferentiableRenderer._detail_transfer``.  Extracts orig's
    high-frequency *luminance* content and adds the (boosted) delta uniformly
    to rendered's three channels::

        ΔY  = boost * Y_orig_hf - Y_rend_hf
        out = rendered + ΔY        # same scalar added to R, G, B

    Adding one scalar per pixel is a pure luminance shift, so chroma stays
    that of ``rendered`` (the chroma-correct edit) while detail comes from
    orig.  See the PyTorch version for the full rationale (it eliminates
    orig→rendered chroma leakage at edges, which previously broke B&W targets
    and hue-shift edits).

    Sigma scales with image short-side (16 px @ 512 ref → 64 px @ 2K).
    """
    sigma = max(2.0, 16.0 * min(orig.shape[:2]) / 512.0)
    ks = max(3, int(6 * sigma + 1) | 1)
    boost = 1.0 + (clarity + texture) / 200.0

    Y_orig = _luminance(orig).astype(np.float32)
    Y_orig_lf = cv2.GaussianBlur(Y_orig, (ks, ks),
                                  sigmaX=sigma).astype(np.float64)
    Y_orig_hf = Y_orig.astype(np.float64) - Y_orig_lf

    Y_rend = _luminance(rendered).astype(np.float32)
    Y_rend_lf = cv2.GaussianBlur(Y_rend, (ks, ks),
                                  sigmaX=sigma).astype(np.float64)
    Y_rend_hf = Y_rend.astype(np.float64) - Y_rend_lf

    delta_Y = (boost * Y_orig_hf - Y_rend_hf)[..., None]                # (H,W,1)
    return rendered.astype(np.float64) + delta_Y


def render(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply the full pipeline.

    img    : (H, W, 3) RGB uint8 *or* float in [0, 1]
    params : dict loaded from params_v0.json
    Returns: (H, W, 3) uint8 RGB
    """
    if img.dtype == np.uint8:
        x = img.astype(np.float64) / 255.0
    else:
        x = img.astype(np.float64)
    x = np.clip(x, 0.0, 1.0)
    orig = x.copy()      # keep reference for detail transfer

    p = params
    dy = np.array(p.get('tone_curve_dy') or [0.0] * 8, dtype=np.float64)
    mixer_list = _params_to_mixer_list(p)
    sh  = _cg_zone(p, 'cg_shadow',    'shadow')
    mid = _cg_zone(p, 'cg_midtone',   'midtone')
    hi  = _cg_zone(p, 'cg_highlight', 'highlight')

    # Stage 1: tone
    x = op_exposure   (x, float(p.get('曝光度', 0.0)))
    x = op_tone_curve (x, dy)
    x = op_whites     (x, float(p.get('白色色阶', 0.0)))
    x = op_blacks     (x, float(p.get('黑色色阶', 0.0)))
    x = op_contrast   (x, float(p.get('对比度',  0.0)))

    # Stage 1d: edge-aware (highlights/shadows; clarity/texture moved to
    # detail-transfer step at the end)
    x = op_highlights (x, float(p.get('高光', 0.0)))
    x = op_shadows    (x, float(p.get('阴影', 0.0)))

    # Stage 2: white balance
    x = op_white_balance(x,
        float(p.get('色温', TEMP_NEUTRAL)),
        float(p.get('色调', 0.0)))

    # Stage 3: colour
    x = op_vibrance    (x, float(p.get('自然饱和度', 0.0)))
    x = op_saturation  (x, float(p.get('饱和度',     0.0)))
    x = op_color_mixer (x, mixer_list)
    x = op_color_grading(x, sh, mid, hi)

    # Stage 4: effects
    x = op_vignette       (x, float(p.get('晕影', 0.0)))

    # Stage 5: detail
    x = op_dehaze         (x, float(p.get('去朦胧', 0.0)))
    x = op_sharpen        (x, float(p.get('锐化',   0.0)))
    x = op_noise_reduction(x, float(p.get('噪点消除', 0.0)))

    # Detail transfer: rendered low-freq + original high-freq * boost
    x = detail_transfer(x, orig,
                        float(p.get('清晰度', 0.0)),
                        float(p.get('纹理',   0.0)))

    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# ============================================================
# CLI
# ============================================================

def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _imwrite_rgb(path: str, img_rgb: np.ndarray):
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(
        description='Render an image with fitted Lightroom-style params (OpenCV).')
    ap.add_argument('image', help='input PNG/JPG (original)')
    ap.add_argument('params', help='params_v0.json from fit.py')
    ap.add_argument('output', help='output PNG path')
    args = ap.parse_args()

    with open(args.params, 'r', encoding='utf-8') as f:
        params = json.load(f)
    img = _imread_rgb(args.image)
    out = render(img, params)
    _imwrite_rgb(args.output, out)
    print(f'wrote {args.output}  ({out.shape[1]}x{out.shape[0]})')


if __name__ == '__main__':
    main()
