#!/usr/bin/env python3
"""
differentiable_renderer/renderer.py
====================================
Differentiable PyTorch renderer for Lightroom-style photo adjustments on PNG.

This module is a 1:1 PyTorch translation of the NumPy renderer in
``step3_assr_cot.py``, enabling gradient-based parameter optimisation via::

    theta_png = argmin_theta  L( Render(img_orig, theta), img_edited )

Design principles (inherited from step3):
  - Entire pipeline operates in **float32 sRGB [0,1]** — no gamma round-trip.
  - Every ``if param == 0: return`` guard is removed so gradients always flow.
  - Operations that use spatial filtering (clarity / texture / sharpening /
    noise reduction) pre-compute separable Gaussian kernels as registered
    buffers for efficiency across many forward passes.

Supported parameters  (57 total):
    Scalar (16):
        曝光度, 对比度, 高光, 阴影, 白色色阶, 黑色色阶,
        色温 (relative), 色调,
        自然饱和度, 饱和度,
        纹理, 清晰度, 去朦胧,
        晕影, 锐化, 噪点消除
    Color mixer (24):
        8 channels  (Red / Orange / Yellow / Green / Aqua / Blue / Purple / Magenta)
        × 3 axes   (H / S / L)
    Color grading (9):
        shadow_rgb (3), midtone_rgb (3), highlight_rgb (3)
    Tone curve (8):
        tone_curve_dy (8 monotonic-PCHIP control-point deltas)

Pipeline order:
    Stage 1  曝光度 → 白色色阶 → 黑色色阶 → 高光 → 阴影 → 对比度
    Stage 2  色温 → 色调
    Stage 3  自然饱和度 → 饱和度 → color_mixer → 颜色分级
    Stage 4  晕影
    Stage 5  纹理 → 清晰度 → 去朦胧 → 锐化 → 噪点消除
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .color_spaces import (
    srgb_to_linear, linear_to_srgb,
    bradford_cat, temp_tint_to_white_xyz,
)
from .guided_filter import guided_filter

# ============================================================
# Constants
# ============================================================

TEMP_NEUTRAL = 5500  # daylight neutral (K), same as step3

# Lightroom colour-mixer channel definitions (same as step3 _MIXER_CHANNELS)
MIXER_NAMES   = ['Red', 'Orange', 'Yellow', 'Green',
                 'Aqua', 'Blue', 'Purple', 'Magenta']
MIXER_CENTERS = [0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 280.0, 320.0]   # degrees
MIXER_WIDTHS  = [40.0, 30.0, 35.0, 50.0,  35.0,  40.0,  35.0,  35.0]   # degrees

# CSS border-color → channel index mapping (for dataset interop)
_CSS_TO_CHANNEL_IDX = {
    'rgb(170, 0, 0)':     0,   # Red
    'rgb(187, 102, 0)':   1,   # Orange
    'rgb(170, 170, 0)':   2,   # Yellow
    'rgb(0, 153, 0)':     3,   # Green
    'rgb(0, 153, 153)':   4,   # Aqua
    'rgb(51, 102, 221)':  5,   # Blue
    'rgb(102, 51, 204)':  6,   # Purple
    'rgb(170, 51, 170)':  7,   # Magenta
}


# ============================================================
# Utilities
# ============================================================

def _luminance(img: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance.  (B,3,H,W) → (B,1,H,W)."""
    return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]


def _make_gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    """1-D Gaussian kernel (float32, CPU)."""
    ks = max(3, int(6 * sigma + 1) | 1)
    x = torch.arange(ks, dtype=torch.float32) - ks // 2
    k = torch.exp(-0.5 * x ** 2 / (sigma ** 2 + 1e-8))
    return k / k.sum()


def _gaussian_blur(img: torch.Tensor,
                   kernel_1d: torch.Tensor) -> torch.Tensor:
    """Separable 2-D Gaussian blur.  img: (B,C,H,W)."""
    C = img.shape[1]
    ks = kernel_1d.shape[0]
    kh = kernel_1d.view(1, 1, 1, ks).expand(C, -1, -1, -1)
    kv = kernel_1d.view(1, 1, ks, 1).expand(C, -1, -1, -1)
    out = F.pad(img, [ks // 2, ks // 2, 0, 0], mode='reflect')
    out = F.conv2d(out, kh, groups=C)
    out = F.pad(out, [0, 0, ks // 2, ks // 2], mode='reflect')
    out = F.conv2d(out, kv, groups=C)
    return out


# ---- Differentiable RGB ↔ HSL (port of step3 _rgb_to_hsl / _hsl_to_rgb) ----

def _rgb_to_hsl(img: torch.Tensor):
    """
    (B,3,H,W) sRGB [0,1] → (H, S, L) each (B,1,H,W).
    H in [0, 360], S and L in [0, 1].
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]

    max_c = torch.max(img, dim=1, keepdim=True).values
    min_c = torch.min(img, dim=1, keepdim=True).values
    delta = max_c - min_c
    eps = 1e-7

    L = (max_c + min_c) / 2.0
    denom = torch.where(L < 0.5,
                        (max_c + min_c).clamp(min=eps),
                        (2.0 - max_c - min_c).clamp(min=eps))
    S = torch.where(delta < eps, torch.zeros_like(delta), delta / denom)

    d = delta.clamp(min=eps)
    h_r = ((g - b) / d) % 6.0
    h_g = (b - r) / d + 2.0
    h_b = (r - g) / d + 4.0

    mask_r = (delta > eps) & (max_c == r)
    mask_g = (delta > eps) & (max_c == g) & ~mask_r
    mask_b = (delta > eps) & ~mask_r & ~mask_g

    H = torch.where(mask_r, h_r,
        torch.where(mask_g, h_g,
        torch.where(mask_b, h_b, torch.zeros_like(d))))
    H = (H * 60.0) % 360.0

    return H, S, L


def _hsl_to_rgb(H: torch.Tensor,
                S: torch.Tensor,
                L: torch.Tensor) -> torch.Tensor:
    """
    HSL → sRGB [0,1].  H (B,1,H,W) in [0,360], S/L in [0,1].
    Returns (B,3,H,W).
    """
    C = (1.0 - (2.0 * L - 1.0).abs()) * S
    H_prime = H / 60.0
    X = C * (1.0 - (H_prime % 2.0 - 1.0).abs())
    m = L - C / 2.0
    Z = torch.zeros_like(C)

    seg = H_prime.long() % 6

    R = torch.where(seg == 0, C,
        torch.where(seg == 1, X,
        torch.where(seg == 4, X,
        torch.where(seg == 5, C, Z)))) + m
    G = torch.where(seg == 0, X,
        torch.where(seg == 1, C,
        torch.where(seg == 2, C,
        torch.where(seg == 3, X, Z)))) + m
    B = torch.where(seg == 2, X,
        torch.where(seg == 3, C,
        torch.where(seg == 4, C,
        torch.where(seg == 5, X, Z)))) + m

    return torch.cat([R, G, B], dim=1)


# ============================================================
# DifferentiableRenderer
# ============================================================

class DifferentiableRenderer(nn.Module):
    """
    Fully differentiable image processing pipeline.

    Every operation mirrors the corresponding NumPy function in
    ``step3_assr_cot.py`` (same coefficients, same sRGB-native design).
    All 57 parameters are ``nn.Parameter`` initialised to 0 (identity / no-op).

    Usage::

        renderer = DifferentiableRenderer().cuda()
        img = load_as_tensor(path)          # (1, 3, H, W) float [0,1]
        out = renderer(img)                 # differentiable forward
        loss = F.l1_loss(out, target)
        loss.backward()                     # gradients flow to all 57 params
    """

    def __init__(self):
        super().__init__()

        # ---- Stage 1: basic tone ----
        self.exposure   = nn.Parameter(torch.tensor(0.0))
        self.contrast   = nn.Parameter(torch.tensor(0.0))
        self.highlights  = nn.Parameter(torch.tensor(0.0))
        self.shadows    = nn.Parameter(torch.tensor(0.0))
        self.whites     = nn.Parameter(torch.tensor(0.0))
        self.blacks     = nn.Parameter(torch.tensor(0.0))

        # ---- Stage 2: white balance ----
        # temp is stored as *relative*: (temp_K − 5500) / 50
        # so 0.0 → 5500 K;  optimiser can push freely in ℝ
        self.temp = nn.Parameter(torch.tensor(0.0))
        self.tint = nn.Parameter(torch.tensor(0.0))

        # ---- Stage 3: colour ----
        self.vibrance   = nn.Parameter(torch.tensor(0.0))
        self.saturation = nn.Parameter(torch.tensor(0.0))
        self.mixer_hue  = nn.Parameter(torch.zeros(8))
        self.mixer_sat  = nn.Parameter(torch.zeros(8))
        self.mixer_lum  = nn.Parameter(torch.zeros(8))

        # ---- Stage 3b: color grading (shadow / midtone / highlight RGB offsets) ----
        self.cg_shadow    = nn.Parameter(torch.zeros(3))   # RGB offsets
        self.cg_midtone   = nn.Parameter(torch.zeros(3))
        self.cg_highlight = nn.Parameter(torch.zeros(3))

        # ---- Stage 4: effects ----
        self.vignette = nn.Parameter(torch.tensor(0.0))

        # ---- Stage 5: detail ----
        self.texture    = nn.Parameter(torch.tensor(0.0))
        self.clarity    = nn.Parameter(torch.tensor(0.0))
        self.dehaze     = nn.Parameter(torch.tensor(0.0))
        self.sharpening = nn.Parameter(torch.tensor(0.0))
        self.noise_red  = nn.Parameter(torch.tensor(0.0))

        # ---- Tone curve: 8-point monotonic cubic Hermite ----
        # x positions are fixed at 0, 1/7, 2/7, ..., 1.
        # y values are *deltas* from identity so 0 = no-op everywhere.
        self.tone_curve_dy = nn.Parameter(torch.zeros(8))

        # ---- Guided-filter hyper-parameters (edge-aware ops) ----
        # All four ops below use cv2.ximgproc.guidedFilter-equivalent maths so
        # the params transfer 1:1 to a NumPy/OpenCV renderer at deploy time.
        self._gf_hl_radius = 8;   self._gf_hl_eps = 0.01 ** 2   # highlights mask
        self._gf_sh_radius = 8;   self._gf_sh_eps = 0.01 ** 2   # shadows mask
        self._gf_cl_radius = 16;  self._gf_cl_eps = 0.04 ** 2   # clarity (mid)
        self._gf_tx_radius = 4;   self._gf_tx_eps = 0.01 ** 2   # texture (high)

        # Pre-compute blur kernels for sharpen + denoise (these stay pixel-level)
        self.register_buffer('_k_sharpen', _make_gaussian_kernel_1d(0.5))
        self.register_buffer('_k_denoise', _make_gaussian_kernel_1d(2.0))

        # Mixer geometry (buffers, not parameters)
        self.register_buffer('_mixer_centers',
                             torch.tensor(MIXER_CENTERS, dtype=torch.float32))
        # Half-widths used by the cosine kernel (full half-width, not sigma)
        self.register_buffer('_mixer_halfwidth',
                             torch.tensor([w for w in MIXER_WIDTHS],
                                          dtype=torch.float32))

        # Black-and-white mode: when True, clamp_params() forces chroma-adding
        # parameters to neutral after every optimiser step.  Set externally
        # by the fit pipeline after detect_bw() returns True.
        self.bw_mode = False

    # ================================================================
    # Forward
    # ================================================================

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the full pipeline.

        Args:
            img: (B, 3, H, W)  sRGB float [0, 1]
        Returns:
            (B, 3, H, W)  sRGB float [0, 1]
        """
        x = img

        # Stage 1a — linear-light exposure (sensor-correct)
        x = self._exposure(x)

        # Stage 1b — parametric tone curve (monotonic cubic Hermite)
        x = self._tone_curve(x)

        # Stage 1c — endpoint shaping (still pixel-level: whites / blacks)
        x = self._whites(x)
        x = self._blacks(x)
        x = self._contrast(x)

        # Stage 1d — luminance-zone tonal ops (still useful for tone shaping;
        # clarity/texture moved to detail-transfer step below).
        x = self._highlights_op(x)
        x = self._shadows_op(x)

        # Stage 2 — white balance via Bradford chromatic adaptation
        x = self._white_balance(x)

        # Stage 3 — colour
        x = self._vibrance(x)
        x = self._saturation(x)
        x = self._color_mixer(x)
        x = self._color_grading(x)

        # Stage 4 — effects
        x = self._vignette_op(x)

        # Stage 5 — detail (the remaining pixel-level operators)
        x = self._dehaze_op(x)
        x = self._sharpen(x)
        x = self._noise_reduction(x)

        # ── Detail transfer (frequency separation) ──
        # The renderer's pipeline is good at colour/tone but unavoidably
        # loses fine detail through repeated re-quantisation, clamping, and
        # smoothing.  Real LR doesn't have this problem because it operates
        # in linear RAW.  As a workaround we extract high-frequency content
        # from the *original* image and graft it onto the renderer's
        # low-frequency output.  Result: rendered tone & colour + original
        # crispness.  ``clarity``/``texture`` now act as multipliers on the
        # transferred detail (1 + (c+t)/200), so they range from
        # 1× (preserve original detail) to 2× (double detail).
        return self._detail_transfer(x, img).clamp(0.0, 1.0)

    # ================================================================
    # Stage 1 — basic tone
    # ================================================================

    def _exposure(self, x):
        """Lightroom-correct exposure: multiplicative in **linear RGB**.

        Lightroom (and any physically-meaningful camera pipeline) applies
        exposure as a linear-light gain of 2^EV before the display compander.
        Doing it directly in sRGB (the legacy implementation) crushes shadows
        for negative EV and over-clips highlights for positive EV.

        The internal linear-domain clamp is set generously to 32 (= 5 EV
        headroom).  At 8.0 (the previous value) bright pixels at exposure
        ≥ +1.5 EV would saturate before downstream operators (highlights,
        tone curve, clarity) had a chance to compress them — destroying
        cloud / sky detail irrecoverably.  32.0 still rejects pathological
        runaway gain while preserving the full LR ±5 EV range.
        """
        gain = 2.0 ** self.exposure
        lin = srgb_to_linear(x.clamp(0.0, 1.0)) * gain
        return linear_to_srgb(lin.clamp(min=0.0, max=32.0)).clamp(0.0, 1.0)

    def _tone_curve(self, x):
        """Monotonic cubic-Hermite (PCHIP) tone curve on luminance.

        8 control points at fixed x = 0, 1/7, ..., 1.
        Learnable y-values are *deltas from identity* so 0 everywhere is a
        perfect no-op.  Fritsch-Carlson slope clipping (vectorised) keeps
        the interpolant monotonic for any δy.
        """
        device, dtype = x.device, x.dtype
        n = self.tone_curve_dy.shape[0]
        xs = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        ys = (xs + self.tone_curve_dy.to(dtype) / 100.0).clamp(0.0, 1.0)

        eps = 1e-8
        h = xs[1:] - xs[:-1]                                 # (n-1,)
        d = (ys[1:] - ys[:-1]) / h.clamp(min=eps)            # secant slopes
        zero_seg = d.abs() < eps                             # flat segments

        # Initial tangents (centered differences interior, one-sided ends)
        m_int = 0.5 * (d[:-1] + d[1:])                       # (n-2,)
        m = torch.cat([d[:1], m_int, d[-1:]], dim=0)         # (n,)

        # Where two adjacent segments have opposite signs OR a segment is flat,
        # force the shared tangent to zero (PCHIP rule).
        sign_flip = (d[:-1] * d[1:]) <= 0
        m_int_zero = sign_flip | zero_seg[:-1] | zero_seg[1:]
        m = torch.cat([
            m[:1],
            torch.where(m_int_zero, torch.zeros_like(m_int), m_int),
            m[-1:]
        ], dim=0)

        # Fritsch-Carlson clip: per segment k, scale (m[k], m[k+1]) so
        # alpha^2 + beta^2 <= 9, where alpha = m[k]/d[k], beta = m[k+1]/d[k].
        d_safe = torch.where(zero_seg, torch.ones_like(d), d)
        alpha = m[:-1] / d_safe
        beta  = m[1:]  / d_safe
        s = (alpha * alpha + beta * beta).clamp(min=eps).sqrt()
        scale = (3.0 / s).clamp(max=1.0)
        # Apply scaling on each side (two passes — a tangent shared by two
        # segments takes the min scale of the two)
        scale_left  = torch.cat([torch.ones(1, device=device, dtype=dtype),
                                 scale])
        scale_right = torch.cat([scale,
                                 torch.ones(1, device=device, dtype=dtype)])
        m = m * torch.minimum(scale_left, scale_right)
        # Zero tangents on either side of a flat segment
        zero_mask_left  = torch.cat([zero_seg,
                                      torch.zeros(1, device=device, dtype=torch.bool)])
        zero_mask_right = torch.cat([torch.zeros(1, device=device, dtype=torch.bool),
                                      zero_seg])
        m = torch.where(zero_mask_left | zero_mask_right, torch.zeros_like(m), m)

        # Apply curve to luminance; scale chroma by the ratio (preserves hue)
        lum = (0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2]
               + 0.0722 * x[:, 2:3]).clamp(min=1e-6)
        u = lum.clamp(0.0, 1.0)
        seg = (u * (n - 1)).clamp(0.0, n - 1.001)
        idx = seg.floor().long().clamp(0, n - 2)
        t = seg - idx.to(dtype)
        y0 = ys[idx];  y1 = ys[idx + 1]
        m0 = m[idx];   m1 = m[idx + 1]
        h_seg = h[idx.clamp(max=n - 2)]

        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        new_lum = (h00 * y0 + h10 * h_seg * m0
                   + h01 * y1 + h11 * h_seg * m1).clamp(0.0, 1.0)

        ratio = (new_lum / lum).clamp(0.05, 20.0)
        return (x * ratio).clamp(0.0, 1.0)

    def _contrast(self, x):
        """Contrast via Photoshop formula with extended input range.

        Maps param to [-200,200] so the S-curve can reach extreme LR values.
        """
        c = self.contrast * 2.0                                     # free → [-200,200]
        factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c) + 1e-8)
        return (factor * (x - 0.5) + 0.5).clamp(0.0, 1.0)

    def _whites(self, x):
        """Whites: lum > 0.75 additive.  Coefficient 0.40 (up from 0.20)."""
        lum = _luminance(x)
        mask = ((lum - 0.75) / 0.25).clamp(0.0, 1.0)
        return (x + mask * (self.whites / 100.0) * 0.40).clamp(0.0, 1.0)

    def _blacks(self, x):
        """Blacks: lum < 0.25 additive.  Coefficient 0.40 (up from 0.20)."""
        lum = _luminance(x)
        mask = ((0.25 - lum) / 0.25).clamp(0.0, 1.0)
        return (x + mask * (self.blacks / 100.0) * 0.40).clamp(0.0, 1.0)

    # ================================================================
    # Stage 1d — guided-filter edge-aware operations
    # ================================================================
    #
    # The four operators below are 1:1 implementable in NumPy + OpenCV via
    # ``cv2.ximgproc.guidedFilter``: same maths, same constants.  This is the
    # explicit replacement for the multi-level Local Laplacian filter, which
    # was edge-aware but introduced unavoidable bilinear-upsample blur.
    #
    # Conventions:
    #   - guide  = Rec.709 luminance of the *current* image (one-channel)
    #   - input  = either a luminance-zone mask (highlights/shadows) or the
    #              full RGB image (clarity/texture)

    def _highlights_op(self, x):
        """Edge-aware highlights.  Mask = ramp from 0 at lum=0.5 to 1 at
        lum=1.0, snapped to image structure by guided filter.  Negative
        ``highlights`` darkens bright zones (highlight recovery)."""
        lum = _luminance(x)
        mask = ((lum - 0.5) / 0.5).clamp(0.0, 1.0)
        r = self._scaled_radius(self._gf_hl_radius, min(x.shape[-2:]))
        mask = guided_filter(lum, mask, radius=r, eps=self._gf_hl_eps)
        return (x + mask * (self.highlights / 100.0) * 0.50).clamp(0.0, 1.0)

    def _shadows_op(self, x):
        """Edge-aware shadows.  Mask = ramp from 0 at lum=0.5 to 1 at lum=0.0,
        snapped via guided filter.  Positive ``shadows`` lifts dark zones."""
        lum = _luminance(x)
        mask = ((0.5 - lum) / 0.5).clamp(0.0, 1.0)
        r = self._scaled_radius(self._gf_sh_radius, min(x.shape[-2:]))
        mask = guided_filter(lum, mask, radius=r, eps=self._gf_sh_eps)
        return (x + mask * (self.shadows / 100.0) * 0.50).clamp(0.0, 1.0)

    @staticmethod
    def _scaled_radius(base: int, short_side: int, ref: int = 512) -> int:
        """Scale a base radius (calibrated at ``ref`` px short-side) to the
        actual image short side.  Keeps the relative-image-size frequency
        band constant across resolutions — Lightroom-style perceptual
        clarity rather than pixel-window clarity.
        """
        return max(2, int(round(base * short_side / ref)))

    @staticmethod
    def _gaussian_blur_dynamic(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """On-the-fly separable Gaussian blur."""
        ks = max(3, int(6 * sigma + 1) | 1)
        device, dtype = x.device, x.dtype
        xs = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        k = torch.exp(-0.5 * xs * xs / (sigma * sigma + 1e-8))
        k = k / k.sum()
        return _gaussian_blur(x, k)

    def _clarity_op(self, x):
        """Mid-band Gaussian unsharp mask.

        Iteration history led here:
          1. guided filter eps=0.04² → too edge-preserving, x-base ≈ noise.
          2. plain Gaussian σ=8 (at 512 px ref) → strong, but produced 30+
             px halos around telephone wires at full resolution.
          3. guided filter eps=0.005² → luminance guide caused all texture
             (cobblestones) to be treated as "edges" preserved in base;
             x-base ≈ 0 → essentially no clarity effect.

        Final: Gaussian unsharp with **smaller σ** (4 px at 512 ref → 16 px
        at 2K) to get a real clarity boost while keeping halo radius below
        the visible threshold.  No filter can perfectly replicate LR's
        adaptive clarity with simple primitives — this is the best
        trade-off.
        """
        sigma = max(1.0, 4.0 * min(x.shape[-2:]) / 512.0)
        base = self._gaussian_blur_dynamic(x, sigma)
        return (x + (self.clarity / 100.0) * 1.20 * (x - base)).clamp(0.0, 1.0)

    def _texture_op(self, x):
        """High-band Gaussian unsharp mask, σ=2 px at 512 ref → 8 px at 2K.

        NOTE: no longer called from ``forward`` — its high-frequency output
        gets washed away by the detail-transfer blur.  Kept for backwards
        compatibility with code that calls it directly.
        """
        sigma = max(0.5, 2.0 * min(x.shape[-2:]) / 512.0)
        base = self._gaussian_blur_dynamic(x, sigma)
        return (x + (self.texture / 100.0) * 1.20 * (x - base)).clamp(0.0, 1.0)

    def _detail_transfer(self, rendered: torch.Tensor,
                          orig: torch.Tensor) -> torch.Tensor:
        """Luminance-only frequency-separation detail transfer (additive).

            σ scales with image short-side (16 px at 512 ref → 64 px at 2K).
            boost = 1.0 + (clarity + texture) / 200.0  ∈ [1.0, 2.0].

        Algorithm (in linear-light luminance):

            Y_orig_hf = luminance(orig)     - blur(luminance(orig))
            Y_rend_hf = luminance(rendered) - blur(luminance(rendered))
            ΔY        = boost * Y_orig_hf - Y_rend_hf
            out       = rendered + ΔY      # broadcast across channels

        i.e. we replace rendered's high-frequency *luminance* content
        with orig's (boosted), and add the resulting delta as a
        uniform RGB shift.  Adding the same scalar to all three
        channels at a pixel is a pure luminance change — it preserves
        the chroma direction of ``rendered`` to first order and
        reduces to the identity when ``rendered == orig`` and
        boost = 1.

        Why additive instead of the multiplicative
        ``rendered * (Y_new / Y_rend)`` form:

          1. Numerical stability.  Multiplicative needs clamping
             (``Y_rend.clamp(min=eps)``) to avoid divide-by-zero on
             dark pixels, which makes the gradient ill-conditioned for
             L-BFGS strong-Wolfe line search — empirically caused NaN
             losses inside the closure when probing aggressive step
             sizes.
          2. For B&W targets ``rendered`` is already grayscale
             (R=G=B), so additive and multiplicative are identical
             on that path.

        Why luminance-only at all (vs the previous "rendered_lf +
        boost * orig_hf" which grafted orig's *full RGB* HF onto
        rendered):  the old formula leaked orig's pre-edit chroma into
        every edge.  For B&W targets that meant visible green/red
        residue at edges even with saturation = -100; for hue-shift
        edits (red shirt → orange) the shirt bulk tracked rendered
        but folds and edges retained the original red.  By transferring
        only luminance high-freq, chroma at every spatial frequency
        comes from rendered (the chroma-correct edit), while
        detail/texture comes from orig.
        """
        sigma = max(2.0, 16.0 * min(orig.shape[-2:]) / 512.0)
        boost = 1.0 + (self.clarity + self.texture) / 200.0

        Y_orig    = _luminance(orig)                                 # (B,1,H,W)
        Y_orig_hf = Y_orig - self._gaussian_blur_dynamic(Y_orig, sigma)

        Y_rend    = _luminance(rendered)
        Y_rend_hf = Y_rend - self._gaussian_blur_dynamic(Y_rend, sigma)

        delta_Y = boost * Y_orig_hf - Y_rend_hf
        return rendered + delta_Y      # broadcast (B,1,H,W) → (B,3,H,W)

    # ================================================================
    # Stage 2 — white balance
    # ================================================================

    def _white_balance(self, x):
        """Lightroom-correct white balance via Bradford chromatic adaptation.

        ``self.temp`` is stored as relative Kelvin (multiplied by 50 +
        TEMP_NEUTRAL = absolute K).  Both source and target white points come
        from the *same* CIE D-illuminant model: the source is the neutral
        point (temp=0, tint=0 ≡ TEMP_NEUTRAL K) and the target is (temp_K,
        tint).  A Bradford CAT then moves the linear-RGB image from the
        neutral white to the target white.

        Using the neutral-temperature white (not the D65 constant) as source
        is what makes ``temp=0, tint=0`` a true no-op: src == dst so the CAT
        degenerates to the identity.  The previous D65 source (≈6504 K) paired
        with a 5500 K neutral *target* left a residual warm cast at θ=0
        (mean ~8/255, max ~56/255) — that broke the "θ=0 is identity"
        invariant and desynced from opencv_renderer's temp_K==5500 shortcut.
        """
        temp_K = self.temp * 50.0 + TEMP_NEUTRAL                 # absolute K
        # Source white = the neutral point (TEMP_NEUTRAL K, tint 0) on the same
        # D-illuminant locus as the target, so temp=0/tint=0 ⇒ src==dst ⇒ no-op.
        neutral_K = torch.tensor(float(TEMP_NEUTRAL), device=x.device, dtype=x.dtype)
        zero_tint = torch.zeros((), device=x.device, dtype=x.dtype)
        src_xyz = temp_tint_to_white_xyz(neutral_K, zero_tint)
        dst_xyz = temp_tint_to_white_xyz(temp_K, self.tint)

        lin = srgb_to_linear(x.clamp(0.0, 1.0))
        adapted = bradford_cat(lin, src_xyz, dst_xyz)
        return linear_to_srgb(adapted.clamp(min=0.0, max=8.0)).clamp(0.0, 1.0)

    # ================================================================
    # Stage 3 — colour
    # ================================================================

    def _vibrance(self, x):
        """Adaptive saturation.  Coefficient 1.5 (up from 1.0)."""
        max_c = torch.max(x, dim=1, keepdim=True).values
        min_c = torch.min(x, dim=1, keepdim=True).values
        sat = (max_c - min_c) / max_c.clamp(min=1e-6)
        boost = (self.vibrance / 100.0) * 1.5 * (1.0 - sat)
        lum = _luminance(x)
        return (x + boost * (x - lum)).clamp(0.0, 1.0)

    def _saturation(self, x):
        """Global chroma scaling.  Coefficient 2.0; clamped to min=0 so
        negative saturation desaturates (factor→0 = grey) rather than
        inverting colours (factor<0).  factor=0 at saturation=-50."""
        lum = _luminance(x)
        factor = (1.0 + self.saturation / 100.0 * 2.0).clamp(min=0.0)
        return (lum + factor * (x - lum)).clamp(0.0, 1.0)

    def _color_mixer(self, x):
        """step3.apply_color_mixer  —  per-channel HSL with Gaussian soft masks.

        8 channels × (H shift, S shift, L shift).
        Masks are always computed from the *original* hue (not accumulated).
        """
        H, S, L = _rgb_to_hsl(x)

        # Saturation guard: near-neutral pixels have meaningless hue → weight 0
        sat_guard = (S / 0.10).clamp(0.0, 1.0)

        H_adj = H
        S_adj = S
        L_adj = L

        for i in range(8):
            center    = self._mixer_centers[i]
            halfwidth = self._mixer_halfwidth[i]

            # Circular hue distance → cosine-weighted soft mask
            # mask = ½(1 + cos(π·d/W)) for d≤W, else 0.
            # This matches Lightroom: compact support, smooth (C¹) at the
            # boundary, and 1.0 at the centre — unlike a Gaussian's long tail.
            dist = (H - center).abs()
            dist = torch.min(dist, 360.0 - dist)
            inside = (dist <= halfwidth).to(H.dtype)
            hue_mask = 0.5 * (1.0 + torch.cos(torch.pi * dist / halfwidth)) * inside
            eff = hue_mask * sat_guard

            # Hue: ±100 → ±30°
            H_adj = (H_adj + eff * self.mixer_hue[i] * 0.30) % 360.0

            # Saturation: proportional push (same formula as step3)
            s_shift = eff * (self.mixer_sat[i] / 100.0)
            S_adj = torch.where(
                s_shift >= 0,
                (S_adj + s_shift * (1.0 - S_adj)).clamp(0.0, 1.0),
                (S_adj + s_shift * S_adj).clamp(0.0, 1.0))

            # Luminance: proportional push
            l_shift = eff * (self.mixer_lum[i] / 200.0)
            L_adj = torch.where(
                l_shift >= 0,
                (L_adj + l_shift * (1.0 - L_adj)).clamp(0.0, 1.0),
                (L_adj + l_shift * L_adj).clamp(0.0, 1.0))

        return _hsl_to_rgb(H_adj, S_adj, L_adj).clamp(0.0, 1.0)

    def _color_grading(self, x):
        """Colour grading: per-zone (shadow / midtone / highlight) RGB tinting.

        Each zone has 3 learnable RGB offset parameters (range ~[-100, 100]).
        Zones are defined by luminance-based soft masks with smooth transitions.
        """
        lum = _luminance(x)                            # (B, 1, H, W)

        # Soft zone masks (overlapping for smooth transitions)
        shadow_mask = (1.0 - lum * 2.0).clamp(0.0, 1.0)          # lum<0.5: fades from 1→0
        highlight_mask = (lum * 2.0 - 1.0).clamp(0.0, 1.0)       # lum>0.5: fades from 0→1
        midtone_mask = 1.0 - shadow_mask - highlight_mask         # bell shape, peak at lum=0.5

        # Apply RGB offsets per zone (param / 100 scales to ~[-1, 1])
        # Reshape (3,) → (1, 3, 1, 1) for broadcasting
        sh_rgb = self.cg_shadow.view(1, 3, 1, 1) / 100.0
        mid_rgb = self.cg_midtone.view(1, 3, 1, 1) / 100.0
        hi_rgb = self.cg_highlight.view(1, 3, 1, 1) / 100.0

        x = x + shadow_mask * sh_rgb + midtone_mask * mid_rgb + highlight_mask * hi_rgb
        return x.clamp(0.0, 1.0)

    # ================================================================
    # Stage 4 — effects
    # ================================================================

    def _vignette_op(self, x):
        """step3.apply_vignette  —  radial corner darkening.

        Parameterised as signed strength (negative = darken corners,
        matching LR convention) to avoid the zero-gradient problem of abs()
        at initialisation.
        """
        B, C, Hh, W = x.shape
        yy = torch.linspace(-1, 1, Hh, device=x.device).view(1, 1, Hh, 1)
        xx = torch.linspace(-1, 1, W,  device=x.device).view(1, 1, 1, W)
        dist = (xx ** 2 + yy ** 2).sqrt()
        # vignette < 0 → darken corners;  vignette > 0 → brighten corners
        mask = (1.0 + self.vignette / 100.0 * dist * 0.6).clamp(min=0.0)
        return (x * mask).clamp(0.0, 1.0)

    # ================================================================
    # Stage 5 — detail
    # ================================================================

    def _dehaze_op(self, x):
        """Mean-axis contrast boost.  Coefficient /80 (up from /150)."""
        mean = x.mean()
        return (mean + (x - mean) * (1.0 + self.dehaze / 80.0)).clamp(0.0, 1.0)

    def _sharpen(self, x):
        """Tiny-σ unsharp mask (σ=0.5).  Coefficient 1.5 (up from 0.8)."""
        blurred = _gaussian_blur(x, self._k_sharpen)
        return (x + (self.sharpening / 150.0) * 1.5 * (x - blurred)).clamp(0.0, 1.0)

    def _noise_reduction(self, x):
        """step3.apply_noise_reduction  —  α-blend with σ=2 blur."""
        blurred = _gaussian_blur(x, self._k_denoise)
        alpha = (self.noise_red / 100.0).clamp(0.0, 1.0)
        return (x * (1.0 - alpha) + blurred * alpha).clamp(0.0, 1.0)

    # ================================================================
    # Helpers
    # ================================================================

    def get_params_dict(self) -> dict:
        """Return a human-readable parameter dictionary (LR-compatible names).

        ``色温`` is converted back to absolute Kelvin for interoperability
        with the dataset JSON format.
        """
        d = {
            '曝光度':     round(self.exposure.item(), 2),
            '对比度':     round(self.contrast.item(), 1),
            '高光':       round(self.highlights.item(), 1),
            '阴影':       round(self.shadows.item(), 1),
            '白色色阶':   round(self.whites.item(), 1),
            '黑色色阶':   round(self.blacks.item(), 1),
            '色温':       round(self.temp.item() * 50.0 + TEMP_NEUTRAL, 0),
            '色调':       round(self.tint.item(), 1),
            '自然饱和度': round(self.vibrance.item(), 1),
            '饱和度':     round(self.saturation.item(), 1),
            '纹理':       round(self.texture.item(), 1),
            '清晰度':     round(self.clarity.item(), 1),
            '去朦胧':     round(self.dehaze.item(), 1),
            '晕影':       round(self.vignette.item(), 1),
            '锐化':       round(self.sharpening.item(), 1),
            '噪点消除':   round(self.noise_red.item(), 1),
        }
        mixer = []
        for i, name in enumerate(MIXER_NAMES):
            mixer.append({
                'channel': name,
                'H': round(self.mixer_hue[i].item(), 1),
                'S': round(self.mixer_sat[i].item(), 1),
                'L': round(self.mixer_lum[i].item(), 1),
            })
        d['color_mixer'] = mixer
        d['tone_curve_dy'] = [round(v, 2) for v in self.tone_curve_dy.tolist()]
        d['颜色分级'] = {
            'shadow':    {c: round(self.cg_shadow[i].item(), 1)
                          for i, c in enumerate(['R', 'G', 'B'])},
            'midtone':   {c: round(self.cg_midtone[i].item(), 1)
                          for i, c in enumerate(['R', 'G', 'B'])},
            'highlight': {c: round(self.cg_highlight[i].item(), 1)
                          for i, c in enumerate(['R', 'G', 'B'])},
        }
        return d

    def param_count(self) -> int:
        """Total scalar learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def clamp_params(self):
        """Sanitise and clamp parameters after each optimiser step.

        First replaces any NaN/Inf (which can arise when Adam's moment
        estimates are corrupted by a NaN gradient) with 0, then enforces
        physically meaningful bounds so the renderer stays numerically stable.
        Note: tensor.clamp_() propagates NaN, so nan_to_num_ must come first.

        When ``self.bw_mode`` is True, additionally pin every chroma-
        introducing parameter so the optimiser can never paint colour back
        onto a grayscale target (saturation / vibrance / mixer-S kept at
        zero-chroma settings; tint forced to 0 so WB doesn't cast magenta-
        green; colour-grading triplets equalised to R=G=B so they shift
        luminance only).
        """
        for p in self.parameters():
            p.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        self.exposure.clamp_(-5.0, 5.0)
        self.contrast.clamp_(-100.0, 100.0)   # c=contrast*2; singularity at c=259
        self.highlights.clamp_(-100.0, 100.0)
        self.shadows.clamp_(-100.0, 100.0)
        self.whites.clamp_(-100.0, 100.0)
        self.blacks.clamp_(-100.0, 100.0)
        self.temp.clamp_(-80.0, 250.0)          # temp_K = [1 500, 18 000] K
        self.tint.clamp_(-200.0, 200.0)
        self.vibrance.clamp_(-100.0, 100.0)
        self.saturation.clamp_(-100.0, 100.0)
        self.mixer_hue.clamp_(-100.0, 100.0)
        self.mixer_sat.clamp_(-100.0, 100.0)
        self.mixer_lum.clamp_(-100.0, 100.0)
        self.cg_shadow.clamp_(-100.0, 100.0)
        self.cg_midtone.clamp_(-100.0, 100.0)
        self.cg_highlight.clamp_(-100.0, 100.0)
        self.vignette.clamp_(-100.0, 100.0)
        # Detail params: forbid negative entirely.  The optimiser persistently
        # gamed lin_l1 by smoothing the output (smooth + slightly wrong colours
        # has lower per-pixel error than sharp + correct colours when edges
        # don't align perfectly).  With these floored at 0, the only direction
        # to reduce ms_ssim error is *adding* detail, which is what we want.
        self.texture.clamp_(0.0, 100.0)     # was [-100, 100]
        self.clarity.clamp_(0.0, 100.0)     # was [-100, 100]
        self.dehaze.clamp_(-100.0, 100.0)
        self.sharpening.clamp_(0.0, 150.0)
        self.noise_red.clamp_(0.0, 30.0)    # cap aggressive blur
        self.tone_curve_dy.clamp_(-100.0, 100.0)

        if self.bw_mode:
            self.saturation.fill_(-100.0)
            self.vibrance.fill_(-100.0)
            self.mixer_sat.zero_()
            # mixer_hue rotates hue but acts as a no-op on zero-saturation
            # pixels.  Zeroing it anyway keeps the JSON output clean and
            # eliminates any residual non-determinism if upstream code
            # somehow re-saturates downstream of mixer.
            self.mixer_hue.zero_()
            # White balance affects colour cast even on near-grey input
            # via Bradford CAT.  Force temp to neutral (0 ≡ TEMP_NEUTRAL K)
            # and tint to 0 so WB becomes the identity transform.
            self.temp.zero_()
            self.tint.zero_()
            # Equalise each colour-grading triplet to its mean: keeps the
            # zone luminance shift but removes any chroma cast.
            for cg in (self.cg_shadow, self.cg_midtone, self.cg_highlight):
                cg.fill_(cg.mean().item())

    @torch.no_grad()
    def init_from_lr_params(self, lr_params: dict):
        """Warm-start from parsed Lightroom parameters (best-effort).

        Because LR operates on RAW and this renderer on sRGB, the
        *same* numeric values do **not** produce the same visual result.
        This merely provides a better starting point than all-zeros for
        the subsequent gradient optimisation.
        """
        _set = lambda p, v: p.data.fill_(float(v))

        if '曝光度'     in lr_params: _set(self.exposure,   lr_params['曝光度'])
        if '对比度'     in lr_params: _set(self.contrast,   lr_params['对比度'])
        if '高光'       in lr_params: _set(self.highlights,  lr_params['高光'])
        if '阴影'       in lr_params: _set(self.shadows,    lr_params['阴影'])
        if '白色色阶'   in lr_params: _set(self.whites,     lr_params['白色色阶'])
        if '黑色色阶'   in lr_params: _set(self.blacks,     lr_params['黑色色阶'])
        if '色调'       in lr_params: _set(self.tint,       lr_params['色调'])
        if '自然饱和度' in lr_params: _set(self.vibrance,   lr_params['自然饱和度'])
        if '饱和度'     in lr_params: _set(self.saturation, lr_params['饱和度'])
        if '纹理'       in lr_params: _set(self.texture,    lr_params['纹理'])
        if '清晰度'     in lr_params: _set(self.clarity,    lr_params['清晰度'])
        if '去朦胧'     in lr_params: _set(self.dehaze,     lr_params['去朦胧'])
        if '晕影'       in lr_params: _set(self.vignette,   lr_params['晕影'])
        if '锐化'       in lr_params: _set(self.sharpening, lr_params['锐化'])
        if '噪点消除'   in lr_params: _set(self.noise_red,  lr_params['噪点消除'])

        # Temperature: convert absolute K → relative
        if '色温' in lr_params:
            _set(self.temp, (float(lr_params['色温']) - TEMP_NEUTRAL) / 50.0)

        # Color mixer
        if 'color_mixer' in lr_params and isinstance(lr_params['color_mixer'], list):
            import re
            for entry in lr_params['color_mixer']:
                css = entry.get('color', '')
                m = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', css)
                if not m:
                    continue
                rgb_key = f"rgb({m.group(1)}, {m.group(2)}, {m.group(3)})"
                idx = _CSS_TO_CHANNEL_IDX.get(rgb_key)
                if idx is None:
                    continue
                if entry.get('H') is not None:
                    self.mixer_hue.data[idx] = float(entry['H'])
                if entry.get('S') is not None:
                    self.mixer_sat.data[idx] = float(entry['S'])
                if entry.get('L') is not None:
                    self.mixer_lum.data[idx] = float(entry['L'])

        # Color grading (accept both LR Chinese key and English aliases)
        cg_block = lr_params.get('颜色分级')
        if isinstance(cg_block, dict):
            for zone_key, attr in (('shadow', self.cg_shadow),
                                    ('midtone', self.cg_midtone),
                                    ('highlight', self.cg_highlight)):
                z = cg_block.get(zone_key) or {}
                for j, c in enumerate(('R', 'G', 'B')):
                    if c in z: attr.data[j] = float(z[c])
        for k_alias, attr in (('cg_shadow', self.cg_shadow),
                              ('cg_midtone', self.cg_midtone),
                              ('cg_highlight', self.cg_highlight)):
            v = lr_params.get(k_alias)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                attr.data.copy_(torch.tensor([float(v[0]), float(v[1]), float(v[2])]))

        # Tone curve (8-vector of deltas)
        tc = lr_params.get('tone_curve_dy')
        if isinstance(tc, (list, tuple)) and len(tc) == self.tone_curve_dy.numel():
            self.tone_curve_dy.data.copy_(torch.tensor([float(v) for v in tc]))

        # English aliases for texture / clarity (predict() output)
        for k, attr in (('texture', self.texture), ('clarity', self.clarity)):
            if k in lr_params and not isinstance(lr_params[k], (list, dict)):
                _set(attr, lr_params[k])
