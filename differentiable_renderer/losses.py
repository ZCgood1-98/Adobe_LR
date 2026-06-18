"""Composite perceptual / colorimetric loss for the differentiable renderer.

Replaces the legacy L1 + (1-MS-SSIM) loss with a five-term colour-aware bundle:

  1. Linear-light L1                — punishes exposure / WB error in the
     same domain the sensor sees, before the sRGB compander squashes shadows.
  2. LAB ``ab`` L1                  — pulls chromaticity to target without
     re-litigating luminance the curve already covered.
  3. Mean CIEDE2000                 — perceptual colour difference, anchors
     overall colorfulness.
  4. Multi-scale Sliced Wasserstein in CIELAB  — distributional match (very
     forgiving of mis-aligned pixels but very strict on palette).
  5. Per-hue-bucket CIEDE2000       — punishes selective hue corrections
     (e.g. the orange channel got pushed too red); mirrors the renderer's
     mixer geometry.

The weights are exposed and have sensible defaults; ``CompositeLoss`` returns
both the scalar total and a dict of components for logging.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# pytorch-msssim provides a fully differentiable implementation that runs on
# the same dtype/device as the input.  Imported lazily inside the function so
# the rest of the module still works if the dep is missing — only the
# ``ms_ssim_loss`` term degrades to a no-op.
try:
    from pytorch_msssim import ms_ssim as _ms_ssim_fn
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False

from .color_spaces import (
    srgb_to_linear,
    srgb_to_lab,
    delta_e_2000,
    rgb_hue_deg,
)


# ============================================================
# Component losses
# ============================================================

def linear_l1(pred_srgb: torch.Tensor, target_srgb: torch.Tensor) -> torch.Tensor:
    """L1 in linear RGB.  sRGB inputs are gamma-decoded first."""
    return F.l1_loss(srgb_to_linear(pred_srgb.clamp(0.0, 1.0)),
                     srgb_to_linear(target_srgb.clamp(0.0, 1.0)))


def lab_ab_l1(pred_lab: torch.Tensor, target_lab: torch.Tensor) -> torch.Tensor:
    """L1 between LAB ``ab`` channels (chromaticity only)."""
    return F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3])


def mean_delta_e2000(pred_lab: torch.Tensor, target_lab: torch.Tensor) -> torch.Tensor:
    return delta_e_2000(pred_lab, target_lab).mean()


# ---- Multi-scale Sliced Wasserstein (CIELAB) ---------------------------

def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor,
                       n_proj: int = 64) -> torch.Tensor:
    """1D-projection sliced W1 between two pixel clouds.

    *x* and *y* are (B, C, H, W); the SW distance is averaged over batch.
    Projections are sampled per call (random, but works fine in practice for
    a smooth gradient because the distance is averaged over many directions).
    """
    B, C, H, W = x.shape
    x_flat = x.reshape(B, C, -1).transpose(1, 2)        # (B, N, C)
    y_flat = y.reshape(B, C, -1).transpose(1, 2)
    proj = torch.randn(C, n_proj, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-8)
    px = x_flat @ proj                                    # (B, N, P)
    py = y_flat @ proj
    px, _ = torch.sort(px, dim=1)
    py, _ = torch.sort(py, dim=1)
    return (px - py).abs().mean()


def ms_swd_lab(pred_srgb: torch.Tensor, target_srgb: torch.Tensor,
               scales: Sequence[float] = (1.0, 0.5, 0.25),
               n_proj: int = 64) -> torch.Tensor:
    """Multi-scale SWD on LAB images.

    LAB is computed once at full resolution and then bilinear-downsampled per
    scale (cheaper than re-converting and gradient-equivalent for our needs).
    """
    pred_lab = srgb_to_lab(pred_srgb)
    targ_lab = srgb_to_lab(target_srgb)
    losses = []
    for s in scales:
        if s != 1.0:
            H, W = pred_lab.shape[-2:]
            new_size = (max(8, int(round(H * s))), max(8, int(round(W * s))))
            p = F.interpolate(pred_lab, size=new_size, mode='bilinear',
                              align_corners=False)
            t = F.interpolate(targ_lab, size=new_size, mode='bilinear',
                              align_corners=False)
        else:
            p, t = pred_lab, targ_lab
        losses.append(sliced_wasserstein(p, t, n_proj=n_proj))
    return torch.stack(losses).mean()


# ---- Luminance CDF / quantile matching ---------------------------------

# Quantiles to match — covers the shadow / midtone / highlight ranges with
# extra weight on the tails where Lightroom edits matter most.
LUM_CDF_QUANTILES: tuple = (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98)


def luminance_cdf_l1(pred_srgb: torch.Tensor, target_srgb: torch.Tensor,
                     quantiles: Sequence[float] = LUM_CDF_QUANTILES) -> torch.Tensor:
    """Match the luminance distribution at fixed quantiles.

    Each quantile q corresponds to "the luminance below which q fraction of
    pixels live".  Pulling the prediction's q-th percentile to the target's
    q-th percentile directly drives:
      * shadows / blacks  (q ∈ {0.02, 0.05, 0.10})
      * tone curve / contrast (q ∈ {0.25, 0.50, 0.75})
      * highlights / whites (q ∈ {0.90, 0.95, 0.98})

    Why this works where lin_l1 doesn't: ``lin_l1`` rewards "match the mean
    brightness", which is achievable by mild global tweaks.  Quantile L1
    forces the *shape* of the histogram to match — a heavily crushed-shadows
    target cannot be approximated by a slight global darkening.
    """
    def _lum(img):
        return (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2]
                + 0.0722 * img[:, 2:3]).clamp(0.0, 1.0)
    pl = _lum(pred_srgb).flatten(1)                            # (B, N)
    tl = _lum(target_srgb).flatten(1)
    q = torch.tensor(list(quantiles), device=pred_srgb.device,
                     dtype=pred_srgb.dtype)
    # torch.quantile is differentiable wrt the input tensor (linear interp).
    pq = torch.quantile(pl, q, dim=1)                          # (Q, B)
    tq = torch.quantile(tl, q, dim=1)
    return (pq - tq).abs().mean()


# ---- Gradient-domain L1 (sharpness) ------------------------------------

def gradient_l1(pred_srgb: torch.Tensor, target_srgb: torch.Tensor) -> torch.Tensor:
    """[DEPRECATED] L1 between finite-difference spatial gradients.

    Kept for backward compatibility but no longer used by ``CompositeLoss`` —
    the magnitude (~0.01) is too small to influence multi-loss optimisation
    against terms like lab_ab (~5).  Use ``ms_ssim_loss`` instead, which has
    a natural magnitude in [0, 1] and matches structural detail across scales.
    """
    def _lum(img):
        return (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2]
                + 0.0722 * img[:, 2:3])

    p = _lum(pred_srgb.clamp(0.0, 1.0))
    t = _lum(target_srgb.clamp(0.0, 1.0))
    loss_x = F.l1_loss(p[:, :, :, 1:] - p[:, :, :, :-1],
                       t[:, :, :, 1:] - t[:, :, :, :-1])
    loss_y = F.l1_loss(p[:, :, 1:, :] - p[:, :, :-1, :],
                       t[:, :, 1:, :] - t[:, :, :-1, :])
    return (loss_x + loss_y) * 0.5


# ---- MS-SSIM (multi-scale structural similarity) -----------------------

def ms_ssim_loss(pred_srgb: torch.Tensor, target_srgb: torch.Tensor) -> torch.Tensor:
    """1 - MS-SSIM, computed on luminance over 5 scales.

    MS-SSIM compares local mean / variance / covariance at multiple
    resolutions, so it directly rewards matching the *structure* of the
    target rather than its pixel values.  Crucially, MS-SSIM is naturally
    in [0, 1] — comparable in magnitude to lab_ab L1 — so it actually
    competes for the optimiser's attention, unlike grad_l1 (~0.01).

    Different scales correspond to different renderer parameters:
      * highest 1-2 scales : sharpening, texture
      * middle scales      : clarity
      * lowest scales      : contrast, tone curve

    Implementation uses ``pytorch_msssim`` if available; degrades to a
    constant 0 (no-op) if the dependency is missing.
    """
    if not _HAS_MSSSIM:
        return pred_srgb.new_zeros(())

    def _lum(img):
        return (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2]
                + 0.0722 * img[:, 2:3])

    p = _lum(pred_srgb.clamp(0.0, 1.0))
    t = _lum(target_srgb.clamp(0.0, 1.0))
    # MS-SSIM needs at least 161 px on the short side at default 5 scales
    # (each scale halves the image; the smallest must still cover an 11-px
    # window).  Fall back to a single-scale SSIM via win_size=7 for tiny crops.
    H, W = p.shape[-2:]
    if min(H, W) < 161:
        # Reduce levels to fit; pytorch_msssim picks them by min dim
        # automatically when we pass `weights=None`, but we cap at 3 scales.
        scale_count = max(1, min(3, int(torch.log2(torch.tensor(float(min(H, W)))).item()) - 4))
        weights_t = torch.tensor([0.5, 0.3, 0.2][:scale_count],
                                 device=p.device, dtype=p.dtype)
        weights_t = weights_t / weights_t.sum()
        return 1.0 - _ms_ssim_fn(p, t, data_range=1.0, win_size=7,
                                 weights=weights_t)
    return 1.0 - _ms_ssim_fn(p, t, data_range=1.0)


# ---- Highlight overshoot (anti blow-out) -------------------------------

def highlight_overshoot(pred_srgb: torch.Tensor, target_srgb: torch.Tensor,
                         thresh: float = 0.92) -> torch.Tensor:
    """Asymmetric L1 on the part of pred that pushes brighter than target
    in the highlight region.

    Once a pixel saturates to 1.0, gradients through the renderer's
    internal ``.clamp(0, 1)`` calls vanish — the optimiser has no signal
    to back off whites/exposure/contrast/cg_highlight even though the
    L1 loss "knows" the pixel is wrong.  This term pushes back through
    the *unsaturated* pixels in the same operator chain: penalising
    overshoot anywhere in the highlight band gives the early operators
    a coherent "reduce overall brightness" gradient that survives the
    clamp.

    Computed per-channel (not on luminance) so a yellow blow-out where
    R and G saturate but B is mid-range is still penalised.
    """
    # Per-channel excess above threshold; only positive overshoot counts.
    pred_excess = (pred_srgb - thresh).clamp(min=0.0)
    targ_excess = (target_srgb - thresh).clamp(min=0.0)
    return (pred_excess - targ_excess).clamp(min=0.0).mean()


# ---- Per-hue-bucket CIEDE2000 ------------------------------------------

# Mirror the 8 mixer channels in renderer.py (same centres / widths).
HUE_BUCKET_CENTERS = (0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 280.0, 320.0)
HUE_BUCKET_WIDTHS = (40.0, 30.0, 35.0, 50.0, 35.0, 40.0, 35.0, 35.0)


def per_hue_delta_e2000(pred_srgb: torch.Tensor, target_srgb: torch.Tensor,
                         centers: Sequence[float] = HUE_BUCKET_CENTERS,
                         widths: Sequence[float] = HUE_BUCKET_WIDTHS) -> torch.Tensor:
    """Weighted mean ΔE2000 per hue bucket, then averaged over buckets.

    Each pixel's contribution to bucket *i* is a soft cosine weight on the
    circular hue distance; this matches the mixer's masking geometry so that
    a per-bucket failure (e.g. orange shifted too red) is amplified rather
    than diluted by the global average.
    """
    pred_lab = srgb_to_lab(pred_srgb)
    targ_lab = srgb_to_lab(target_srgb)
    de = delta_e_2000(pred_lab, targ_lab)                # (B, 1, H, W)
    H = rgb_hue_deg(target_srgb)                          # weight by *target* hue
    # Saturation guard: ignore near-neutral pixels where hue is unstable
    max_c = target_srgb.max(dim=1, keepdim=True).values
    min_c = target_srgb.min(dim=1, keepdim=True).values
    sat = (max_c - min_c) / max_c.clamp(min=1e-6)
    sat_w = (sat / 0.10).clamp(0.0, 1.0)

    bucket_means = []
    for c, w in zip(centers, widths):
        d = (H - c).abs()
        d = torch.min(d, 360.0 - d)
        # cosine-weighted bucket: 1 at centre, 0 at half-width
        mask = 0.5 * (1.0 + torch.cos(torch.pi * d / w))
        mask = torch.where(d <= w, mask, torch.zeros_like(mask)) * sat_w
        denom = mask.sum().clamp(min=1.0)
        bucket_means.append((mask * de).sum() / denom)
    return torch.stack(bucket_means).mean()


# ---- Asymmetric "parity" terms ----------------------------------------
#
# Both terms are *one-sided* relu losses: zero when pred ≥ target on the
# given axis, positive otherwise.  They counter the systematic bias of
# lab_ab / ms_ssim toward "safe" undersaturated, undersharp output that
# scores well on those metrics by avoiding misalignment-induced error.
#
# Diagnostic on the 28-sample second run:
#   - 13/28 saturation < 0   (mean -6.6, range to -100)
#   - 18/28 sharpening = 0   (mean +2.0)
#   - 11/28 clarity = 0      (mean +12.2)
#   - 12/28 texture = 0      (mean +6.4)
# i.e. Adam consistently parks detail params at the floor and pulls
# saturation negative.  These two losses re-introduce a perceptual signal
# that "less colour / less detail than target" is bad without forbidding
# the legitimate cases where target IS less colour / less detail.

RANGE_PARITY_BANDS: tuple = ((0.10, 0.90), (0.25, 0.75))


def range_parity(pred_srgb: torch.Tensor, target_srgb: torch.Tensor,
                  bands: tuple = RANGE_PARITY_BANDS) -> torch.Tensor:
    """One-sided dynamic-range loss: penalise pred having NARROWER luminance
    range than target, evaluated across multiple quantile bands.

    Why this is needed even with ``lum_cdf``: ``lum_cdf`` is per-quantile L1
    over a fixed set of points and **uniformly weighted**.  When the
    optimiser has reduced low-end and high-end residuals to ~0.01 each but
    has a +0.04 mid-tone residual, it's locally indifferent — the L1
    gradient says "redistribute equally", which leaves contrast slightly
    compressed.

    Why multiple bands: the q05/q95 extremes can be matched (or even
    over-shot via crushed blacks / blown whites) while the mid-band
    (q10-q90, IQR q25-q75) is meaningfully narrower in pred than target.
    Summing one-sided relu over (0.10, 0.90) and (0.25, 0.75) catches
    both "overall span" and "mid-tone spread" contrast issues with a
    single term.

    One-sided so a target with genuinely lower dynamic range doesn't get
    over-stretched.
    """
    def _lum(img):
        return (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2]
                + 0.0722 * img[:, 2:3]).clamp(0.0, 1.0)
    pl = _lum(pred_srgb).flatten(1)
    tl = _lum(target_srgb).flatten(1)
    q_vals = []
    for lo, hi in bands:
        q_vals.append(lo); q_vals.append(hi)
    q = torch.tensor(q_vals, device=pred_srgb.device, dtype=pred_srgb.dtype)
    pq = torch.quantile(pl, q, dim=1)                      # (2*N, B)
    tq = torch.quantile(tl, q, dim=1)
    loss = pred_srgb.new_zeros(())
    for i, _ in enumerate(bands):
        pred_range = pq[2 * i + 1] - pq[2 * i]
        targ_range = tq[2 * i + 1] - tq[2 * i]
        loss = loss + F.relu(targ_range - pred_range).mean()
    return loss / len(bands)


def lum_lowres_l1(pred_srgb: torch.Tensor, target_srgb: torch.Tensor,
                   target_size: int = 32) -> torch.Tensor:
    """L1 of low-resolution luminance maps.

    Why this exists, when ``lum_cdf`` and ``ms_ssim`` already touch luminance:

    * ``lum_cdf`` matches the *histogram* of luminance — it is spatially
      blind.  A target with "bright centre + dark corners" (post-crop
      vignette) shares its histogram with a prediction that has "uniform
      mid-grey + heavily darkened patches anywhere", so ``lum_cdf`` reports
      zero residual even when the spatial layout is completely wrong.

    * ``ms_ssim`` is intentionally invariant to global luminance shifts
      (compares local mean/variance/covariance, not absolute values).  A
      prediction that is uniformly dimmer than target but structurally
      identical wins on ``ms_ssim``.

    This term closes the gap: ``F.adaptive_avg_pool2d`` to a tiny grid
    (32×32 default) collapses high-frequency / alignment noise but
    preserves the global *spatial* luminance distribution — centre vs
    corners, sky vs ground, subject vs background.  L1 in this domain
    forces the renderer to reproduce the absolute luminance map, which
    is the signal driving vignette + exposure/whites/highlights to their
    target values.

    Costs one extra avg-pool + L1 per evaluation; negligible.
    """
    def _lum(img):
        return (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2]
                + 0.0722 * img[:, 2:3])
    H, W = pred_srgb.shape[-2:]
    # Preserve aspect ratio: scale the longer side to target_size.
    if H >= W:
        new_h = target_size
        new_w = max(4, int(round(W * target_size / H)))
    else:
        new_w = target_size
        new_h = max(4, int(round(H * target_size / W)))
    lp = F.adaptive_avg_pool2d(_lum(pred_srgb.clamp(0.0, 1.0)),  (new_h, new_w))
    lt = F.adaptive_avg_pool2d(_lum(target_srgb.clamp(0.0, 1.0)), (new_h, new_w))
    return F.l1_loss(lp, lt)


def ab_mean_parity(pred_lab: torch.Tensor,
                   target_lab: torch.Tensor) -> torch.Tensor:
    """L1 between spatial means of LAB ``a`` and ``b`` channels.

    Anchors *direction* of overall chromatic balance.  Where ``lab_ab`` is
    per-pixel and gets dragged toward "average to neutral" by crop/alignment
    error, and ``chroma_parity`` only sees chroma *magnitude* (so a desaturated
    blue cast satisfies it equally well as a desaturated orange one), this
    term directly punishes "pred is on the cool/green side of target" by
    comparing the mean (a, b) of each image as 2D points.

    Symmetric, so it fires whether pred is too warm or too cool.
    """
    pred_ab = pred_lab[:, 1:3].mean(dim=(-2, -1))     # (B, 2)
    targ_ab = target_lab[:, 1:3].mean(dim=(-2, -1))
    return (pred_ab - targ_ab).abs().mean()


def chroma_parity(pred_srgb: torch.Tensor,
                  target_srgb: torch.Tensor) -> torch.Tensor:
    """One-sided chroma loss: ``relu(target_chroma − pred_chroma)``.

    Per-pixel chroma is ``max(RGB) − min(RGB)`` (saturation-like).  We
    average across the image then take the relu of (target − pred) so:
      - pred more saturated than target → 0 (no penalty)
      - pred less saturated → linear penalty in chroma units

    This pushes back against lab_ab's tendency to win on misaligned
    edges by globally desaturating.
    """
    pc = pred_srgb.max(dim=1).values - pred_srgb.min(dim=1).values   # (B,H,W)
    tc = target_srgb.max(dim=1).values - target_srgb.min(dim=1).values
    return F.relu(tc - pc).mean()  # per-pixel relu → mean


def detail_parity(pred_srgb: torch.Tensor,
                  target_srgb: torch.Tensor,
                  kernel: int = 5) -> torch.Tensor:
    """One-sided high-frequency energy loss.

    Compares the average ``|HF|`` of pred vs target where HF is computed
    as ``image - box_blur(image)`` (separable avg pool — good enough for
    energy comparison, ~5x cheaper than Gaussian).  Returns
    ``relu(|HF_target| − |HF_pred|)`` so:
      - pred has at least as much HF energy as target → 0
      - pred is softer/blurrier → linear penalty

    Combined with the existing detail-positive ``param_reg`` floors (in
    a follow-up step), this gives the optimiser a *target-aware* reason
    to push clarity / texture / sharpening above zero, instead of the
    current "as long as no detail is destroyed, parking at 0 is free".
    """
    pad = kernel // 2
    p_lf = F.avg_pool2d(pred_srgb,   kernel_size=kernel, stride=1, padding=pad)
    t_lf = F.avg_pool2d(target_srgb, kernel_size=kernel, stride=1, padding=pad)
    p_hf = (pred_srgb   - p_lf).abs().mean()
    t_hf = (target_srgb - t_lf).abs().mean()
    return F.relu(t_hf - p_hf)


# ============================================================
# Composite
# ============================================================

class CompositeLoss(nn.Module):
    """Bundle the five components with adjustable weights.

    The default weights are tuned so that each component's gradient
    magnitude is roughly comparable on a typical Lightroom-style edit; tune
    if you scale to a different domain.
    """

    # B&W-target weights.  Chroma losses are zeroed because both pred and
    # target are luminance-only by construction (saturation pinned at -100),
    # so they fire on degenerate signal and waste optimiser bandwidth.
    # Luminance-targeting losses (lin_l1, lum_cdf, ms_ssim) are boosted so
    # they dominate the now-uncontested gradient, and detail_pa is boosted
    # because B&W edits emphasise tonal range and texture more than colour
    # photos.  Used by ``_fit_attempt`` when ``renderer.bw_mode is True``.
    BW_LOSS_WEIGHTS = {
        'lin_l1':    1.5,      # boost: only essential pixel-domain signal
        'lab_ab':    0.0,      # ab is 0 for both → no information
        'de2000':    0.0,      # ΔE collapses to luminance dE → redundant w/ lin_l1
        'ms_swd':    0.0,      # LAB ab projections degenerate; gradient is noise
        'hue_de':    0.0,      # hue undefined on grey → mask sums collapse
        'hl_over':   2.0,      # luminance overshoot still real
        'param_reg': 0.5,      # detail-floor / highlight-clip still applicable
        'lum_cdf':   2.0,      # double: dominant luminance-distribution driver
        'ms_ssim':   5.0,      # boost from 3.0: no chroma competition
        'chroma_pa': 0.0,      # both grey → degenerate, no signal
        'ab_mean':   0.0,      # mean(ab) is 0 for both → no information
        'detail_pa': 8.0,      # boost: B&W edits often emphasise micro-contrast
        'lum_low':   3.0,      # boost: BW edits depend heavily on
                               # vignette + exposure spatial gradient
        'range_pa':  5.0,      # one-sided range parity: BW edits prize wide
                               # dynamic range.  Raw value ~0.02-0.05 when
                               # significant residual; weight 5.0 keeps
                               # contribution comparable to lum_cdf / lum_low
                               # rather than dominating.
    }

    # Near-BW weights: target is heavily desaturated but has a residual tint
    # (sepia, cool grey, etc.).  Halfway between DEFAULT and BW: keep enough
    # chroma signal to fit the tint, but boost the luminance terms toward BW
    # levels so they're not drowned by chroma_pa/ab_mean noise on degenerate
    # colour channels.  Used when ``bw_level == 'near_bw'`` (chroma 99th
    # percentile in [0.04, 0.10]).
    NEAR_BW_LOSS_WEIGHTS = {
        'lin_l1':    1.3,      # mild boost toward BW's 1.5
        'lab_ab':    0.15,     # 30% of default — enough to anchor tint hue
        'de2000':    0.02,
        'ms_swd':    0.08,
        'hue_de':    0.03,     # bucket signal weak with low chroma; keep small
        'hl_over':   2.0,
        'param_reg': 0.5,
        'lum_cdf':   1.8,      # near-BW: tonal distribution is the main story
        'ms_ssim':   4.0,      # halfway between default 3.0 and BW 5.0
        'chroma_pa': 1.5,      # 30% of default — keep some chroma push
        'ab_mean':   0.15,     # 30% of default — direction anchor for the tint
        'detail_pa': 7.0,      # near-BW shares BW's "crisp blacks" preference
        'lum_low':   2.5,      # near-BW shares BW's vignette+exposure spatial
                               # signature; full BW boost is 3.0
        'range_pa':  4.0,      # near-BW also wants stretched dynamic range
                               # (slightly less aggressive than full BW since
                               # some real tint may need preserving)
    }

    DEFAULT_WEIGHTS = {
        'lin_l1':    1.0,
        'lab_ab':    0.5,
        'de2000':    0.05,    # ΔE values run 0..30+, smaller weight balances scale
        'ms_swd':    0.25,
        'hue_de':    0.10,    # 0.05 → 0.10: per-hue-bucket ΔE raw values run
                              # ~5-10 in late stage.  A 4x boost (0.20) caused
                              # high-scale divergence because the residual has
                              # an irreducible floor (crop-alignment + renderer
                              # expressivity); at 0.20 the optimiser destroys
                              # already-good params to fight this floor.  2x
                              # (0.10) keeps the hue-bucket pressure useful
                              # without overwhelming the rest of the bundle.
        'ab_mean':   0.5,     # mean-(a,b) L1.  Catches global cool/warm cast
                              # that lab_ab smooths over and chroma_pa is
                              # blind to (chroma_pa is magnitude-only).
                              # Raw values usually 1-8 for a meaningful cast,
                              # so weight 0.5 contributes ~0.5-4.
        'hl_over':   2.0,     # highlight overshoot; raw values are tiny (~1e-3)
        'param_reg': 0.5,     # was 0.01: too weak to overcome lin_l1's preference
                              # for "smooth + slightly wrong" over "sharp + right";
                              # 0.5 makes the detail-destruction L1 cost ~12% of
                              # total, enough to deter texture<0 / noise_red>0.
        'lum_cdf':   1.0,     # luminance distribution match
        'ms_ssim':   3.0,     # 5.0 caused single-step explosions at scale 256
                              # because ms_ssim's gradient direction can swing
                              # sharply between iterations and at 5× weight it
                              # dominated the total gradient direction past
                              # what gradient clipping (norm=1.0) could damp.
                              # 3.0 contributes ~20% — enough to motivate
                              # clarity > 0 without destabilising convergence.
        'chroma_pa': 5.0,     # one-sided per-pixel relu: relu(tc-pc).mean().
                              # Per-pixel formulation gives non-zero gradient
                              # even when mean chroma matches (pixels where
                              # pred < target still fire).  Weight 5.0 makes
                              # this ~10-20% of total loss.
        'detail_pa': 5.0,     # one-sided: penalise pred HF energy < target HF.
                              # Counters Adam parking clarity/texture at 0
                              # (~40% of samples in the 28-sample run).  Raw
                              # range ~0..0.05; weight 5.0 contributes ~10-15%.
        'lum_low':   1.0,     # low-resolution luminance map L1: catches
                              # vignette/exposure spatial layout that lum_cdf
                              # (histogram) and ms_ssim (structure) both miss.
                              # Raw value ~0.02-0.08; weight 1.0 contributes
                              # 0.02-0.08 to total ~2 (1-4%).  BW/near-BW
                              # boost this further since vignette is a
                              # standard BW edit idiom.
        'range_pa':  1.0,     # one-sided range parity (q10-q90 ∪ q25-q75
                              # stretch).  Modest weight on colour because
                              # chroma_pa already covers some of the same
                              # "don't under-edit" intent; BW/near-BW boost
                              # much higher since luminance dynamic range is
                              # the defining feature of BW edits.
    }

    def __init__(self, weights: dict | None = None,
                 swd_n_proj: int = 64,
                 swd_scales: Sequence[float] = (1.0, 0.5, 0.25),
                 renderer: nn.Module | None = None):
        super().__init__()
        w = dict(self.DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        self.weights = w
        self.swd_n_proj = swd_n_proj
        self.swd_scales = swd_scales
        # Optional renderer reference for parameter regularisation.
        # Stored as a plain attribute (not a submodule) so the renderer
        # parameters are NOT registered twice with the optimiser.
        object.__setattr__(self, '_renderer', renderer)

    def _param_reg(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Asymmetric L1 on parameter axes whose extremes destroy the image.

        Two reg directions:

        (a) Highlight blow-out — positive side of:
              exposure, whites, contrast, dehaze, cg_highlight
            Once these clip pixels to 1.0 the local clamp kills the gradient
            and the optimiser can't back off.  This was the original purpose
            of ``param_reg``.

        (b) Detail destruction — *anti*-detail directions:
              texture < 0   (actively blurs mid-frequencies)
              clarity < 0   (actively flattens local contrast)
              noise_red > 0 (gaussian-blurs the output)
            The optimiser likes these because lin_l1 prefers smooth outputs
            over edge-preserving sharp ones (smooth = lower per-pixel error
            on misaligned edges).  Penalising the wrong direction stops it
            actively *removing* detail to game lin_l1 — Adam can still
            choose 0 if a positive value isn't justified.

        Magnitudes are normalised so a single ``param_reg`` weight controls
        all of these.  ``ref_tensor`` provides device/dtype for the no-op
        zero return when no renderer is attached.
        """
        r = self._renderer
        if r is None:
            return ref_tensor.new_zeros(())
        pos = lambda p: p.clamp(min=0.0)
        neg_abs = lambda p: p.clamp(max=0.0).abs()
        # Base reg: detail-destruction and rarely-used highlight-blowout knobs
        # that apply uniformly across colour and BW targets.
        reg = (
            pos(r.dehaze)                / 100.0
            + pos(r.cg_highlight).mean() / 100.0
            # detail destruction
            + neg_abs(r.texture)         / 100.0
            + neg_abs(r.clarity)         / 100.0
            + pos(r.noise_red)           / 100.0
        )
        # Positive-direction penalties on the *primary* brightening knobs
        # (exposure, whites, contrast).  These deter highlight blow-out from
        # over-driven contrast on COLOUR edits, where blow-out also kills
        # chroma detail.  For BW edits the typical Lightroom recipe is
        # "exposure +0.5~1.0 EV, whites +20~+40, contrast +20~+40" paired
        # with a vignette — these penalties fight the user's actual workflow.
        # ``hl_over`` is an independent term that still catches genuine
        # blow-out (per-channel asymmetric L1 above 0.92), so removing the
        # positive-side reg here does not remove the safety net.
        if not getattr(r, 'bw_target', False):
            reg = reg + pos(r.exposure) / 5.0
            reg = reg + pos(r.whites)   / 100.0
            reg = reg + pos(r.contrast) / 100.0
        # (c) **Chroma desaturation floor** — only when the target is
        # genuinely colourful (``bw_target`` is False).  Adam routinely lands
        # on vibrance ≈ -20 / saturation ≈ -5 because it can satisfy the
        # chroma residual via "warm temp + cool tint" or HSL mixer shifts at
        # lower gradient cost.  Penalising the *negative* direction on the
        # global chroma knobs forces Adam to use saturation/vibrance when
        # more colour is needed, rather than misrouting through WB or mixer.
        # Gated by ``bw_target`` (not ``bw_mode``) so near_bw — where the
        # target IS heavily desaturated and saturation should sit near -40 —
        # is also exempt.  Without this gate the floor would pull near_bw's
        # saturation back to 0 against the chroma loss profile, which is
        # what was producing the "near_bw fit comes out fully coloured" bug.
        if not getattr(r, 'bw_target', False):
            reg = reg + neg_abs(r.vibrance)   / 100.0
            reg = reg + neg_abs(r.saturation) / 100.0
        return reg

    def forward(self, pred_srgb: torch.Tensor,
                target_srgb: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Skip computation for zero-weight components.  Reason: even at
        # weight=0 the autograd graph still runs the component's backward
        # pass, which for the LAB-derived terms (de2000, ms_swd, hue_de)
        # involves divisions that can produce NaN gradients on degenerate
        # inputs (e.g. grayscale-vs-grayscale B&W targets where chroma is
        # uniformly zero).  Those NaN grads then propagate into params via
        # Adam's m/v moments and get silently zeroed by ``clamp_params()``,
        # parking the fit at identity (observed bug for the B&W loss
        # profile prior to this guard).
        w = self.weights
        def use(k): return w.get(k, 0.0) != 0.0

        # LAB only needed if any LAB-derived term is active.
        needs_lab = use('lab_ab') or use('de2000') or use('ab_mean')
        pred_lab = srgb_to_lab(pred_srgb) if needs_lab else None
        targ_lab = srgb_to_lab(target_srgb) if needs_lab else None

        comps = {}
        if use('lin_l1'):    comps['lin_l1']    = linear_l1(pred_srgb, target_srgb)
        if use('lab_ab'):    comps['lab_ab']    = lab_ab_l1(pred_lab, targ_lab)
        if use('de2000'):    comps['de2000']    = mean_delta_e2000(pred_lab, targ_lab)
        if use('ms_swd'):    comps['ms_swd']    = ms_swd_lab(pred_srgb, target_srgb,
                                                              scales=self.swd_scales,
                                                              n_proj=self.swd_n_proj)
        if use('hue_de'):    comps['hue_de']    = per_hue_delta_e2000(pred_srgb, target_srgb)
        if use('hl_over'):   comps['hl_over']   = highlight_overshoot(pred_srgb, target_srgb)
        if use('param_reg'): comps['param_reg'] = self._param_reg(pred_srgb)
        if use('lum_cdf'):   comps['lum_cdf']   = luminance_cdf_l1(pred_srgb, target_srgb)
        if use('ms_ssim'):   comps['ms_ssim']   = ms_ssim_loss(pred_srgb, target_srgb)
        if use('chroma_pa'): comps['chroma_pa'] = chroma_parity(pred_srgb, target_srgb)
        if use('ab_mean'):   comps['ab_mean']   = ab_mean_parity(pred_lab, targ_lab)
        if use('detail_pa'): comps['detail_pa'] = detail_parity(pred_srgb, target_srgb)
        if use('lum_low'):   comps['lum_low']   = lum_lowres_l1(pred_srgb, target_srgb)
        if use('range_pa'):  comps['range_pa']  = range_parity(pred_srgb, target_srgb)

        total = sum(w[k] * v for k, v in comps.items())
        log = {k: v.detach().item() for k, v in comps.items()}
        log['total'] = total.detach().item()
        return total, log
