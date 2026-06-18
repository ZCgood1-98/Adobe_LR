"""Differentiable Fast Local Laplacian Filtering.

Aubry / Paris / Hasinoff approximation:
  1. Compute K full-resolution remapped versions of the luminance image, one
     per reference intensity g_k (linspace 0..1).
  2. Build a Laplacian pyramid for each remapped image.
  3. For every output pyramid level, blend the K Laplacian residuals at that
     level by linearly interpolating along K based on the *local* Gaussian
     pyramid intensity of the original image.
  4. Collapse the synthesised Laplacian pyramid back to a full-resolution
     luminance, then re-modulate the original chroma by the new/old ratio.

Highlights / shadows enter the remap (sign-conditioned tone scaling).
Clarity / texture enter as per-level Laplacian amplification (clarity is the
mid-band, texture is the high-band).  The whole module is differentiable
end-to-end and stateless: parameter values come from the renderer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .color_spaces import srgb_to_linear, linear_to_srgb


# ============================================================
# Pyramid utilities
# ============================================================

def _make_gauss_kernel(sigma: float, device, dtype) -> torch.Tensor:
    ks = max(3, int(6 * sigma + 1) | 1)
    x = torch.arange(ks, device=device, dtype=dtype) - ks // 2
    k = torch.exp(-0.5 * x * x / (sigma * sigma + 1e-8))
    return k / k.sum()


def _separable_blur(img: torch.Tensor, k1d: torch.Tensor) -> torch.Tensor:
    C = img.shape[1]
    ks = k1d.shape[0]
    kh = k1d.view(1, 1, 1, ks).expand(C, -1, -1, -1).to(img.dtype)
    kv = k1d.view(1, 1, ks, 1).expand(C, -1, -1, -1).to(img.dtype)
    out = F.pad(img, [ks // 2, ks // 2, 0, 0], mode='reflect')
    out = F.conv2d(out, kh, groups=C)
    out = F.pad(out, [0, 0, ks // 2, ks // 2], mode='reflect')
    out = F.conv2d(out, kv, groups=C)
    return out


def _gauss_pyramid(img: torch.Tensor, n_levels: int,
                   k1d: torch.Tensor) -> list[torch.Tensor]:
    pyr = [img]
    for _ in range(n_levels - 1):
        pyr.append(F.avg_pool2d(_separable_blur(pyr[-1], k1d), 2))
    return pyr


def _laplacian_pyramid(img: torch.Tensor, n_levels: int,
                       k1d: torch.Tensor) -> list[torch.Tensor]:
    g = _gauss_pyramid(img, n_levels + 1, k1d)
    lap = []
    for i in range(n_levels):
        up = F.interpolate(g[i + 1], size=g[i].shape[-2:],
                           mode='bilinear', align_corners=False)
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def _collapse(lap: list[torch.Tensor]) -> torch.Tensor:
    out = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        out = F.interpolate(out, size=lap[i].shape[-2:],
                            mode='bilinear', align_corners=False) + lap[i]
    return out


# ============================================================
# Remap (Paris / Hasinoff)
# ============================================================

def _remap(I: torch.Tensor, g0: torch.Tensor, sigma_r: float,
           alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """One-pass remap.

    I:        (B,1,H,W) input intensity
    g0:       scalar or (B,1,H,W) reference intensity
    sigma_r:  detail / tone threshold
    alpha:    detail exponent  (1 -> identity, <1 enhance, >1 suppress)
    beta:     tone scaling for the over-range portion (broadcastable)
    """
    d = I - g0
    sign = torch.sign(d)
    a = (d.abs() / sigma_r).clamp(min=1e-6)
    detail = sign * sigma_r * a.pow(alpha)
    tone = sign * (sigma_r + beta * (d.abs() - sigma_r).clamp(min=0.0))
    near = (d.abs() <= sigma_r)
    return g0 + torch.where(near, detail, tone)


# ============================================================
# Module
# ============================================================

class LocalLaplacianFilter(nn.Module):
    """Stateless edge-aware tone/detail operator.

    Use ``apply(srgb, highlights, shadows, clarity, texture)``.
    """

    def __init__(self,
                 n_levels: int = 5,
                 n_intensity: int = 5,
                 sigma_r: float = 0.30,
                 sigma_blur: float = 1.0):
        super().__init__()
        self.n_levels = n_levels
        self.n_intensity = n_intensity
        self.sigma_r = sigma_r
        self.sigma_blur = sigma_blur

    # ------------------------------------------------------------
    @staticmethod
    def _sign_beta(sign: torch.Tensor,
                   highlights: torch.Tensor,
                   shadows: torch.Tensor) -> torch.Tensor:
        """Per-pixel β: positive residual → highlights, negative → shadows."""
        hi = (1.0 - 0.7 * highlights / 100.0).clamp(min=0.1, max=3.0)
        lo = (1.0 - 0.7 * shadows    / 100.0).clamp(min=0.1, max=3.0)
        return torch.where(sign >= 0, hi, lo)

    # ------------------------------------------------------------
    def apply(self, srgb: torch.Tensor,
              highlights: torch.Tensor,
              shadows: torch.Tensor,
              clarity: torch.Tensor,
              texture: torch.Tensor) -> torch.Tensor:
        device, dtype = srgb.device, srgb.dtype
        k1d = _make_gauss_kernel(self.sigma_blur, device, dtype)

        lin = srgb_to_linear(srgb.clamp(0.0, 1.0))
        lum = (0.2126 * lin[:, 0:1] + 0.7152 * lin[:, 1:2]
               + 0.0722 * lin[:, 2:3]).clamp(min=1e-6)

        # K full-res remapped luminance images (only TONE remap; detail is
        # applied later as a per-level multiplier on the Laplacian residual).
        K = self.n_intensity
        L = self.n_levels
        refs = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
        alpha_one = torch.tensor(1.0, device=device, dtype=dtype)

        # (K, B, 1, H, W) — store remapped images for indexing
        remapped_pyrs: list[list[torch.Tensor]] = []  # per-k list of pyramids
        for k in range(K):
            g_k = refs[k]
            sign = torch.sign(lum - g_k)
            beta = self._sign_beta(sign, highlights, shadows)
            remapped = _remap(lum, g_k, self.sigma_r, alpha_one, beta)
            remapped_pyrs.append(_laplacian_pyramid(remapped, L, k1d))

        # Canonical Gaussian pyramid of the *original* luminance for lookup
        g_pyr = _gauss_pyramid(lum, L + 1, k1d)

        # Per-level: blend the K Laplacian residuals using local intensity.
        out_lap: list[torch.Tensor] = []
        for i in range(L + 1):
            stack = torch.stack([rp[i] for rp in remapped_pyrs], dim=0)  # (K,B,1,Hi,Wi)
            g_i = g_pyr[i]                                                # (B,1,Hi,Wi)
            idx = (g_i.nan_to_num(0.0) * (K - 1)).clamp(0.0, float(K - 1))
            i0 = idx.floor().long().clamp(0, K - 2)  # i0 in [0,K-2], i1=i0+1 in [1,K-1]
            w = (idx - i0.to(dtype)).clamp(0.0, 1.0)                     # (B,1,Hi,Wi)
            i0_exp = i0.unsqueeze(0)                                      # (1,B,1,Hi,Wi)
            i1_exp = i0_exp + 1
            lo = torch.gather(stack, 0, i0_exp).squeeze(0)
            hi = torch.gather(stack, 0, i1_exp).squeeze(0)
            blended = (1.0 - w) * lo + w * hi

            # Per-level detail amplification (clarity vs texture).
            # i = 0 (highest freq) -> texture; i = L-1 -> clarity.
            if i < L:
                t = i / max(1, L - 1)
                detail = texture * (1.0 - t) + clarity * t
                boost = (1.0 + detail / 100.0).clamp(min=0.0, max=3.0)
                blended = blended * boost
            out_lap.append(blended)

        new_lum = _collapse(out_lap).clamp(min=1e-6)

        # Re-apply chroma by scaling linear RGB by the lum ratio (preserves hue).
        ratio = (new_lum / lum).clamp(0.05, 20.0)
        return linear_to_srgb(
            (lin * ratio).clamp(min=0.0, max=8.0)
        ).clamp(0.0, 1.0)
