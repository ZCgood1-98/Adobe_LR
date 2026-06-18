"""Differentiable guided filter (He, Sun, Tang 2010).

Edge-preserving local linear model:

    q_i = a_k * I_i + b_k     for all i in window k

with (a_k, b_k) chosen by least-squares on (I, p) inside the window.
The classic closed-form expressions are entirely linear/quadratic in the
inputs, so the whole filter is differentiable end-to-end via PyTorch
autograd.

The output of this module is **bit-equivalent (up to float precision) to
``cv2.ximgproc.guidedFilter``** when:

  - ``I`` is single-channel (grayscale guide)
  - ``p`` is single- or multi-channel
  - same radius / eps
  - same boundary condition (we use REFLECT padding to match OpenCV's
    BORDER_DEFAULT)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1) x (2r+1) window with reflect padding.

    Implemented via a Summed-Area Table (running cumulative sums) so the
    cost is **O(H*W) regardless of radius**, vs ``F.avg_pool2d`` which is
    O(H*W*k^2).  Bit-equivalent to the avg_pool2d implementation up to
    float-32 cumsum rounding (~1e-5 on 1k+ images, well below uint8
    quantisation).  Differentiable via cumsum's autograd.
    """
    if radius <= 0:
        return x

    # 1) reflect-pad so output windows for boundary pixels match OpenCV
    pad = radius
    x_p = F.pad(x, [pad, pad, pad, pad], mode='reflect')          # (B,C,H+2r,W+2r)

    # 2) build SAT (cumsum along H, then W) **in float64** to keep precision
    #    on large images (a 2K-side image has ~4M cumsum partial sums; float32
    #    would accumulate ~0.5 of error and translate into visible per-pixel
    #    drift after the guided-filter division).  Cast back to original
    #    dtype once the SAT differences are taken.  Prepend a row+col of zeros
    #    so the SAT-difference formula has a valid 0 origin at index 0.
    orig_dtype = x.dtype
    cs = x_p.to(torch.float64).cumsum(dim=-2).cumsum(dim=-1)        # (B,C,H+2r,W+2r)
    cs = F.pad(cs, [1, 0, 1, 0], mode='constant', value=0.0)        # (B,C,H+2r+1,W+2r+1)

    # 3) sum over a k x k window starting at (i, j) in the padded coord:
    #    sum = SAT[i+k, j+k] - SAT[i, j+k] - SAT[i+k, j] + SAT[i, j]
    H, W = x.shape[-2:]
    k = 2 * radius + 1
    sum_ = (cs[..., k:k + H, k:k + W]
            - cs[..., :H,    k:k + W]
            - cs[..., k:k + H, :W]
            + cs[..., :H,    :W])
    return (sum_ / (k * k)).to(orig_dtype)


def guided_filter(I: torch.Tensor,
                  p: torch.Tensor,
                  radius: int = 8,
                  eps: float = 1e-4) -> torch.Tensor:
    """Edge-preserving guided filter with a *single-channel* guide.

    Args
    ----
    I : (B, 1, H, W) — guide (typically Rec.709 luminance in [0, 1])
    p : (B, C, H, W) — filter input; C may be 1 or 3
    radius : window half-size in pixels
    eps    : regularisation; eps = sigma^2 where sigma is the image-intensity
             scale below which detail is treated as smoothable noise.
             eps = 0.01**2 = 1e-4 preserves edges with contrast > ~1%.

    Returns
    -------
    q : (B, C, H, W) — same shape as ``p``
    """
    if I.dim() != 4 or I.shape[1] != 1:
        raise ValueError(
            f"guide must have shape (B, 1, H, W); got {tuple(I.shape)}")

    mean_I  = box_filter(I, radius)                    # (B,1,H,W)
    mean_p  = box_filter(p, radius)                    # (B,C,H,W)
    mean_Ip = box_filter(I * p, radius)                # (B,C,H,W) via broadcast
    cov_Ip  = mean_Ip - mean_I * mean_p                # (B,C,H,W)

    mean_II = box_filter(I * I, radius)                # (B,1,H,W)
    # SAT-based box_filter has cumsum precision errors that, in *flat*
    # regions where mean_II ≈ mean_I**2 to begin with, can push the variance
    # slightly negative.  Without clamping, ``var_I + eps`` flips sign and
    # ``a`` blows up — observed as huge per-pixel diffs vs OpenCV's reference.
    var_I = (mean_II - mean_I * mean_I).clamp(min=0.0)  # (B,1,H,W)

    a = cov_Ip / (var_I + eps)                         # (B,C,H,W)
    b = mean_p - a * mean_I                            # (B,C,H,W)

    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    return mean_a * I + mean_b
