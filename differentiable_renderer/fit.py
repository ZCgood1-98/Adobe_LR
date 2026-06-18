#!/usr/bin/env python3
"""Gradient-based parameter fitting for the differentiable renderer.

Per-sample fit, wrapped in an N-way multi-start (``--n_restarts``) that keeps
the attempt with the lowest full-resolution default-loss:
  1. Init: amortised CNN warm-start (``--amortized_init``) and/or parsed-LR
     warm-start (``--warm_start``), plus B&W / vignette / detail heuristics.
     Each restart cycles its strategy through [warm, zero, perturb].
  2. Multi-scale Adam (default 256 → 512 → 1024) with per-parameter-group
     LRs, gradient clipping, divergence detection and LR-anneal-on-plateau.
  3. Block-coordinate descent over five semantic blocks
     (WB → colour → HSL mixer → curves → local), each solved with
     L-BFGS / strong-Wolfe line search.
  4. Final global L-BFGS strong-Wolfe polish over **all** parameters jointly.
  5. Chroma + detail micro-fit (``--micro_fit_iters``) with boosted
     chroma_pa / detail_pa weights.
  6. Restore the lowest-loss state observed across every phase.

Loss is the composite ``CompositeLoss`` (linear-L1 + LAB-ab + ΔE2000 +
MS-SWD + per-hue ΔE2000 + MS-SSIM + one-sided chroma/detail/range parity)
defined in ``losses.py``.

Recommended invocation — the explicit values below are also the tuned
defaults, so drop any flag you don't need to override:

    python -m differentiable_renderer.fit \
        --data_dir try_new_dataset1/train \
        --output_dir fit_results \
        --amortized_init amortized_init.pt \
        --warm_start \
        --all_versions \
        --scales 256 512 1024 \
        --adam_iters 800 \
        --cd_sweeps 5 \
        --cd_block_iters 25 \
        --polish_iters 200 \
        --micro_fit_iters 200 \
        --n_restarts 2 \
        --device cuda

``--all_versions`` fits every ``version_*`` subdir of each sample; results mirror
the dataset layout — ``{output_dir}/{sample_id}/version_{n}/`` with one
``params.json`` / ``rendered.png`` / ``compare.png`` / ``history.json`` /
``renderer.pt`` per version.  Omit it to fit only ``--version`` (default 0).
Add ``--limit N`` to process only the first N samples (quick smoke test).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running from the package directory or the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from differentiable_renderer.renderer   import DifferentiableRenderer
from differentiable_renderer.losses     import CompositeLoss
from differentiable_renderer.crop_align import align_original_to_edited, needs_crop


# ============================================================
# Image I/O
# ============================================================

def detect_bw_level(img: torch.Tensor, q: float = 0.99) -> str:
    """Classify the target's saturation regime for tiered initialisation.

    Returns one of:
      'full_bw' — true greyscale; pin chroma params hard (saturation=-100)
      'near_bw' — heavily desaturated single-tone edit (cool grey, sepia).
                  Pre-init saturation around -70 so Adam doesn't waste steps
                  climbing out of the "barely desaturated" basin
      'color'   — normal coloured edit; no special handling

    Uses the *q*-th percentile of per-pixel chroma (max-min channel range)
    rather than the mean, because mean is dragged toward 0 by large
    near-neutral regions even when skin / foliage carry strong colour.

    Thresholds: chroma_q is the 99th-percentile per-pixel chroma.  A pure-BW
    JPEG measures ~0.05-0.08 in chroma_q because JPEG chroma subsampling and
    edge aliasing leak a few pixels' worth of low-amplitude chroma even when
    the target is visually monochrome.  The original 0.04/0.10 cutoffs
    classified those as 'near_bw' and let the residual chroma signal pull
    saturation back toward 0 (rendered came out visibly tinted).  Raising
    full_bw to 0.08 catches "looks pure BW but has JPEG noise" cases,
    leaving the 0.08-0.13 band for genuinely tinted near-BW edits (sepia,
    cool grey) where the tint amplitude exceeds JPEG-noise levels.
    """
    with torch.no_grad():
        max_c = img.max(dim=1).values
        min_c = img.min(dim=1).values
        chroma = (max_c - min_c).flatten()
        chroma_q = torch.quantile(chroma, q).item()
        # Secondary: mean chroma.  JPEG noise is high-frequency / sparse
        # (high p99 but low mean), while a real tint stains many pixels
        # uniformly.
        chroma_mean = chroma.mean().item()
        # Tertiary: fraction of pixels with chroma above 0.15.  This is the
        # decisive signal for "mostly achromatic image with a small but
        # vivid colour element" — e.g. a near-greyscale swan with a red
        # beak measures p99=0.086, mean=0.038 (would look like JPEG noise on
        # the global stats above), but ~0.3% of pixels (the beak) have
        # chroma > 0.15 and max chroma exceeds 0.5.  Pure-BW JPEG noise
        # never produces clusters this strong (max ~0.10-0.15, frac_15=0).
        # A 0.001 threshold catches the swan-beak case while remaining
        # robust to single-pixel outliers.
        frac_15 = (chroma > 0.15).float().mean().item()
        # Quaternary: fraction of pixels with *moderate* chroma (> 0.10).
        # This separates two cases that look identical to the p99/mean stats
        # (high p99, low mean): (a) JPEG chroma-subsampling noise on a true
        # greyscale — the elevated chroma is a sparse scatter of edge pixels,
        # frac_10 stays well under 1%; (b) a heavily-desaturated edit that
        # *retains a real colour region* (a warm subject, a coloured sky
        # gradient) — that region paints a contiguous patch, easily pushing
        # frac_10 to several percent.  The flower-on-sand sunset edit measures
        # p99=0.12, mean=0.03, frac_15=0.0002 (below the swan-beak trigger) but
        # frac_10=0.045 — clearly real colour the user wants visible, not noise.
        frac_10 = (chroma > 0.10).float().mean().item()
    # Real localised colour content → never classify as BW.  The colour
    # element (logo, lips, eyes, accent prop, etc.) is usually the most
    # aesthetically important part of the edit; collapsing it to grey
    # destroys the intent.  ``frac_15`` catches small-but-vivid accents;
    # ``frac_10`` catches larger moderately-coloured regions that never reach
    # 0.15 chroma but still read as colour to the eye.
    if frac_15 >= 0.001 or frac_10 >= 0.02:
        return 'color'
    if chroma_q < 0.08:
        return 'full_bw'
    if chroma_q < 0.13 and chroma_mean < 0.045:
        # High p99 but low mean.  Treat as full BW *only* if the elevated
        # chroma is genuinely sparse (JPEG noise).  If a meaningful fraction
        # of pixels carry moderate chroma the image has real surviving colour
        # — fall through to near_bw rather than crushing it to grey.
        if frac_10 < 0.01:
            return 'full_bw'
        return 'near_bw'
    if chroma_q < 0.13:
        return 'near_bw'
    return 'color'


def detect_bw(img: torch.Tensor) -> bool:
    """Backwards-compatible alias: True iff image is effectively greyscale."""
    return detect_bw_level(img) == 'full_bw'


def estimate_vignette(img: torch.Tensor) -> float:
    """Estimate a sensible initial value for the vignette parameter.

    Compares the mean luminance of the four image corners against the
    central region.  If the corners are noticeably darker (relative to the
    centre) we pre-set the vignette parameter so Adam starts from a sensible
    point rather than 0.  The vignette renderer formula is::

        x_out = x * (1 + (vignette/100) * dist * 0.6)

    where ``dist`` peaks at ~1.41 in the corners.  Solving for vignette
    given the observed corner / centre luminance ratio:

        ratio ≈ 1 + (vignette/100) * 1.41 * 0.6
        vignette ≈ (ratio - 1) * 100 / (1.41 * 0.6)

    Clamped to [-90, 0] (positive vignette / brightening corners is rare and
    we'd rather let Adam discover it from 0).
    """
    with torch.no_grad():
        H, W = img.shape[-2:]
        ch, cw = H // 4, W // 4
        lum = (0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2]
               + 0.0722 * img[:, 2:3])
        corners = torch.cat([
            lum[..., :ch,    :cw   ].flatten(),
            lum[..., :ch,    -cw:  ].flatten(),
            lum[..., -ch:,   :cw   ].flatten(),
            lum[..., -ch:,   -cw:  ].flatten(),
        ])
        center = lum[..., H // 4: 3 * H // 4, W // 4: 3 * W // 4].mean().item()
        corner_mean = corners.mean().item()
    if center < 1e-3:
        return 0.0
    ratio = corner_mean / center
    v = (ratio - 1.0) * 100.0 / (1.41 * 0.6)
    return float(max(-90.0, min(0.0, v)))


def load_image(path_or_pil, long_side: int | None = None,
               grayscale: bool = False) -> torch.Tensor:
    """Load an image to a (1,3,H,W) tensor in [0,1].

    ``grayscale=True``: replicate Rec.709 luminance into all three
    channels.  Used by the BW shortcut in ``fit_single`` so the renderer
    optimises a gray-input → gray-target problem when the target is
    truly grayscale, eliminating the colour input the optimiser would
    otherwise have to compensate for.
    """
    if isinstance(path_or_pil, Image.Image):
        img = path_or_pil.convert('RGB')
    else:
        img = Image.open(path_or_pil).convert('RGB')
    if long_side is not None:
        w, h = img.size
        scale = long_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if grayscale:
        # Rec.709 luminance, replicated across channels so downstream
        # pipeline (which expects (B,3,H,W)) is unaffected.
        Y = (0.2126 * t[:, 0:1] + 0.7152 * t[:, 1:2] + 0.0722 * t[:, 2:3])
        t = Y.expand(-1, 3, -1, -1).contiguous()
    return t


def save_image(tensor: torch.Tensor, path: str):
    arr = (tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255
           ).byte().cpu().numpy()
    Image.fromarray(arr).save(path)


def save_comparison(orig: torch.Tensor, rendered: torch.Tensor,
                    target: torch.Tensor, path: str):
    h = min(orig.shape[2], rendered.shape[2], target.shape[2])
    def _resize(t):
        if t.shape[2] != h:
            w = int(t.shape[3] * h / t.shape[2])
            return F.interpolate(t, size=(h, w), mode='bilinear',
                                 align_corners=False)
        return t
    parts = [_resize(t) for t in [orig, rendered, target]]
    save_image(torch.cat(parts, dim=3), path)


# ============================================================
# Parameter blocks (semantic groups)
# ============================================================

def parameter_blocks(renderer: DifferentiableRenderer) -> dict[str, list[torch.nn.Parameter]]:
    """5 semantic blocks for coordinate descent.

    Block             | parameters
    ------------------|---------------------------------------------------------
    ``wb``            | temp, tint
    ``color``         | vibrance, saturation, cg_shadow/midtone/highlight
    ``hsl``           | mixer_hue / mixer_sat / mixer_lum   (8-vec each)
    ``curves``        | exposure, contrast, whites, blacks, tone_curve_dy (8)
    ``local``         | highlights, shadows, clarity, texture, vignette,
                      |   dehaze, sharpening, noise_red

    Frozen params (``requires_grad=False`` — typically set when
    ``renderer.bw_mode`` is True) are filtered out: the L-BFGS instance
    would otherwise crash on the first ``_gather_flat_grad`` call, and
    Adam would silently accumulate m/v moments for parameters it can't
    actually move.  Empty blocks are dropped from the result entirely so
    callers iterating ``blocks.items()`` skip them naturally.
    """
    r = renderer
    raw = {
        'wb':     [r.temp, r.tint],
        'color':  [r.vibrance, r.saturation,
                   r.cg_shadow, r.cg_midtone, r.cg_highlight],
        'hsl':    [r.mixer_hue, r.mixer_sat, r.mixer_lum],
        'curves': [r.exposure, r.contrast, r.whites, r.blacks,
                   r.tone_curve_dy],
        'local':  [r.highlights, r.shadows, r.clarity, r.texture,
                   r.vignette, r.dehaze, r.sharpening, r.noise_red],
    }
    out = {}
    for k, params in raw.items():
        active = [p for p in params if p.requires_grad]
        if active:
            out[k] = active
    return out


# Per-group Adam learning rates (chosen so each group makes ~similar
# percentage progress per step regardless of how the param scale was defined).
DEFAULT_GROUP_LRS = {
    'wb':     0.5,         # temp uses (K-5500)/50 → ±200 typical, large LR ok
    'color':  0.3,
    'hsl':    0.5,
    'curves': 0.1,         # tone_curve_dy is sensitive
    'local':  0.3,
}


def adam_param_groups(blocks: dict[str, list[torch.nn.Parameter]],
                      group_lrs: dict[str, float] | None = None
                      ) -> list[dict]:
    g_lrs = dict(DEFAULT_GROUP_LRS)
    if group_lrs:
        g_lrs.update(group_lrs)
    return [{'params': blocks[k], 'lr': g_lrs[k], 'name': k} for k in blocks]


# ============================================================
# Optimisation primitives
# ============================================================

def _clamp_grads(params: Iterable[torch.nn.Parameter], max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad],
                                    max_norm=max_norm)


def _make_adam_sched(optim: torch.optim.Optimizer, total_iters: int,
                     warmup_start: float = 0.05,
                     warmup_frac: float = 0.10,
                     warmup_cap: int = 20):
    """LinearLR-style warmup followed by cosine decay, via LambdaLR.

    Adam's first step has a very large effective magnitude
    (``m_hat / sqrt(v_hat) ≈ sign(grad)`` until the second-moment estimate
    catches up), so with our base LRs of 0.1–0.5 a single step can move
    a parameter by 0.1–0.5 of its scale — large enough to leap straight
    over the clamp boundary on a warm-started initialisation.  Once
    parameters are clamp-pinned, the gradient keeps pushing outward and
    the clamp keeps pulling them back, deadlocking the optimisation.

    Default linear ramp from 5% → 100% of base LR over the first 10% of
    iterations (capped at 20 steps) defuses this for a CNN-init cold start.

    At scale transitions the prior-scale's best state is a delicate balance
    that doesn't survive Adam's iter-0 ``sign(grad)`` kick — every Adam phase
    after the first one was deadlocking within 1-3 steps on this benchmark.
    The caller bypasses this by passing ``warmup_start=0.01, warmup_frac=0.30,
    warmup_cap=60`` for scale_idx > 0 (gentler 1%→100% ramp over up to 60
    steps), which keeps Adam in the existing basin while the second-moment
    estimate fills in.
    """
    warmup = max(1, min(warmup_cap, int(round(total_iters * warmup_frac))))

    def lr_factor(it: int) -> float:
        if it < warmup:
            return warmup_start + (1.0 - warmup_start) * it / max(1, warmup - 1)
        progress = (it - warmup) / max(1, total_iters - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_factor)


def _track_best(best: dict | None, loss_value: float, renderer: torch.nn.Module):
    """Snapshot ``renderer.state_dict()`` if ``loss_value`` beats best['loss'].

    Cheap (49 scalar params + small buffers ≈ a few hundred bytes) and only
    triggers on improvement, so the running cost is negligible.  The point
    is to guarantee the final returned renderer is at least as good as any
    finite-loss state ever observed — a safety net for downstream phases
    (L-BFGS strong-Wolfe in particular) that can numerically derail and
    leave the renderer worse than where they started.

    NOTE on scale: ``best['loss']`` is comparable only against losses
    evaluated at the *same scale* (image resolution).  Different scales have
    different loss surfaces (e.g. ms_ssim depends on resolution).  When the
    fitter transitions between scales, ``_rebaseline_best`` re-anchors the
    saved loss to the new scale before any cross-comparison.
    """
    if best is None or not math.isfinite(loss_value):
        return
    if loss_value < best['loss']:
        best['loss'] = loss_value
        best['sd'] = {k: v.detach().clone()
                      for k, v in renderer.state_dict().items()}


def _rebaseline_best(best: dict | None, renderer, orig, target, loss_fn):
    """Re-evaluate ``best['sd']`` at the current (orig, target) resolution.

    Why: ``best['loss']`` was recorded at whatever scale the renderer was
    being optimised in when the snapshot was taken.  When we move to a
    different scale (or just want to compare against the current renderer's
    loss at this scale), the previously-saved loss value is on a different
    surface and not directly comparable.  Re-running the saved params
    through the loss at the *current* resolution gives us a value we can
    legitimately compare to subsequent losses.
    """
    if best is None or best['sd'] is None:
        return
    cur_sd = {k: v.detach().clone() for k, v in renderer.state_dict().items()}
    renderer.load_state_dict(best['sd'])
    with torch.no_grad():
        rendered = renderer(orig)
        loss, _ = loss_fn(rendered, target)
    best['loss'] = loss.item() if torch.isfinite(loss) else float('inf')
    renderer.load_state_dict(cur_sd)


def _restore_if_worse(best: dict | None, renderer, orig, target, loss_fn,
                       tag: str = '', verbose: bool = True) -> bool:
    """Pick whichever is lower — the current renderer state or the best
    snapshot — by *actually evaluating both at the current scale*.

    Why two forward passes (vs trusting ``best['loss']``):
    ``best['loss']`` is recorded at whatever scale was being optimised
    when the snapshot was taken (and ``_rebaseline_best`` may not have
    caught all transitions, especially after silent L-BFGS crashes that
    bypass tracking).  Comparing a stale-scale ``best['loss']`` against
    a fresh-scale ``cur_loss`` produces wrong decisions in both
    directions.  This implementation re-evaluates both at the *same*
    scale and updates ``best['loss']`` to the scale-aligned value as a
    side effect.  Returns True iff a restore happened.
    """
    if best is None or best['sd'] is None:
        return False
    # Current state's loss at this scale
    with torch.no_grad():
        rendered = renderer(orig)
        cur_loss, _ = loss_fn(rendered, target)
    cur = cur_loss.item() if torch.isfinite(cur_loss) else float('inf')

    # Best snapshot's loss at this scale (load → forward → restore)
    cur_sd = {k: v.detach().clone() for k, v in renderer.state_dict().items()}
    renderer.load_state_dict(best['sd'])
    with torch.no_grad():
        rendered = renderer(orig)
        best_loss, _ = loss_fn(rendered, target)
    best_at_scale = (best_loss.item() if torch.isfinite(best_loss)
                     else float('inf'))

    if best_at_scale < cur:
        # Best is better at this scale — keep it loaded, sanitise, and
        # update best['loss'] so subsequent comparisons are coherent.
        renderer.clamp_params()
        best['loss'] = best_at_scale
        if verbose:
            print(f"  [restore{('/' + tag) if tag else ''}] "
                  f"cur={cur:.4f} > best={best_at_scale:.4f} → reverted")
        return True
    else:
        # Current is at least as good — put it back and leave best
        # untouched (its 'loss' remains stale-but-irrelevant; tracking
        # inside the next phase will correct it).
        renderer.load_state_dict(cur_sd)
        return False


def adam_phase(renderer, orig, target, loss_fn,
               iters: int, lr_groups: list[dict],
               clip_norm: float = 1.0,
               verbose: bool = True, scale: int | None = None,
               best: dict | None = None,
               divergence_ratio: float = 1.5,
               divergence_streak: int = 40,
               max_resets: int = 2,
               anneal_patience: int = 30,
               anneal_rel_threshold: float = 0.001,
               max_anneals: int = 4,
               phase_tag: str = 'adam',
               warmup_start: float = 0.05,
               warmup_frac: float = 0.10,
               warmup_cap: int = 20):
    """Adam pass with divergence detection.

    The 50-sample benchmark showed 26% of fits had Adam excursions where
    loss climbed above 1.5x the running min and stayed there for hundreds
    of steps before partially recovering — pure wasted compute, and often
    leaving best-state worse than it could have been.

    Mitigation: every step we compare current loss against the in-phase
    minimum.  After ``divergence_streak`` consecutive steps above
    ``divergence_ratio`` × min, restore best, halve the LR, and reset the
    Adam optimiser state (m, v).  Up to ``max_resets`` such recoveries; if
    the third trigger fires we break out of the phase entirely (the
    surface is too sharp for our LR schedule and further iteration just
    burns compute).

    Adam eps=1e-3 (default 1e-8): prevents the "denominator collapse"
    divergence where a parameter sits at a clamp boundary for many steps
    (gradient = 0, v_hat decays toward 0), then a non-zero gradient arrives
    and the update step = lr * m_hat / sqrt(v_hat + eps) becomes
    astronomically large.  Empirically this caused single-step explosions
    (lab_ab doubling, ms_ssim doubling) at scale 512.  eps=1e-3 floors the
    denominator to a sane value while preserving Adam's adaptive nature.
    """
    base_lrs = [g['lr'] for g in lr_groups]
    optim = torch.optim.Adam(lr_groups, eps=1e-3)
    sched = _make_adam_sched(optim, iters,
                             warmup_start=warmup_start,
                             warmup_frac=warmup_frac,
                             warmup_cap=warmup_cap)
    history = []
    nan_resets = 0
    div_resets = 0
    in_phase_min = float('inf')
    streak = 0
    # LR annealing on plateau: track the best loss seen and how long since
    # we last beat it by a relative margin.  When the streak exceeds
    # ``anneal_patience``, halve the LR — this turns a stalled fit into
    # a refinement phase rather than wasting compute at unmatched LR.
    last_best_for_anneal = float('inf')
    anneal_streak = 0
    anneal_count = 0
    for it in range(iters):
        optim.zero_grad()
        rendered = renderer(orig)
        loss, log = loss_fn(rendered, target)
        if not torch.isfinite(loss):
            nan_resets += 1
            if nan_resets > 3:
                if verbose:
                    print(f"  [adam {it}] NaN/Inf — giving up after 3 resets")
                break
            if verbose:
                print(f"  [adam {it}] NaN/Inf — sanitising params (reset #{nan_resets})")
            renderer.clamp_params()
            optim = torch.optim.Adam(lr_groups, eps=1e-3)
            sched = _make_adam_sched(optim, max(1, iters - it),
                                     warmup_start=warmup_start,
                                     warmup_frac=warmup_frac,
                                     warmup_cap=warmup_cap)
            continue
        loss.backward()
        _clamp_grads(renderer.parameters(), clip_norm)
        # Snapshot BEFORE step so saved params correspond to the evaluated loss.
        _track_best(best, log['total'], renderer)

        # Divergence monitor: track the min loss seen so far in this phase
        # and count consecutive steps above the ratio threshold.
        if math.isfinite(log['total']):
            in_phase_min = min(in_phase_min, log['total'])
            if (in_phase_min > 0 and
                    log['total'] > divergence_ratio * in_phase_min):
                streak += 1
            else:
                streak = 0
            if streak >= divergence_streak:
                if div_resets >= max_resets:
                    if verbose:
                        print(f"  [adam {it}] persistent divergence "
                              f"({log['total']:.3f} > {divergence_ratio}× "
                              f"min={in_phase_min:.3f}) — breaking phase")
                    # Restore best on the way out so the next phase starts
                    # from a sane state instead of from divergence territory.
                    # ``_restore_if_worse`` would do this anyway, but doing
                    # it inline keeps the renderer state internally
                    # consistent with the printed loss.
                    if best is not None and best['sd'] is not None:
                        renderer.load_state_dict(best['sd'])
                        renderer.clamp_params()
                    break
                div_resets += 1
                if best is not None and best['sd'] is not None:
                    renderer.load_state_dict(best['sd'])
                    renderer.clamp_params()
                # Halve LRs and rebuild Adam state (clears m, v that fed
                # the divergence in the first place).
                for g, base in zip(lr_groups, base_lrs):
                    base *= 0.5
                    g['lr'] = base
                base_lrs = [g['lr'] for g in lr_groups]
                optim = torch.optim.Adam(lr_groups, eps=1e-3)
                sched = _make_adam_sched(optim, max(1, iters - it),
                                         warmup_start=warmup_start,
                                         warmup_frac=warmup_frac,
                                         warmup_cap=warmup_cap)
                streak = 0
                in_phase_min = float('inf')
                if verbose:
                    print(f"  [adam {it}] divergence reset #{div_resets}: "
                          f"restored best={best['loss']:.4f}, LR halved")
                continue

        optim.step()
        renderer.clamp_params()
        sched.step()

        # LR annealing on plateau.  If we beat the previous "anneal best"
        # by a relative margin, reset the streak — otherwise count up.
        if math.isfinite(log['total']):
            if last_best_for_anneal == float('inf') or \
               log['total'] < last_best_for_anneal * (1.0 - anneal_rel_threshold):
                last_best_for_anneal = log['total']
                anneal_streak = 0
            else:
                anneal_streak += 1
            if anneal_streak >= anneal_patience and anneal_count < max_anneals:
                anneal_count += 1
                # Halve LRs across all groups; rebuild the cosine sched on
                # the remaining iterations so we don't immediately wind LR
                # back up on the next sched.step().
                for g, base in zip(lr_groups, base_lrs):
                    base *= 0.5
                    g['lr'] = base
                base_lrs = [g['lr'] for g in lr_groups]
                # Rebuild Adam state: with the much smaller LR, the existing
                # m/v moments are calibrated for the wrong step size and
                # would push us past the local optimum on the first step.
                optim = torch.optim.Adam(lr_groups, eps=1e-3)
                sched = _make_adam_sched(optim, max(1, iters - it),
                                         warmup_start=warmup_start,
                                         warmup_frac=warmup_frac,
                                         warmup_cap=warmup_cap)
                anneal_streak = 0
                last_best_for_anneal = float('inf')
                if verbose:
                    print(f"  [{phase_tag} {it}] LR anneal #{anneal_count}: "
                          f"plateau detected → LRs halved")

        history.append({'phase': phase_tag, 'scale': scale, 'iter': it, **log})
        if verbose and (it % 100 == 0 or it == iters - 1):
            tag = f"s{scale}|" if scale else ""
            # Use .get() because B&W loss profile skips lab_ab/de2000/ms_swd
            # entirely (weight=0 → omitted from comps), so the keys are absent.
            print(f"  [{tag}{phase_tag} {it:3d}/{iters}] "
                  f"L={log['total']:.4f} "
                  f"lin={log.get('lin_l1', 0.0):.4f} "
                  f"ab={log.get('lab_ab', 0.0):.3f} "
                  f"ΔE={log.get('de2000', 0.0):.3f} "
                  f"swd={log.get('ms_swd', 0.0):.3f}")
    return history


def lbfgs_polish(renderer, orig, target, loss_fn,
                 params: list[torch.nn.Parameter] | None = None,
                 max_iter: int = 30,
                 tag: str = 'global',
                 verbose: bool = True,
                 best: dict | None = None):
    """L-BFGS with strong-Wolfe line search.

    PyTorch's L-BFGS only minimises over the parameters fed to it, so it can
    serve both the per-block coordinate-descent step and the final global
    polish — pass ``params=None`` for "all parameters".
    """
    # Sanitise before L-BFGS: NaN/Inf parameters crash the strong-Wolfe
    # line search with "value cannot be converted to type float".
    renderer.clamp_params()

    if params is None:
        # Global polish: optimise every trainable parameter.  Filter out
        # frozen ones (set by bw_mode and similar): PyTorch L-BFGS calls
        # ``_gather_flat_grad`` which dereferences ``p.grad`` and crashes
        # on None — which is what frozen parameters' grads stay as.
        p = [q for q in renderer.parameters() if q.requires_grad]
    else:
        p = list(params)
    # L-BFGS expects requires_grad; freeze the rest by zeroing their grad
    # explicitly inside the closure (nothing else updates them).
    optim = torch.optim.LBFGS(p, max_iter=max_iter, line_search_fn='strong_wolfe',
                              tolerance_grad=1e-5, tolerance_change=1e-7,
                              history_size=20)
    history = []
    state = {'last': None}

    def closure():
        optim.zero_grad()
        # Zero grads of *all* params so unrelated grads don't accumulate
        for q in renderer.parameters():
            if q.grad is not None:
                q.grad = None
        rendered = renderer(orig)
        loss, log = loss_fn(rendered, target)
        if torch.isfinite(loss):
            loss.backward()
            _clamp_grads(p, max_norm=1.0)
            # Track best *inside* the closure: L-BFGS strong-Wolfe evaluates
            # multiple candidate points during line search (and across
            # max_iter inner iterations).  The renderer's parameters at this
            # exact moment correspond to ``log['total']``, so this is the
            # legitimate snapshot point.  Tracking only after optim.step()
            # — the previous behaviour — captured at most the last
            # evaluation, missing every better intermediate point.  This
            # was responsible for the bulk of the observed 54%
            # best-restore failure rate (final > min by an avg of 0.69).
            _track_best(best, log['total'], renderer)
        state['last'] = log
        return loss

    try:
        optim.step(closure)
    except (RuntimeError, ValueError, IndexError) as e:
        # IndexError: PyTorch L-BFGS strong-Wolfe bracket collapses on
        # non-smooth surfaces (heavy clamping) — safe to skip the block.
        if verbose:
            print(f"  [lbfgs/{tag}] numerical error ({type(e).__name__}) — skipping block")
        renderer.clamp_params()
        # The internal partial step may have left the renderer in a worse
        # state than where this block started.  Roll back to best if so —
        # otherwise subsequent blocks inherit the degraded params and
        # cascade-fail (observed in 50-sample run: one block crashing
        # poisoned the rest of the CD sweep).
        if best is not None and best['sd'] is not None:
            _restore_if_worse(best, renderer, orig, target, loss_fn,
                              tag=f'{tag}/post-error', verbose=verbose)
        return history

    if state['last'] is not None:
        log = state['last']
        history.append({'phase': f'lbfgs/{tag}', **log})
        if verbose:
            print(f"  [lbfgs/{tag}] "
                  f"L={log['total']:.4f} "
                  f"lin={log.get('lin_l1', 0.0):.4f} "
                  f"ΔE={log.get('de2000', 0.0):.3f}")
    return history


# ============================================================
# Coordinate descent
# ============================================================

def coordinate_descent(renderer, orig, target, loss_fn,
                        block_order: list[str],
                        sweeps: int = 2,
                        block_iters: int = 25,
                        verbose: bool = True,
                        best: dict | None = None):
    """Block-coordinate descent.

    Each *sweep* visits every block in ``block_order`` and runs a short
    L-BFGS solve restricted to that block's parameters; other parameters
    are held fixed (their grads are simply ignored by the L-BFGS instance).
    """
    blocks = parameter_blocks(renderer)
    history = []
    active_order = [bk for bk in block_order if bk in blocks]
    for s in range(sweeps):
        if verbose:
            print(f"  [coord sweep {s + 1}/{sweeps}] order = {active_order}")
        for bk in active_order:
            history += lbfgs_polish(
                renderer, orig, target, loss_fn,
                params=blocks[bk], max_iter=block_iters,
                tag=f"sweep{s + 1}/{bk}", verbose=verbose,
                best=best,
            )
    return history


# ============================================================
# Top-level fit
# ============================================================

DEFAULT_BLOCK_ORDER = ['wb', 'color', 'hsl', 'curves', 'local']


# Polish-stage weight overrides for the chroma+detail micro-fit.  See
# ``_fit_attempt`` for usage rationale.
POLISH_BOOST_WEIGHTS = {
    'chroma_pa': 20.0,   # 4× the default 5.0 — actively penalise residual graying
    'detail_pa': 20.0,   # 4× the default — actively penalise residual softness
    'ab_mean':    1.0,   # 2× the default 0.5 — anchor hue *direction* during
                         # polish.  chroma_pa is magnitude-only, so without
                         # ab_mean the boost stage happily adds chroma in the
                         # wrong direction (cool/cyan instead of warm).  Kept
                         # to 2× (not 4×) because combining it with the strong
                         # chroma_pa boost was making the optimiser pay for
                         # chroma via exposure↑ → highlight blow-out.
    'param_reg':  0.25,  # 0.5 → 0.25: modest relaxation of the highlight
                         # blow-out / detail-destruction floors.  Was 0.05 but
                         # that fully removed the exposure cap, letting the
                         # boosted chroma drivers push exposure to +2.5 EV and
                         # blow whites on bright subjects (e.g. wedding
                         # dresses) — chroma should be paid for by mixer/cg/wb,
                         # not by global brightness.
}


def _apply_init_strategy(renderer, *, strategy: str, warm_init: dict | None,
                         amortized_model, orig_source, edited_path,
                         device: str, verbose: bool = True):
    """Apply one of three initialisation strategies in place on ``renderer``.

    - ``warm``    : warm_init LR params + amortised CNN (current default).
    - ``zero``    : identity (all params at zero/neutral).  Useful for samples
                    where the CNN's prior dominates and traps Adam in a
                    wrong basin.
    - ``perturb`` : warm + amortised + small Gaussian perturbation on every
                    parameter.  Tests whether the CNN init's basin has a
                    nearby better minimum that Adam can't reach with eps=1e-3
                    on a plateau.
    """
    if strategy == 'zero':
        if verbose:
            print("  init=zero (identity)")
        return
    # Both 'warm' and 'perturb' apply the warm + amortized init.
    if warm_init:
        renderer.init_from_lr_params(warm_init)
        if verbose:
            print("  warm-started from LR params")
    if amortized_model is not None:
        from differentiable_renderer.amortized_init import predict as _amort_predict
        small = 128
        orig_s = load_image(orig_source, long_side=small).to(device)
        edit_s = load_image(edited_path, long_side=small).to(device)
        if orig_s.shape[2:] != edit_s.shape[2:]:
            orig_s = F.interpolate(orig_s, size=edit_s.shape[2:],
                                    mode='bilinear', align_corners=False)
        theta_dict = _amort_predict(amortized_model, orig_s, edit_s)
        renderer.init_from_lr_params(theta_dict)
        if verbose:
            print("  amortised CNN initialiser applied")
    if strategy == 'perturb':
        # Small noise on each parameter so multi-start explores a different
        # basin than the deterministic warm+amortized init.  Magnitude is
        # chosen as ~5% of the parameter's natural range.
        with torch.no_grad():
            for name, p in renderer.named_parameters():
                if not p.requires_grad:
                    continue
                # Sniff the natural range: if param values typically span
                # [-100, 100] (most LR sliders), use std=5.0; if they're
                # tone-curve coefficients (small), use std=0.05.
                if 'curve' in name or 'cg_' in name:
                    std = 0.05
                elif 'temp' in name:
                    std = 200.0
                else:
                    std = 5.0
                p.add_(torch.randn_like(p) * std)
            renderer.clamp_params()
        if verbose:
            print("  init=perturb (warm + amortized + Gaussian noise)")


def fit_single(orig_source,
               edited_path: str,
               *,
               warm_init: dict | None = None,
               amortized_model=None,
               device: str = 'cuda',
               scales: list = (256, 512, 1024),
               adam_iters_per_scale: int = 800,
               cd_sweeps: int = 5,
               cd_block_iters: int = 25,
               polish_iters: int = 200,
               micro_fit_iters: int = 200,
               loss_weights: dict | None = None,
               n_restarts: int = 2,
               restart_strategies: list[str] | None = None,
               skip_scale_threshold: float | None = None,
               lbfgs_pre_warm_iters: int = 15,
               verbose: bool = True):
    """Multi-start wrapper around ``_fit_attempt``.

    For per-image overfit we want to explore multiple basins of the loss
    landscape, since Adam + L-BFGS jointly are not strong enough to escape
    a wrong basin chosen by the CNN init.  This wrapper runs N attempts
    (each with a different ``init_strategy``) and returns the renderer
    whose final full-resolution loss is lowest.

    The cost is ``n_restarts``× the single-fit time, but for samples that
    were stuck in a wrong basin (e.g. 023d187f with sat=0, cl=0, tx=0)
    multi-start is the cheapest way to find the right one.
    """
    if restart_strategies is None:
        restart_strategies = ['warm', 'zero', 'perturb'][:n_restarts]
    assert len(restart_strategies) == n_restarts

    attempts = []
    # Pre-load full-res images once for final ranking (reuse across attempts).
    eval_orig = load_image(orig_source).to(device)
    eval_target = load_image(edited_path).to(device)
    if eval_orig.shape[2:] != eval_target.shape[2:]:
        eval_orig = F.interpolate(eval_orig, size=eval_target.shape[2:],
                                  mode='bilinear', align_corners=False)
    for ri, strategy in enumerate(restart_strategies):
        if verbose:
            print(f"\n  ═══ restart {ri+1}/{n_restarts}  init={strategy} ═══")
        # Use a deterministic seed per restart so reruns are reproducible.
        torch.manual_seed(1234 + 7919 * ri)
        renderer, history = _fit_attempt(
            orig_source, edited_path,
            warm_init=warm_init, amortized_model=amortized_model,
            init_strategy=strategy,
            device=device, scales=scales,
            adam_iters_per_scale=adam_iters_per_scale,
            cd_sweeps=cd_sweeps, cd_block_iters=cd_block_iters,
            polish_iters=polish_iters,
            micro_fit_iters=micro_fit_iters,
            loss_weights=loss_weights,
            skip_scale_threshold=skip_scale_threshold,
            lbfgs_pre_warm_iters=lbfgs_pre_warm_iters,
            verbose=verbose,
        )
        # Rank by full-resolution loss using DEFAULT (non-boosted) weights —
        # this is the metric that matches the user's perception, not the
        # last polish phase's boosted total.
        with torch.no_grad():
            loss_fn_eval = CompositeLoss(weights=loss_weights, renderer=renderer).to(device)
            rendered = renderer(eval_orig)
            full_loss, _ = loss_fn_eval(rendered, eval_target)
        attempts.append({
            'renderer': renderer, 'history': history,
            'loss': full_loss.item(), 'strategy': strategy,
        })
        if verbose:
            print(f"  ── restart {ri+1} ({strategy}) full-res loss = "
                  f"{full_loss.item():.4f}")

    # Pick best by full-res loss
    best_attempt = min(attempts, key=lambda x: x['loss'])
    if verbose:
        scores = ", ".join(f"{a['strategy']}={a['loss']:.3f}" for a in attempts)
        print(f"\n  ✓ best of {n_restarts}: {best_attempt['strategy']} "
              f"({best_attempt['loss']:.4f})  |  all: {scores}")
    return best_attempt['renderer'], best_attempt['history']


def _fit_attempt(orig_source,
                 edited_path: str,
                 *,
                 warm_init: dict | None = None,
                 amortized_model=None,
                 init_strategy: str = 'warm',
                 device: str = 'cuda',
                 scales: list = (256, 512, 1024),
                 adam_iters_per_scale: int = 800,
                 cd_sweeps: int = 5,
                 cd_block_iters: int = 25,
                 polish_iters: int = 200,
                 micro_fit_iters: int = 200,
                 loss_weights: dict | None = None,
                 skip_scale_threshold: float | None = None,
                 lbfgs_pre_warm_iters: int = 15,
                 verbose: bool = True):
    """One attempt of the full fit pipeline (used by multi-start).

    Parameters of interest for scale-transition stability:

    skip_scale_threshold
        When set, after each Adam scale completes, if ``best['loss']``
        (evaluated at that scale) is below this value, skip remaining
        higher scales.  Use case: high-quality samples already converged
        at scale 256 — pushing to 512/1024 burns compute on alignment
        noise rather than improving the fit.  Set to ``None`` (default)
        to always run all scales.

    lbfgs_pre_warm_iters
        Iterations of L-BFGS run *before* Adam at scale_idx > 0.  L-BFGS
        uses curvature-aware line search rather than Adam's ``sign(grad)``
        first-step kick, so it can ease parameters off the prior scale's
        delicate balance onto the new scale's surface without launching
        them over a clamp cliff.  Set to 0 to disable.
    """
    renderer = DifferentiableRenderer().to(device)
    _apply_init_strategy(
        renderer, strategy=init_strategy, warm_init=warm_init,
        amortized_model=amortized_model,
        orig_source=orig_source, edited_path=edited_path,
        device=device, verbose=verbose,
    )

    # ---- Tiered B&W detection ----
    # 'full_bw': pure grayscale → pin chroma at neutral, run in bw_mode.
    # 'near_bw': single-tone desaturated edit → pre-init saturation/vibrance
    #            around the target's likely value so Adam doesn't waste
    #            iterations climbing out of the "barely desaturated" basin
    #            that ``lin_l1`` is happy to settle into.
    edit_probe = load_image(edited_path, long_side=256).to(device)
    bw_level = detect_bw_level(edit_probe)
    # ``bw_target`` flag (distinct from ``bw_mode``): true for both full_bw and
    # near_bw, false for colour targets.  Used by ``losses._param_reg`` to drop
    # the positive-contrast penalty — contrast is the primary BW-edit knob
    # (Lightroom-typical +20~+40), and the highlight-blowout case the penalty
    # was designed for is already covered by the ``hl_over`` term.
    renderer.bw_target = (bw_level != 'color')
    if bw_level == 'full_bw':
        renderer.bw_mode = True
        with torch.no_grad():
            # Pin every chroma-introducing channel.  Zeroing alone is
            # not enough: Adam will happily push these back as a way to
            # buy lab_ab loss against high-freq chroma leakage from
            # _detail_transfer.  ``requires_grad=False`` below makes them
            # invisible to the optimiser entirely.
            renderer.saturation.fill_(-100.0)
            renderer.vibrance.fill_(-100.0)
            renderer.mixer_hue.zero_()
            renderer.mixer_sat.zero_()
            renderer.mixer_lum.zero_()
            renderer.temp.zero_()
            renderer.tint.zero_()
            for cg in (renderer.cg_shadow, renderer.cg_midtone,
                       renderer.cg_highlight):
                cg.zero_()
            # Pre-init contrast positive: BW edits in this dataset almost
            # always sit at +20~+40, but the optimiser starts from CNN init
            # (typically near 0) and ``lin_l1`` has a wide plateau around 0
            # that Adam doesn't traverse quickly.  Pre-seeding 25 puts Adam
            # in the right basin from step 0; the lower bound is unclamped
            # so Adam can pull back if a specific edit wants less.
            if float(renderer.contrast.item()) < 15.0:
                renderer.contrast.fill_(25.0)
        # Freeze: optimisers ignore params with requires_grad=False
        # (their .grad stays None, so step() is a no-op for them).
        # mixer_lum is ALSO frozen now: with the BW shortcut feeding a
        # grayscale orig, every pixel's hue collapses to 0 (the "red"
        # bucket), making mixer_lum a degenerate global-L shift that
        # competes with exposure and the tone curve.  Leaving it free
        # cost ~7/8 of the param's gradient budget for zero benefit and
        # introduced an identifiability issue (exposure +1 vs mixer_lum
        # all +X have similar effect, so Adam wandered).
        for p in (renderer.saturation, renderer.vibrance,
                  renderer.mixer_hue, renderer.mixer_sat,
                  renderer.mixer_lum,
                  renderer.temp, renderer.tint,
                  renderer.cg_shadow, renderer.cg_midtone,
                  renderer.cg_highlight):
            p.requires_grad_(False)
        if verbose:
            print("  B&W target detected — chroma + mixer_lum pinned and frozen "
                  "(bw_mode=True, 20 free params)")
    elif bw_level == 'near_bw':
        # IMPORTANT: do not push saturation past the factor=0 clamp boundary
        # (which is at saturation=-50 with the *2.0 coefficient).  Past that,
        # ``factor.clamp(min=0)`` masks the gradient and Adam can't relax
        # saturation back up to admit a slight tint, even if doing so would
        # reduce loss.  -40 keeps factor at ~0.2 — heavily desaturated but
        # gradient flows freely so Adam can adjust either direction.
        with torch.no_grad():
            renderer.saturation.fill_(-40.0)
            renderer.vibrance.fill_(-30.0)
            # Same contrast pre-init as full_bw — near-BW edits share the
            # "boost contrast / crush blacks" idiom (silhouettes against
            # bright skies, etc.).  See full_bw branch for rationale.
            if float(renderer.contrast.item()) < 15.0:
                renderer.contrast.fill_(25.0)
        if verbose:
            print("  near-B&W target detected — saturation -40, vibrance -30, contrast +25")

    # ---- Vignette pre-initialisation ----
    # The heuristic is grounded in the *actual* target's corner/centre luminance
    # ratio, while the CNN's prediction comes from a generic prior.  Override
    # the CNN whenever the heuristic suggests *meaningfully stronger* corner
    # darkening — empirically Adam can't push past the CNN's local minimum on
    # its own (loss landscape around vignette is flat-ish away from the answer).
    v_init_heuristic = estimate_vignette(edit_probe)
    v_init_cnn       = float(renderer.vignette.item())
    if v_init_heuristic <= -10.0 and v_init_heuristic < v_init_cnn - 5.0:
        with torch.no_grad():
            renderer.vignette.fill_(v_init_heuristic)
        if verbose:
            print(f"  vignette: CNN={v_init_cnn:+.0f} → heuristic={v_init_heuristic:+.0f} "
                  f"(corner/centre lum ratio)")

    # ---- Detail-positive initialisation ----
    # Lightroom edits in this dataset are dominated by detail-enhancing
    # operations (clarity / texture / sharpening positive).  But left alone the
    # optimiser persistently chooses 0 / negative because lab_ab penalises edge
    # shifts.  Pre-seeding clarity / texture positive (combined with the [0,100]
    # clamp) means Adam can only stay there or push higher — never destroy
    # detail.  Skip if bw_mode (B&W edits don't necessarily want detail boost)
    # or if CNN already predicted strong positive values.
    if not renderer.bw_mode:
        with torch.no_grad():
            if float(renderer.clarity.item()) < 10.0:
                renderer.clarity.fill_(20.0)
            if float(renderer.texture.item()) < 5.0:
                renderer.texture.fill_(8.0)
        if verbose:
            print(f"  detail: clarity={renderer.clarity.item():.0f}, "
                  f"texture={renderer.texture.item():.0f} (forced positive)")

    # ---- Chroma-positive initialisation (color targets only) ----
    # Symmetric to the detail-positive trick: when the target is genuinely
    # colourful, Adam empirically lands on vibrance ~ -20 / saturation ~ -5
    # for many samples (it satisfies the chroma residual through HSL mixer
    # or WB shifts instead).  Pre-seeding the global chroma knobs positive
    # gives Adam a head start in the right basin; combined with the
    # vibrance/saturation neg_abs term in param_reg, it makes "push the
    # chroma knobs up" the path-of-least-resistance.  bw/near_bw skip
    # this — for those, negative chroma is the target value.
    if bw_level == 'color':
        with torch.no_grad():
            if float(renderer.vibrance.item()) < 5.0:
                renderer.vibrance.fill_(10.0)
            if float(renderer.saturation.item()) < 3.0:
                renderer.saturation.fill_(5.0)
        if verbose:
            print(f"  chroma: vibrance={renderer.vibrance.item():.0f}, "
                  f"saturation={renderer.saturation.item():.0f} (forced positive for color target)")

    # ---- Loss weights: tiered profile based on bw_level ----
    # full_bw  → BW_LOSS_WEIGHTS:      chroma losses zeroed, luminance boosted
    # near_bw  → NEAR_BW_LOSS_WEIGHTS: chroma reduced (anchor tint only),
    #                                  luminance boosted toward BW levels
    # color    → caller's loss_weights (DEFAULT_WEIGHTS if None)
    # Caller's ``loss_weights`` override always wins on top of the profile.
    if renderer.bw_mode:
        weights_for_loss = dict(CompositeLoss.BW_LOSS_WEIGHTS)
        if loss_weights:
            weights_for_loss.update(loss_weights)
        if verbose:
            print(f"  loss profile: BW (lin_l1=1.5, lum_cdf=2, ms_ssim=5, "
                  f"detail_pa=8, chroma losses=0)")
    elif bw_level == 'near_bw':
        weights_for_loss = dict(CompositeLoss.NEAR_BW_LOSS_WEIGHTS)
        if loss_weights:
            weights_for_loss.update(loss_weights)
        if verbose:
            print(f"  loss profile: near-BW (lum_cdf=1.8, ms_ssim=4, "
                  f"detail_pa=7, chroma 30% of default)")
    else:
        weights_for_loss = loss_weights
    loss_fn = CompositeLoss(weights=weights_for_loss, renderer=renderer).to(device)
    history = []
    # Best-state tracking — every phase below updates this in place when it
    # observes a new lowest finite total loss.  Acts as a safety net: if any
    # downstream phase derails (e.g. L-BFGS hitting a numerical singularity
    # and producing NaN, which clamp_params then zeroes), we can still fall
    # back to the best state ever seen.
    best = {'loss': float('inf'), 'sd': None}

    # ---- Multi-scale Adam ----
    # Aggressive LR decay (0.2 per scale, was 0.5): once the coarse pass has
    # converged, the next-finer pass should *refine*, not re-explore.  At
    # 0.5 the second scale still moved params by 0.5/step which was enough
    # to blow up the converged solution within ~50 steps when the loss
    # surface is sharp (e.g. with ms_ssim weight 3.0).  0.2 keeps the per
    # step displacement under 0.1 — small enough to walk along the local
    # optimum without escaping it.
    blocks = parameter_blocks(renderer)
    # BW shortcut: when the target is grayscale, also feed a grayscale
    # version of orig so the renderer optimises a gray→gray problem.
    # Why: with bw_mode True the pipeline already collapses chroma
    # (saturation pinned at -100), but the *gradient* of the loss still
    # has to back-propagate through that collapse, and dead chroma
    # channels make the curvature of luminance-only operators
    # (curves/exposure/highlights/shadows) noisier than they need to be.
    # Matching gray→gray removes that noise entirely; the params we
    # learn are deployment-equivalent because at saturation=-100 the
    # renderer's output is independent of the input chroma.
    grayscale_input = (renderer.bw_mode is True)
    if grayscale_input and verbose:
        print("  BW shortcut: feeding grayscale orig to fitter")
    for scale_idx, sc in enumerate(scales):
        orig   = load_image(orig_source,  long_side=sc,
                            grayscale=grayscale_input).to(device)
        target = load_image(edited_path,  long_side=sc).to(device)
        if orig.shape[2:] != target.shape[2:]:
            orig = F.interpolate(orig, size=target.shape[2:],
                                 mode='bilinear', align_corners=False)
        if verbose:
            print(f"  ── scale {sc} ──")
        # Re-anchor best to this resolution's loss surface (different
        # scales = different ms_ssim / loss values for the same params).
        _rebaseline_best(best, renderer, orig, target, loss_fn)
        # If the prior phase ended worse than best, revert before kicking
        # off Adam at the new scale — gives Adam a sane starting point
        # even after a divergent previous scale.
        _restore_if_worse(best, renderer, orig, target, loss_fn,
                          tag=f'pre-s{sc}', verbose=verbose)
        lr_decay = 0.2 ** scale_idx
        # B&W-aware LR multipliers.  When chroma is pinned/frozen, the only
        # axes that move pixels are the luminance ones — but the default LRs
        # were tuned for the colour case where chroma sucks up most of the
        # gradient.  Bumping curves/local 2-3× in B&W lets contrast, vignette,
        # whites/blacks, the tone curve, and the detail knobs make faster
        # progress against the now-undivided lin_l1 + lum_cdf + ms_ssim signal.
        # User-requested tilt: contrast & vignette specifically.
        if renderer.bw_mode:
            bw_lr_mul = {
                'wb':     0.0,   # frozen, no-op
                'color':  0.0,   # frozen, no-op
                'hsl':    0.0,   # frozen, no-op
                'curves': 3.0,   # exposure / contrast / whites / blacks / curve
                'local':  2.5,   # highlights / shadows / clarity / texture /
                                 # vignette / dehaze / sharpening / noise_red
            }
            lr_groups = adam_param_groups(
                blocks,
                group_lrs={k: v * lr_decay * bw_lr_mul.get(k, 1.0)
                           for k, v in DEFAULT_GROUP_LRS.items()},
            )
        else:
            lr_groups = adam_param_groups(
                blocks,
                group_lrs={k: v * lr_decay
                           for k, v in DEFAULT_GROUP_LRS.items()},
            )
        renderer.clamp_params()

        # Scale-transition warm-in: brief L-BFGS pass before Adam at
        # scale_idx > 0.  The prior-scale best is loaded above by
        # _restore_if_worse — it's a delicate balance tuned for that scale's
        # smooth surface, and Adam's iter-0 ``sign(grad)`` kick was launching
        # it off a clamp cliff into a zero-gradient flat region (observed:
        # adam@512/1024 always diverged within 1-3 steps on this benchmark,
        # then sat at loss ~20 for 120 iters until the phase broke).  L-BFGS
        # strong-Wolfe line search takes curvature-aware steps that stay in
        # the basin while it adapts to the new scale's surface.
        if scale_idx > 0 and lbfgs_pre_warm_iters > 0:
            if verbose:
                print(f"  ── scale {sc} L-BFGS warm-in ({lbfgs_pre_warm_iters} iters) ──")
            history += lbfgs_polish(
                renderer, orig, target, loss_fn,
                params=None, max_iter=lbfgs_pre_warm_iters,
                tag=f's{sc}/warmin', verbose=verbose, best=best,
            )
            _restore_if_worse(best, renderer, orig, target, loss_fn,
                              tag=f'post-warmin-s{sc}', verbose=verbose)

        # Gentler warmup for scale_idx > 0: 1% → 100% over up to 60 steps
        # rather than 5% → 100% over 20 steps.  Even after the L-BFGS
        # warm-in the Adam iter-0 step can be too large because Adam has no
        # second-moment history; the slower ramp lets ``v_hat`` build up
        # before LR reaches base.
        if scale_idx == 0:
            warmup_kw = dict(warmup_start=0.05, warmup_frac=0.10, warmup_cap=20)
        else:
            warmup_kw = dict(warmup_start=0.01, warmup_frac=0.30, warmup_cap=60)

        history += adam_phase(
            renderer, orig, target, loss_fn,
            iters=adam_iters_per_scale,
            lr_groups=lr_groups,
            scale=sc, verbose=verbose, best=best,
            **warmup_kw,
        )

        # Early skip: if this scale's best is already excellent, skip
        # higher scales.  Pushing to finer resolutions on a well-fit sample
        # mostly amplifies crop-alignment noise — the L-BFGS sweeps and
        # divergence resets confirm the gradient at higher scales points
        # *away* from the current best.
        if skip_scale_threshold is not None and best is not None:
            _rebaseline_best(best, renderer, orig, target, loss_fn)
            if best['loss'] < skip_scale_threshold and scale_idx + 1 < len(scales):
                if verbose:
                    print(f"  ── skipping scales {list(scales[scale_idx+1:])}: "
                          f"best={best['loss']:.4f} < threshold={skip_scale_threshold} ──")
                break

    # ---- Coordinate descent on the largest scale ----
    if verbose:
        print("  ── coordinate descent (L-BFGS / strong Wolfe) ──")
    # Re-anchor best to the final scale and revert if Adam left us worse.
    _rebaseline_best(best, renderer, orig, target, loss_fn)
    _restore_if_worse(best, renderer, orig, target, loss_fn,
                      tag='pre-cd', verbose=verbose)
    history += coordinate_descent(
        renderer, orig, target, loss_fn,
        block_order=DEFAULT_BLOCK_ORDER,
        sweeps=cd_sweeps, block_iters=cd_block_iters,
        verbose=verbose, best=best,
    )

    # ---- Final global L-BFGS polish ----
    if verbose:
        print("  ── final global L-BFGS polish ──")
    _restore_if_worse(best, renderer, orig, target, loss_fn,
                      tag='pre-polish', verbose=verbose)
    history += lbfgs_polish(
        renderer, orig, target, loss_fn,
        params=None, max_iter=polish_iters, tag='polish', verbose=verbose,
        best=best,
    )

    # ---- Residual-driven chroma+detail micro-fit ----
    # The default loss balance trades a little chroma/detail residual to win
    # on lab_ab / hue_de / ms_swd.  At this point the global structure is
    # already correct, so we can flip the trade: boost chroma_pa/detail_pa
    # 4×, slash param_reg, and let Adam push parameter values that the
    # main phase regularised away (extreme saturation, full-strength
    # texture/clarity).  Best-state is rebaselined to this loss surface
    # so we don't accidentally restore a state that had higher boosted loss.
    #
    # Skip for B&W: ``chroma_pa`` is a degenerate zero signal, and the BW
    # loss profile already sets detail_pa=8 (vs the colour default 5), so
    # the marginal benefit of another phase doesn't outweigh the
    # boost-rebaseline complexity.
    if micro_fit_iters > 0 and not renderer.bw_mode:
        boost_weights = dict(CompositeLoss.DEFAULT_WEIGHTS)
        if loss_weights:
            boost_weights.update(loss_weights)
        boost_weights.update(POLISH_BOOST_WEIGHTS)
        boost_loss_fn = CompositeLoss(weights=boost_weights,
                                      renderer=renderer).to(device)
        _restore_if_worse(best, renderer, orig, target, loss_fn,
                          tag='pre-micro', verbose=verbose)
        # Rebaseline best to the boosted loss so the micro-fit phase has
        # a comparable reference point — otherwise its restore-on-divergence
        # logic would compare apples (boosted) to oranges (default loss).
        _rebaseline_best(best, renderer, orig, target, boost_loss_fn)
        if verbose:
            print(f"  ── chroma+detail micro-fit ({micro_fit_iters} iters, "
                  f"chroma_pa=20, detail_pa=20) ──")
        # Reuse the latest scale's images.  Adam params: small LR (0.5×
        # the smallest scale's LR) since we're refining, not exploring.
        micro_lr_groups = adam_param_groups(
            blocks,
            group_lrs={k: v * 0.05  # gentle refinement
                       for k, v in DEFAULT_GROUP_LRS.items()},
        )
        history += adam_phase(
            renderer, orig, target, boost_loss_fn,
            iters=micro_fit_iters,
            lr_groups=micro_lr_groups,
            scale=scales[-1], verbose=verbose, best=best,
            phase_tag='micro_adam',
            anneal_patience=20,  # tighter since we're already converged
        )
        # Re-rank best on the *default* loss before final restore so the
        # caller (multi-start) sees a comparable score across attempts.
        _rebaseline_best(best, renderer, orig, target, loss_fn)

    # ---- Restore best observed state ----
    # If the final polish (or any preceding phase) ended at a higher loss
    # than the best seen — typical when L-BFGS line search overshoots and
    # the closure's last call lands on a worse point — fall back to the
    # snapshot.  bw_mode is a Python attribute on the module, not a tensor
    # in state_dict, so it survives load_state_dict unchanged.
    if best['sd'] is not None:
        with torch.no_grad():
            rendered_now = renderer(orig)
            cur_loss, _ = loss_fn(rendered_now, target)
        if (not torch.isfinite(cur_loss)) or cur_loss.item() > best['loss']:
            renderer.load_state_dict(best['sd'])
            renderer.clamp_params()
            if verbose:
                print(f"  ── restored best params (loss={best['loss']:.4f}, "
                      f"final phase had {cur_loss.item():.4f}) ──")

    return renderer, history


# ============================================================
# Data loading
# ============================================================

def load_sample(data_dir: str, sample_id: str, version: int = 0):
    sample_dir = os.path.join(data_dir, sample_id)
    orig_path   = os.path.join(sample_dir, 'original_captured_cropped.png')
    edited_path = os.path.join(sample_dir, f'version_{version}', 'main_image.png')

    lr_params = {}
    if os.path.exists(os.path.join(sample_dir, 'combined_details.json')):
        try:
            from step1_parse_params import extract_params
            detail_json = os.path.join(sample_dir, f'version_{version}', 'details.json')
            if os.path.exists(detail_json):
                lr_params = extract_params(detail_json)
        except Exception:
            pass
    return orig_path, edited_path, lr_params


def discover_versions(data_dir: str, sample_id: str) -> list[int]:
    """发现某个样本目录下所有可用的 version 编号（升序）。

    扫描 ``{data_dir}/{sample_id}/version_<N>/``，只保留确实含 ``main_image.png``
    的版本——有些样本的 version 子目录可能只有 details.json 而缺图。
    """
    import re
    sample_dir = os.path.join(data_dir, sample_id)
    if not os.path.isdir(sample_dir):
        return []
    versions = []
    for name in os.listdir(sample_dir):
        m = re.fullmatch(r'version_(\d+)', name)
        if m and os.path.exists(
                os.path.join(sample_dir, name, 'main_image.png')):
            versions.append(int(m.group(1)))
    return sorted(versions)


def fit_and_save(args, amortized_model, sid: str, version: int):
    """拟合单个 (sample, version) 并写入 ``{output_dir}/{sid}/*_v{version}.*``。

    原图 ``original_captured_cropped.png`` 在所有 version 间共享，但 crop 对齐与
    拟合都按各 version 的 ``main_image.png`` 单独进行（不同 version 的 LR crop
    可能不同）。产物写入 ``{output_dir}/{sid}/version_{n}/``，镜像数据集结构。
    """
    # 输出镜像数据集布局：{output_dir}/{sid}/version_{n}/...，与
    # {data_dir}/{sid}/version_{n}/ 平行。文件名去掉 _v{n} 后缀（目录已区分）。
    result_dir = os.path.join(args.output_dir, sid, f'version_{version}')
    param_file = os.path.join(result_dir, 'params.json')

    orig_path, edited_path, lr_params = load_sample(args.data_dir, sid, version)
    if not os.path.exists(orig_path) or not os.path.exists(edited_path):
        print(f"  [v{version}] SKIP (缺 original 或 version_{version}/main_image.png)")
        return

    print(f"  ── version {version} ──")
    # Crop 对齐：按本 version 的编辑图各自对齐（同一原图、不同 version 可能不同 crop）。
    orig_pil = Image.open(orig_path).convert('RGB')
    edited_pil = Image.open(edited_path).convert('RGB')
    force_align = not args.no_force_align
    aspect_diff = needs_crop(orig_pil.size, edited_pil.size)
    if force_align or aspect_diff:
        orig_source = align_original_to_edited(
            orig_path, edited_path, force_sift=force_align)
        crop_aligned_flag = True
        print(f"  Crop aligned ({'forced SIFT' if force_align else 'aspect-gated'}): "
              f"{orig_pil.size} → {orig_source.size} (target {edited_pil.size})")
    else:
        orig_source = orig_path
        crop_aligned_flag = False

    t0 = time.time()
    renderer, history = fit_single(
        orig_source, edited_path,
        warm_init=lr_params if args.warm_start else None,
        amortized_model=amortized_model,
        device=args.device,
        scales=args.scales,
        adam_iters_per_scale=args.adam_iters,
        cd_sweeps=args.cd_sweeps,
        cd_block_iters=args.cd_block_iters,
        polish_iters=args.polish_iters,
        micro_fit_iters=args.micro_fit_iters,
        n_restarts=args.n_restarts,
        skip_scale_threshold=args.skip_scale_threshold,
        lbfgs_pre_warm_iters=args.lbfgs_pre_warm_iters,
    )
    elapsed = time.time() - t0

    # ---- 落盘 ----
    os.makedirs(result_dir, exist_ok=True)
    params = renderer.get_params_dict()
    params['_crop_aligned'] = crop_aligned_flag
    with open(param_file, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    with torch.no_grad():
        orig_full   = load_image(orig_source).to(args.device)
        target_full = load_image(edited_path).to(args.device)
        if orig_full.shape[2:] != target_full.shape[2:]:
            orig_full = F.interpolate(orig_full, size=target_full.shape[2:],
                                       mode='bilinear', align_corners=False)
        rendered_full = renderer(orig_full)
        save_image(rendered_full,
                   os.path.join(result_dir, 'rendered.png'))
        save_comparison(orig_full, rendered_full, target_full,
                        os.path.join(result_dir, 'compare.png'))
        _, final_log = CompositeLoss().to(args.device)(rendered_full, target_full)

    with open(os.path.join(result_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    torch.save(renderer.state_dict(),
               os.path.join(result_dir, 'renderer.pt'))

    # ---- fit_meta.json：质量元数据，便于过滤退化样本 ----
    # param_l1 = Σ|θ|（identity 严格为 0）。退化样本 = 拟合后 loss 仍高 *且* 参数
    # 几乎没动（≈identity）——典型是局部光效/渐变/蒙版等全局参数表达不了的 target，
    # 它们对 SFT 是有害监督（教模型"这种输入别调整"），应从训练集剔除。
    param_l1 = sum(p.detach().abs().sum().item() for p in renderer.parameters())
    n_active = int(sum((p.detach().abs() > 1e-3).sum().item()
                       for p in renderer.parameters()))
    final_loss = float(final_log['total'])
    degenerate = (final_loss > args.degenerate_loss
                  and param_l1 < args.degenerate_param_l1)
    meta = {
        'sample_id': sid,
        'version': version,
        'final_loss': round(final_loss, 4),
        'param_l1': round(param_l1, 4),
        'n_active_params': n_active,
        'bw_mode': bool(getattr(renderer, 'bw_mode', False)),
        'crop_aligned': crop_aligned_flag,
        'degenerate': degenerate,
        'loss_components': {k: round(float(v), 4) for k, v in final_log.items()},
        'fit_seconds': round(elapsed, 1),
    }
    with open(os.path.join(result_dir, 'fit_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    flag = '  ⚠️ DEGENERATE(拟合≈identity, 建议过滤)' if degenerate else ''
    print(f"  [v{version}] full-res: total={final_log['total']:.4f}  "
          f"lin={final_log['lin_l1']:.4f}  ΔE={final_log['de2000']:.3f}  "
          f"|θ|₁={param_l1:.1f} ({elapsed:.1f}s)  → {result_dir}{flag}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description='Fit differentiable renderer θ to match edited images')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--sample_id', default=None)
    p.add_argument('--version', type=int, default=0,
                   help='which edited version to fit when --all_versions is off '
                        '(default 0)')
    p.add_argument('--all_versions', action='store_true',
                   help='fit every version_* subdir of each sample (auto-discovered); '
                        'results mirror the dataset layout at '
                        '{output_dir}/{sample_id}/version_{n}/. Overrides --version.')
    p.add_argument('--degenerate_loss', type=float, default=5.0,
                   help='fit_meta.json marks a (sample,version) degenerate when its '
                        'full-res loss exceeds this AND param_l1 < --degenerate_param_l1 '
                        '(default 5.0). 退化样本拟合≈identity，不宜作监督目标。')
    p.add_argument('--degenerate_param_l1', type=float, default=5.0,
                   help='param-L1 (sum|θ|, identity=0) below which a high-loss fit is '
                        'flagged degenerate (default 5.0)')
    p.add_argument('--output_dir', default='fit_results')
    p.add_argument('--warm_start', action='store_true',
                   help='initialise from parsed LR params')
    p.add_argument('--scales', type=int, nargs='+', default=[256, 512, 1024])
    p.add_argument('--adam_iters', type=int, default=800,
                   help='Adam iterations per scale (default: 800; was 300 pre-overfit)')
    p.add_argument('--cd_sweeps', type=int, default=5,
                   help='Coordinate-descent sweeps over the block order (default: 5)')
    p.add_argument('--cd_block_iters', type=int, default=25)
    p.add_argument('--polish_iters', type=int, default=200,
                   help='Final L-BFGS polish iterations (default: 200)')
    p.add_argument('--micro_fit_iters', type=int, default=200,
                   help='Residual chroma+detail micro-fit iterations after polish '
                        '(0 disables; default 200)')
    p.add_argument('--n_restarts', type=int, default=2,
                   help='Multi-start: number of attempts per sample (default 2). '
                        'Each restart uses a different init strategy from '
                        '[warm, zero, perturb].')
    p.add_argument('--device', default='cuda')
    p.add_argument('--amortized_init', default=None,
                   help='path to AmortizedInitNet checkpoint (.pt) for warm-start')
    p.add_argument('--force', action='store_true',
                   help='overwrite existing results instead of skipping')
    p.add_argument('--limit', type=int, default=None,
                   help='only process the first N samples (after sorted listing); '
                        'useful for smoke tests')
    p.add_argument('--skip_scale_threshold', type=float, default=2.5,
                   help='After each Adam scale, if best loss at that scale is '
                        'below this value, skip higher scales (default 2.5; '
                        'set to a negative number to always run all scales). '
                        'Adam@512/@1024 are typically counterproductive once '
                        'adam@256 has converged — they amplify alignment noise.')
    p.add_argument('--lbfgs_pre_warm_iters', type=int, default=15,
                   help='L-BFGS iterations run before Adam at each scale '
                        'transition (scale_idx > 0).  Curvature-aware warm-in '
                        'prevents the iter-0 sign(grad) cliff-jump observed at '
                        'higher scales (default 15; set to 0 to disable).')
    p.add_argument('--no_force_align', action='store_true',
                   help='disable SIFT-based crop alignment for same-aspect-ratio '
                        'images (by default we always run SIFT alignment for '
                        'sub-pixel content matching, which materially affects '
                        'lab_ab / ms_ssim losses)')
    args = p.parse_args()
    # Treat negative threshold as "disabled" so user can opt out via CLI.
    if args.skip_scale_threshold is not None and args.skip_scale_threshold < 0:
        args.skip_scale_threshold = None

    amortized_model = None
    if args.amortized_init:
        from differentiable_renderer.amortized_init import load_model
        amortized_model = load_model(args.amortized_init, device=args.device)
        print(f"Loaded amortised init CNN from {args.amortized_init}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample_id:
        sample_ids = [args.sample_id]
    else:
        sample_ids = sorted(d for d in os.listdir(args.data_dir)
                            if os.path.isdir(os.path.join(args.data_dir, d)))
        if args.limit is not None:
            sample_ids = sample_ids[:args.limit]

    total = len(sample_ids)
    print(f"Fitting {total} sample(s) | scales {args.scales} | "
          f"adam {args.adam_iters} × {len(args.scales)} | "
          f"cd {args.cd_sweeps}×{args.cd_block_iters} | "
          f"polish {args.polish_iters} | micro {args.micro_fit_iters} | "
          f"restarts {args.n_restarts}")

    for idx, sid in enumerate(sample_ids):
        # 决定本样本要拟合哪些 version
        if args.all_versions:
            versions = discover_versions(args.data_dir, sid)
            if not versions:
                print(f"[{idx + 1}/{total}] SKIP {sid[:24]}…  (无 version_* 子目录)")
                continue
        else:
            versions = [args.version]

        # resume：跳过已产出 version_{n}/params.json 的 version（--force 时不跳）
        sample_out = os.path.join(args.output_dir, sid)
        todo = [v for v in versions
                if args.force or not os.path.exists(
                    os.path.join(sample_out, f'version_{v}', 'params.json'))]
        if not todo:
            continue   # 该样本所有 version 已完成 → 静默跳过

        print(f"\n{'=' * 60}\n[{idx + 1}/{total}] {sid}  "
              f"(versions {todo} / 共 {len(versions)})\n{'=' * 60}")
        for version in todo:
            try:
                fit_and_save(args, amortized_model, sid, version)
            except Exception as e:
                # 单个 version 失败不应中断整批拟合
                print(f"  [v{version}] ERROR {type(e).__name__}: {e}")
                continue

    print(f"\nDone.  Results in {args.output_dir}/")



if __name__ == '__main__':
    main()
