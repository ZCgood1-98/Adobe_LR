"""Amortised CNN initialiser for the differentiable renderer.

Trains a small siamese CNN that, given a pair (original sRGB, edited sRGB),
predicts the renderer parameter vector θ.  The training data is fully
synthetic: we sample θ from realistic priors, render the edit ourselves,
and use the (original, render(original, θ), θ) triple as a supervision
example.  This is the same recipe used by Exposure (Hu et al., 2018),
NeurOp (Wang et al., 2021) and Distort-and-Recover (Park et al., 2018).

Two public entry points:
  * ``train(data_dir, ...)``   builds a dataset of original PNGs from
    ``data_dir`` (any folder of images works; we ignore subdirectory
    structure), generates synthetic targets on the fly, and trains the CNN.
  * ``predict(model, orig, edited)``  returns a θ-dict ready to feed
    ``DifferentiableRenderer.init_from_lr_params``.

The CNN is intentionally small (~1.6M params) so it can be trained from
scratch in a few minutes on a modest GPU; its job is not to be perfect, but
to land within the basin of attraction of the gradient-based fitter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from differentiable_renderer.renderer import (
    DifferentiableRenderer, MIXER_NAMES, TEMP_NEUTRAL,
)


# ============================================================
# θ vector layout — one canonical flattening for I/O
# ============================================================

# (name, length) — keep stable: the CNN's output dimensionality depends on it.
THETA_LAYOUT = [
    ('exposure',     1),
    ('contrast',     1),
    ('highlights',   1),
    ('shadows',      1),
    ('whites',       1),
    ('blacks',       1),
    ('temp',         1),     # relative (K - 5500) / 50
    ('tint',         1),
    ('vibrance',     1),
    ('saturation',   1),
    ('texture',      1),
    ('clarity',      1),
    ('vignette',     1),
    ('dehaze',       1),
    ('sharpening',   1),
    ('noise_red',    1),
    ('mixer_hue',    8),
    ('mixer_sat',    8),
    ('mixer_lum',    8),
    ('cg_shadow',    3),
    ('cg_midtone',   3),
    ('cg_highlight', 3),
    ('tone_curve_dy', 8),
]
THETA_DIM = sum(n for _, n in THETA_LAYOUT)


def renderer_to_vector(r: DifferentiableRenderer) -> torch.Tensor:
    pieces = []
    for name, _ in THETA_LAYOUT:
        v = getattr(r, name).detach().reshape(-1)
        pieces.append(v)
    return torch.cat(pieces).cpu()


def vector_to_renderer(vec: torch.Tensor, r: DifferentiableRenderer):
    i = 0
    for name, n in THETA_LAYOUT:
        getattr(r, name).data.copy_(vec[i:i + n].reshape(getattr(r, name).shape))
        i += n


# ============================================================
# Priors  (matched to Lightroom user-edit distributions)
# ============================================================

# Each entry: (mean, std, lo, hi).  Sampling: clip(N(mean, std), lo, hi).
# These reproduce the rough shape of LR slider distributions in our scrape.
PRIORS = {
    'exposure':     (0.0, 0.6,   -3.0, 3.0),
    'contrast':     (0.0, 25.0,  -100.0, 100.0),
    'highlights':   (0.0, 35.0,  -100.0, 100.0),
    'shadows':      (0.0, 35.0,  -100.0, 100.0),
    'whites':       (0.0, 30.0,  -100.0, 100.0),
    'blacks':       (0.0, 30.0,  -100.0, 100.0),
    'temp':         (0.0, 30.0,  -200.0, 200.0),     # relative
    'tint':         (0.0, 12.0,  -150.0, 150.0),
    'vibrance':     (0.0, 25.0,  -100.0, 100.0),
    'saturation':   (0.0, 15.0,  -100.0, 100.0),
    'texture':      (0.0, 20.0,  -100.0, 100.0),
    'clarity':      (0.0, 20.0,  -100.0, 100.0),
    'vignette':     (0.0, 15.0,  -100.0, 100.0),
    'dehaze':       (0.0, 15.0,  -100.0, 100.0),
    'sharpening':   (0.0, 20.0,  0.0,    150.0),
    'noise_red':    (0.0, 15.0,  0.0,    100.0),
    'mixer_hue':    (0.0, 15.0,  -100.0, 100.0),
    'mixer_sat':    (0.0, 20.0,  -100.0, 100.0),
    'mixer_lum':    (0.0, 18.0,  -100.0, 100.0),
    'cg_shadow':    (0.0, 12.0,  -100.0, 100.0),
    'cg_midtone':   (0.0, 12.0,  -100.0, 100.0),
    'cg_highlight': (0.0, 12.0,  -100.0, 100.0),
    'tone_curve_dy': (0.0, 8.0,  -50.0, 50.0),
}


def sample_theta(rng: torch.Generator, device='cpu') -> dict[str, torch.Tensor]:
    """Sample a parameter dict from the priors."""
    out = {}
    for name, n in THETA_LAYOUT:
        mu, sd, lo, hi = PRIORS[name]
        v = torch.empty(n, device=device).normal_(mean=mu, std=sd, generator=rng)
        v = v.clamp(min=lo, max=hi)
        out[name] = v
    return out


def write_theta(theta: dict[str, torch.Tensor], r: DifferentiableRenderer):
    for name, _ in THETA_LAYOUT:
        getattr(r, name).data.copy_(theta[name].reshape(getattr(r, name).shape).to(
            getattr(r, name).device))


def theta_dict_to_vector(theta: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([theta[name].reshape(-1).cpu()
                      for name, _ in THETA_LAYOUT])


# Per-component standard deviations (for normalising the regression target)
THETA_STD = torch.tensor(
    [PRIORS[name][1] for name, n in THETA_LAYOUT for _ in range(n)],
    dtype=torch.float32,
).clamp(min=1.0)
THETA_HI = torch.tensor(
    [PRIORS[name][3] for name, n in THETA_LAYOUT for _ in range(n)],
    dtype=torch.float32,
)
THETA_LO = torch.tensor(
    [PRIORS[name][2] for name, n in THETA_LAYOUT for _ in range(n)],
    dtype=torch.float32,
)


# ============================================================
# Model
# ============================================================

def conv_bn_relu(c_in, c_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(inplace=True),
    )


class _ImgEncoder(nn.Module):
    """Tiny VGG-ish encoder, ~0.7M params, 128x128 input -> 256-d feature.

    Resolution-agnostic thanks to AdaptiveAvgPool2d(1); 128 is the default
    used at training & inference (see fit.py amortised init call site).
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            conv_bn_relu(3,    32, stride=2),     # 128
            conv_bn_relu(32,   64, stride=2),     # 64
            conv_bn_relu(64,   96, stride=2),     # 32
            conv_bn_relu(96,  128, stride=2),     # 16
            conv_bn_relu(128, 192, stride=2),     # 8
            conv_bn_relu(192, 256, stride=2),     # 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.layers(x)).flatten(1)


class AmortizedInitNet(nn.Module):
    """Siamese encoder + MLP head predicting θ.

    Predicts θ in a normalised space (per-component / std) and converts back
    inside ``predict``.  The split between the difference (edited - orig) and
    the absolute pair lets the head reason about both the *operation* and
    the original tonality (e.g. clarity has different effect on flat vs busy
    images).
    """

    def __init__(self, theta_dim: int = THETA_DIM):
        super().__init__()
        self.enc_orig = _ImgEncoder()
        self.enc_edit = _ImgEncoder()
        feat = 256
        self.head = nn.Sequential(
            nn.Linear(3 * feat, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, theta_dim),
        )
        # tanh + range-scale at the end keeps predictions inside the prior box.
        self.register_buffer('_lo', THETA_LO)
        self.register_buffer('_hi', THETA_HI)

    def forward(self, orig: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
        f_o = self.enc_orig(orig)
        f_e = self.enc_edit(edited)
        f_d = f_e - f_o
        h = torch.cat([f_o, f_e, f_d], dim=1)
        z = self.head(h)
        # Map to (lo, hi) via tanh
        c = (self._lo + self._hi) / 2.0
        s = (self._hi - self._lo) / 2.0
        return c + s * torch.tanh(z)


# ============================================================
# Synthetic data generation
# ============================================================

class SyntheticEditDataset(torch.utils.data.IterableDataset):
    """CPU-only (orig, θ_dict) producer; rendering happens in the train loop.

    The previous design rendered on GPU inside ``__iter__``, which forced
    ``num_workers=0`` (CUDA contexts don't survive ``fork``) and serialised
    all image I/O against the training step.  Now this class only does PIL
    decode + crop + tensor conversion — pure CPU work that parallelises
    cleanly across DataLoader workers.  The main process consumes the
    (orig_batch, theta_batch) pairs and renders on GPU once per step.
    """

    def __init__(self, image_dir: str, image_size: int = 128,
                 seed: int = 0):
        super().__init__()
        self.paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.paths.append(os.path.join(root, f))
        if not self.paths:
            raise RuntimeError(f"No images found under {image_dir}")
        self.image_size = image_size
        self.seed = seed

    def _load(self, path: str, py_rng: random.Random) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        s = self.image_size
        # random square crop after a short-side resize to s
        w, h = img.size
        scale = s / min(w, h)
        img = img.resize((max(s, int(w * scale)), max(s, int(h * scale))),
                         Image.LANCZOS)
        w, h = img.size
        x0 = py_rng.randint(0, w - s)
        y0 = py_rng.randint(0, h - s)
        img = img.crop((x0, y0, x0 + s, y0 + s))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __iter__(self):
        # Per-worker RNG so each worker draws different (path, θ) pairs.
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        seed = self.seed + wid * 1009 + os.getpid()
        torch_rng = torch.Generator(device='cpu').manual_seed(seed)
        py_rng = random.Random(seed)
        while True:
            path = py_rng.choice(self.paths)
            try:
                orig_cpu = self._load(path, py_rng)
            except Exception:
                continue
            theta = sample_theta(torch_rng)         # dict[str, CPU tensor]
            yield orig_cpu, theta


# ============================================================
# Training
# ============================================================

@torch.no_grad()
def _render_batch(renderer: DifferentiableRenderer,
                  orig_batch: torch.Tensor,
                  theta_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Render an entire batch by stamping each sample's θ onto the shared
    renderer in turn (in eval / no-grad mode).

    Returns:
        edited : (B, 3, H, W) on the renderer's device
        theta_v: (B, THETA_DIM) flat θ vector on the renderer's device
    """
    device = orig_batch.device
    B = orig_batch.size(0)
    edits = []
    flats = []
    for i in range(B):
        # write per-sample θ into the renderer
        for name, _ in THETA_LAYOUT:
            getattr(renderer, name).data.copy_(
                theta_batch[name][i].to(device).reshape(getattr(renderer, name).shape)
            )
        edits.append(renderer(orig_batch[i:i + 1]))
        flats.append(torch.cat([theta_batch[name][i].reshape(-1).to(device)
                                for name, _ in THETA_LAYOUT]))
    return torch.cat(edits, dim=0), torch.stack(flats, dim=0)


def _worker_init_fn(worker_id: int):
    """Re-seed each worker's stdlib random so PIL crops decorrelate."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)


def train(image_dir: str,
          out_path: str = 'amortized_init.pt',
          steps: int = 20_000,
          batch_size: int = 16,
          image_size: int = 128,
          lr: float = 3e-4,
          device: str = 'cuda',
          num_workers: int = 4,
          log_every: int = 100):
    # Why batch_size=16 (not the originally proposed 64):
    #   The batched render in `_render_batch` is a per-sample for-loop in the
    #   main process; render time scales linearly with batch_size.  At
    #   batch=16, workers fully hide CPU image loading behind GPU rendering
    #   (~107 samples/s).  At batch≥32, render time exceeds load time, the
    #   prefetch advantage vanishes (~83 samples/s).  True batched rendering
    #   would require giving the renderer a (B,)-shape parameter API — out
    #   of scope for this change.
    """Train the amortised-init CNN.

    Pipeline:
        Workers (num_workers procs, CPU only):
            PIL load → resize → crop → tensor → sample θ ~ priors → return
        Main process (this loop, GPU):
            collate B samples → render edited batch → CNN forward → Huber loss
    """
    ds = SyntheticEditDataset(image_dir, image_size=image_size)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=(device == 'cuda'),
    )

    # Renderer lives in the main process so its CUDA context is fork-safe.
    renderer = DifferentiableRenderer().to(device).eval()

    model = AmortizedInitNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    std = THETA_STD.to(device)

    model.train()
    it = iter(dl)
    losses = []
    for step in range(steps):
        orig_cpu, theta_batch = next(it)
        orig_batch = orig_cpu.to(device, non_blocking=True)
        edit_batch, theta_vec = _render_batch(renderer, orig_batch, theta_batch)

        pred = model(orig_batch, edit_batch)
        # Huber on normalised θ — robust to the long-tail priors
        loss = F.smooth_l1_loss(pred / std, theta_vec / std, beta=0.5)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == steps - 1:
            avg = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            print(f"[{step:5d}/{steps}] huber={loss.item():.4f}  "
                  f"avg_{log_every}={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}")

    torch.save({'state_dict': model.state_dict(),
                'theta_layout': THETA_LAYOUT}, out_path)
    print(f"Saved {out_path}")


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def predict(model: AmortizedInitNet,
            orig_srgb: torch.Tensor, edited_srgb: torch.Tensor) -> dict:
    """Return a dict matching ``DifferentiableRenderer.init_from_lr_params``."""
    if orig_srgb.dim() == 3: orig_srgb   = orig_srgb.unsqueeze(0)
    if edited_srgb.dim() == 3: edited_srgb = edited_srgb.unsqueeze(0)
    device = next(model.parameters()).device
    pred = model(orig_srgb.to(device), edited_srgb.to(device)).squeeze(0).cpu()
    out = {}
    i = 0
    for name, n in THETA_LAYOUT:
        out[name] = pred[i:i + n].tolist()
        i += n
    # Map to LR-style names for compatibility with init_from_lr_params
    lr_keys = {
        'exposure':   '曝光度', 'contrast':  '对比度',
        'highlights': '高光',   'shadows':   '阴影',
        'whites':     '白色色阶','blacks':    '黑色色阶',
        'tint':       '色调',   'vibrance':  '自然饱和度',
        'saturation': '饱和度', 'vignette':  '晕影',
        'dehaze':     '去朦胧', 'sharpening':'锐化',
        'noise_red':  '噪点消除',
    }
    lr_dict = {lr: out[en][0] for en, lr in lr_keys.items()}
    lr_dict['色温'] = out['temp'][0] * 50.0 + TEMP_NEUTRAL
    lr_dict['纹理']    = out['texture'][0]
    lr_dict['清晰度']  = out['clarity'][0]
    # English aliases too (optional, init_from_lr_params accepts both)
    lr_dict['texture'] = out['texture'][0]
    lr_dict['clarity'] = out['clarity'][0]
    # Color mixer
    lr_dict['color_mixer'] = []
    for k, name in enumerate(MIXER_NAMES):
        lr_dict['color_mixer'].append({
            'channel': name,
            'H': out['mixer_hue'][k],
            'S': out['mixer_sat'][k],
            'L': out['mixer_lum'][k],
        })
    # Color grading
    lr_dict['cg_shadow']    = out['cg_shadow']
    lr_dict['cg_midtone']   = out['cg_midtone']
    lr_dict['cg_highlight'] = out['cg_highlight']
    lr_dict['tone_curve_dy'] = out['tone_curve_dy']
    return lr_dict


def load_model(path: str, device: str = 'cuda') -> AmortizedInitNet:
    ckpt = torch.load(path, map_location=device)
    model = AmortizedInitNet().to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# ============================================================
# CLI
# ============================================================

def _cli():
    p = argparse.ArgumentParser(description='Train the amortised init CNN')
    p.add_argument('--image_dir', required=True,
                   help='directory of original PNG/JPG images (used as the '
                        'pool of "originals" for synthetic edit generation)')
    p.add_argument('--out', default='amortized_init.pt')
    p.add_argument('--steps', type=int, default=20_000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    train(a.image_dir, out_path=a.out, steps=a.steps,
          batch_size=a.batch_size, image_size=a.image_size,
          lr=a.lr, device=a.device, num_workers=a.num_workers)


if __name__ == '__main__':
    _cli()
