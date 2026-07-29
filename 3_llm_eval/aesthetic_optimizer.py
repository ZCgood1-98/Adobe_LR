#!/usr/bin/env python3
"""
aesthetic_optimizer.py
======================
Two-phase aesthetic optimization for photos using a Multimodal LLM (MLLM).

Phase 1 — One-shot Analysis:
  MLLM performs a five-dimension aesthetic diagnosis (subject, exposure, color,
  contrast, texture) on the original image, then directly generates one parameter set.
  The result is rendered locally using OpenCV/NumPy.

Phase 2 — Iterative Optimization:
  Dimension-by-dimension optimization in the order:
    Subject → Exposure → Color → Contrast → Texture
  For each dimension, 5 candidate parameter values are generated, rendered locally,
  then the MLLM selects the best one visually from a montage.

Usage:
  python aesthetic_optimizer.py [--sample <uuid>] [--output-dir <path>]
  # 指定单张
    python3 aesthetic_optimizer.py --sample 45691faa-8044-410d-a8d5-b5ed6fdf8b5d

    # 前5张，自定义输出目录
    python3 aesthetic_optimizer.py --count 5 --output-dir /tmp/test_run

    # 不加参数：只跑第1张（默认行为）
    python3 aesthetic_optimizer.py
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import os
from dotenv import load_dotenv
import openai

# ============================================================
# Configuration
# ============================================================
load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "scraped_data_v4" / "test"

# 安全地获取 API Key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("未找到 API Key，请检查 .env 文件是否配置正确。")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "openai/gpt-4o-mini"

RENDER_SIZE = (600, 450)   # (width, height) for all rendered images

# ============================================================
# Parameter definitions (reused from step3_assr_cot.py)
# ============================================================

PARAM_RANGES = {
    '曝光度':     (-5.0,   5.0,  'float'),
    '对比度':     (-100,   100,  'int'),
    '高光':       (-100,   100,  'int'),
    '阴影':       (-100,   100,  'int'),
    '白色色阶':   (-100,   100,  'int'),
    '黑色色阶':   (-100,   100,  'int'),
    '色温':       (2000,   10000, 'int'),
    '色调':       (-150,   150,  'int'),
    '自然饱和度': (-100,   100,  'int'),
    '饱和度':     (-100,   100,  'int'),
    '去朦胧':     (-100,   100,  'int'),
    '清晰度':     (-100,   100,  'int'),
    '纹理':       (-100,   100,  'int'),
    '晕影':       (-100,     0,  'int'),
    '锐化':       (0,      150,  'int'),
    '噪点消除':   (0,      100,  'int'),
}

# Five aesthetic dimensions and their associated parameters
DIMENSIONS = [
    {
        "name": "subject",
        "name_cn": "主体",
        "params": ["晕影"],
        "desc": "Subject prominence and vignette to draw attention",
    },
    {
        "name": "exposure",
        "name_cn": "曝光",
        "params": ["曝光度", "高光", "阴影", "白色色阶", "黑色色阶"],
        "desc": "Overall brightness, highlights, shadows, and tonal range",
    },
    {
        "name": "color",
        "name_cn": "色彩",
        "params": ["色温", "色调", "自然饱和度", "饱和度"],
        "desc": "White balance, color temperature, vibrance and saturation",
    },
    {
        "name": "contrast",
        "name_cn": "对比度",
        "params": ["对比度", "去朦胧"],
        "desc": "Global contrast and local clarity / dehaze",
    },
    {
        "name": "texture",
        "name_cn": "质感",
        "params": ["纹理", "清晰度", "锐化", "噪点消除"],
        "desc": "Texture, clarity, sharpening and noise reduction",
    },
]

# ALL dimension params are split into (row_group, col_group).
# Every param in a group gets 3 candidate values; the 3×3 grid is:
#   rows  → index into each row_group param's 3 values
#   cols  → index into each col_group param's 3 values
# This means each of the 9 cells fully specifies every parameter in the dimension
# independently — no param is left at a potentially hallucinated Phase-1 value.
# The first param in each group is the "primary" axis (shown in cell labels and prompt).
# Dimensions with only one logical axis use col_group = [].
DIMENSION_GRID_PARAMS: dict[str, tuple[list[str], list[str]]] = {
    "subject":  (["晕影"],                       []),
    "exposure": (["曝光度", "阴影", "黑色色阶"], ["高光", "白色色阶"]),
    "color":    (["色温",  "色调"],              ["自然饱和度", "饱和度"]),
    "contrast": (["对比度"],                     ["去朦胧"]),
    "texture":  (["纹理", "清晰度"],             ["锐化", "噪点消除"]),
}

RENDER_NEUTRAL_TEMP = 4500  # reference color temperature for neutral rendering

# ============================================================
# sRGB ↔ linear light conversion
# ============================================================

_SRGB_TO_LINEAR_LUT: np.ndarray = np.array(
    [
        (c / 255.0 / 12.92) if (c / 255.0) <= 0.04045
        else (((c / 255.0) + 0.055) / 1.055) ** 2.4
        for c in range(256)
    ],
    dtype=np.float32,
)


def srgb_to_linear(img_uint8: np.ndarray) -> np.ndarray:
    return _SRGB_TO_LINEAR_LUT[img_uint8]


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    c = np.clip(img_lin, 0.0, 1.0)
    encoded = np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.maximum(c, 1e-9), 1.0 / 2.4) - 0.055,
    )
    return (np.clip(encoded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# Per-parameter render functions (linear light space)
# ============================================================

def apply_exposure(arr: np.ndarray, ev: float) -> np.ndarray:
    return arr * (2.0 ** ev)


def apply_contrast(arr: np.ndarray, contrast: float) -> np.ndarray:
    if contrast == 0:
        return arr
    c = contrast * 1.27
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    return np.clip(factor * (arr - 0.18) + 0.18, 0.0, 1.0)


def apply_highlights(arr: np.ndarray, highlights: float) -> np.ndarray:
    if highlights == 0:
        return arr
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2])
    mask = np.clip((lum - 0.18) / (1.0 - 0.18), 0.0, 1.0)[:, :, np.newaxis]
    return arr * np.maximum(1.0 + mask * (highlights / 100.0) * 1.5, 0.0)


def apply_shadows(arr: np.ndarray, shadows: float) -> np.ndarray:
    if shadows == 0:
        return arr
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2])
    mask = np.clip((0.18 - lum) / 0.18, 0.0, 1.0)[:, :, np.newaxis]
    return arr * np.maximum(1.0 + mask * (shadows / 100.0) * 3.0, 0.0)


def apply_whites(arr: np.ndarray, whites: float) -> np.ndarray:
    if whites == 0:
        return arr
    # Lift/compress the upper tonal range
    factor = 1.0 + whites / 400.0
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2])
    mask = np.clip(lum / 1.0, 0.0, 1.0)[:, :, np.newaxis]
    return arr + mask * (arr * factor - arr)


def apply_blacks(arr: np.ndarray, blacks: float) -> np.ndarray:
    if blacks == 0:
        return arr
    factor = 1.0 + blacks / 400.0
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2])
    mask = np.clip(1.0 - lum / 0.18, 0.0, 1.0)[:, :, np.newaxis]
    return arr + mask * (arr * factor - arr)


def apply_temperature(arr: np.ndarray, temp: float) -> np.ndarray:
    relative = np.clip((temp - RENDER_NEUTRAL_TEMP) / 50.0, -100.0, 100.0)
    result = arr.copy()
    result[:, :, 0] = result[:, :, 0] * (1.0 + relative / 300.0)
    result[:, :, 2] = result[:, :, 2] * (1.0 - relative / 300.0)
    return result


def apply_tint(arr: np.ndarray, tint: float) -> np.ndarray:
    if tint == 0:
        return arr
    result = arr.copy()
    result[:, :, 1] = result[:, :, 1] * (1.0 + tint / 600.0)
    return result


def apply_vibrance(arr: np.ndarray, vibrance: float) -> np.ndarray:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-6)
    boost = (vibrance / 100.0) * (1.0 - sat)[:, :, np.newaxis]
    mean_c = ((r + g + b) / 3.0)[:, :, np.newaxis]
    return arr + boost * (arr - mean_c)


def apply_saturation(arr: np.ndarray, saturation: float) -> np.ndarray:
    if saturation == 0:
        return arr
    mean_c = arr.mean(axis=2, keepdims=True)
    factor = 1.0 + saturation / 100.0
    return mean_c + (arr - mean_c) * factor


def apply_dehaze(arr: np.ndarray, dehaze: float) -> np.ndarray:
    if dehaze <= 0:
        return arr
    mean = arr.mean(axis=(0, 1), keepdims=True)  # per-channel mean, shape (1,1,3)
    return mean + (arr - mean) * (1.0 + dehaze / 150.0)


def apply_clarity(arr: np.ndarray, clarity: float) -> np.ndarray:
    if clarity == 0:
        return arr
    # Midtone contrast via unsharp mask on luminance
    blurred = cv2.GaussianBlur(arr, (0, 0), 15)
    factor = clarity / 100.0
    return arr + factor * (arr - blurred)


def apply_texture(arr: np.ndarray, texture: float) -> np.ndarray:
    if texture == 0:
        return arr
    blurred = cv2.GaussianBlur(arr, (0, 0), 3)
    factor = texture / 100.0
    return arr + factor * (arr - blurred)


def apply_sharpening(arr: np.ndarray, sharpening: float) -> np.ndarray:
    if sharpening <= 0:
        return arr
    blurred = cv2.GaussianBlur(arr, (0, 0), 1.0)
    factor = sharpening / 150.0
    return arr + factor * (arr - blurred)


def apply_noise_reduction(arr: np.ndarray, nr: float) -> np.ndarray:
    if nr <= 0:
        return arr
    uint8 = linear_to_srgb(arr)
    h_param = int(nr * 0.1)
    if h_param < 1:
        h_param = 1
    denoised = cv2.fastNlMeansDenoisingColored(uint8, None, h_param, h_param, 7, 21)
    return srgb_to_linear(denoised)


def apply_vignette(arr: np.ndarray, vignette: float) -> np.ndarray:
    if vignette >= 0:
        return arr
    H, W = arr.shape[:2]
    xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    dist = np.sqrt(xx ** 2 + yy ** 2)
    mask = np.clip(1.0 - abs(vignette) / 100.0 * dist * 0.6, 0.0, 1.0)[:, :, np.newaxis]
    return arr * mask


# Ordered render pipeline matching Lightroom workflow
RENDER_PIPELINE = [
    ('曝光度',     apply_exposure),
    ('高光',       apply_highlights),
    ('阴影',       apply_shadows),
    ('白色色阶',   apply_whites),
    ('黑色色阶',   apply_blacks),
    ('对比度',     apply_contrast),
    ('色温',       apply_temperature),
    ('色调',       apply_tint),
    ('自然饱和度', apply_vibrance),
    ('饱和度',     apply_saturation),
    ('晕影',       apply_vignette),
    ('纹理',       apply_texture),
    ('清晰度',     apply_clarity),
    ('去朦胧',     apply_dehaze),
    ('锐化',       apply_sharpening),
    ('噪点消除',   apply_noise_reduction),
]


def render_params(original_path: Path, params: dict, size: tuple = RENDER_SIZE) -> np.ndarray:
    """
    Apply a full parameter set to the original image.
    Returns a uint8 RGB numpy array (OpenCV-friendly after channel swap).
    """
    img = Image.open(original_path).convert('RGB')
    img.thumbnail(size, Image.LANCZOS)  # scale down keeping aspect ratio, never upscale beyond original
    arr = srgb_to_linear(np.array(img))
    for label, fn in RENDER_PIPELINE:
        val = params.get(label)
        if val is not None and val != 0:
            arr = fn(arr, float(val))
    return linear_to_srgb(arr)  # uint8 RGB


def save_image_cv2(rgb_array: np.ndarray, path: Path) -> None:
    """Save RGB uint8 numpy array via OpenCV (converts to BGR)."""
    cv2.imwrite(str(path), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))


# ============================================================
# Parameter parsing from combined_details.json
# ============================================================

def parse_param_value(value_str) -> float | None:
    if value_str is None:
        return None
    s = str(value_str).strip()
    s = s.replace('\u2212', '-').replace('\u2013', '-')
    s = s.rstrip('%')
    try:
        return float(s)
    except ValueError:
        return None


def load_params_from_combined(combined_json_path: Path) -> dict | None:
    """Load parameters from combined_details.json, returning the first version's params."""
    with open(combined_json_path, encoding='utf-8') as f:
        data = json.load(f)
    versions = data.get('versions', {})
    if not versions:
        return None
    first_version = next(iter(versions.values()))
    params = {}
    for item in first_version.get('details', []):
        if item.get('type') == 'standard':
            label = item.get('label', '')
            val = parse_param_value(item.get('value'))
            if val is not None and label in PARAM_RANGES:
                params[label] = val
    return params if params else None


# ============================================================
# Image → base64 helpers
# ============================================================

def ndarray_to_b64_jpeg(rgb_array: np.ndarray, quality: int = 85) -> str:
    pil = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def path_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ============================================================
# Montage builder for candidate sweep
# ============================================================

def _label_height() -> int:
    return 36


def _add_label(rgb_array: np.ndarray, label_text: str) -> np.ndarray:
    """Add a text label bar at the bottom of an RGB image (numpy array)."""
    H, W = rgb_array.shape[:2]
    lh = _label_height()
    canvas = np.full((H + lh, W, 3), 20, dtype=np.uint8)
    canvas[:H] = rgb_array

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label_text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, H + 6), label_text, fill=(255, 255, 255), font=font)
    return np.array(pil)


# ============================================================
# Candidate value generation + 3×3 grid montage
# ============================================================

def _axis_values(param: str, center: float, n: int) -> list[float]:
    """
    Generate n evenly-spaced values centred on `center`, clamped to the valid range.
    Step size = 20% of parameter span (wider than the old 12% to make grid cells
    visually distinct enough for the MLLM to distinguish).

    When boundary clipping causes fewer than n unique integer values after dedup,
    extend outward into whichever side of the range still has room — never duplicate.
    """
    lo, hi, dtype = PARAM_RANGES.get(param, (-100, 100, 'int'))
    span = hi - lo
    if param == '色温':
        step = 400
    elif dtype == 'float':
        step = span * 0.10
    else:
        step = max(3, int(span * 0.10))

    half = n // 2
    raw = [center + (i - half) * step for i in range(n)]

    # Normalise: int params → rounded ints stored as float; float params → round to 4 dp
    # so that boundary-clipped duplicates (e.g. 5.0, 5.0, 5.0) are detectable via set lookup.
    def normalise(v: float) -> float:
        if dtype == 'int':
            return float(int(round(max(lo, min(hi, v)))))
        return round(max(lo, min(hi, v)), 4)

    # Build deduplicated list, preserving original order
    seen: set[float] = set()
    result: list[float] = []
    for v in raw:
        nv = normalise(v)
        if nv not in seen:
            seen.add(nv)
            result.append(nv)

    # If boundary clipping caused duplicate normalised values and left us short,
    # extend outward into whichever side of the valid range still has room.
    while len(result) < n:
        low_room  = min(result) - lo
        high_room = hi - max(result)

        next_low  = normalise(min(result) - step)
        next_high = normalise(max(result) + step)

        # Prefer the direction with more remaining range
        if low_room >= high_room:
            candidates = [(next_low, 'low'), (next_high, 'high')]
        else:
            candidates = [(next_high, 'high'), (next_low, 'low')]

        added = False
        for val, side in candidates:
            if lo <= val <= hi and val not in seen:
                seen.add(val)
                if side == 'low':
                    result.insert(0, val)
                else:
                    result.append(val)
                added = True
                break

        if not added:
            break  # valid range genuinely exhausted; stop rather than duplicate

    return result[:n]


def _fmt_val(param: str, val: float) -> str:
    """Format a parameter value for display in grid cell labels."""
    _, _, dtype = PARAM_RANGES.get(param, (-100, 100, 'int'))
    if dtype == 'float':
        return f"{val:+.2f}"
    iv = int(round(val))
    return f"{iv:+d}" if iv != 0 else "0"


def _neutral(param: str) -> float:
    """Return the perceptual neutral value for a parameter (0 for most, 4500 for 色温)."""
    return 4500.0 if param == '色温' else 0.0


def generate_grid_candidates(
    dim_name: str,
    initial_params: dict,
) -> tuple[list[dict], dict[str, list[float]], dict[str, list[float]]]:
    """
    Build the 9 candidate parameter dicts for a 3×3 grid, covering ALL dimension params.

    DIMENSION_GRID_PARAMS splits every dim param into (row_group, col_group).
    Each param in a group gets 3 candidate values independently swept around its
    Phase-1 center.  The 9 cells are the cross-product:
        cell(r, c) = {p: row_vals[p][r] for p in row_group}
                   | {p: col_vals[p][c] for p in col_group}

    This guarantees:
      - Every dim param varies across the 9 cells — no param is stuck at a fixed
        (possibly hallucinated) Phase-1 value.
      - Row-group params covary (same row index r across all cells in that row),
        naturally coupling semantically related adjustments.

    Returns
    -------
    cells       : list[dict]  — 9 full param-override dicts (all dim params).
    row_val_map : {param: [v0, v1, v2]}  for each param in row_group.
    col_val_map : {param: [v0, v1, v2]}  for each param in col_group (empty for 1-axis dims).

    Grid layout (reading order):
        [1] [2] [3]   row 0
        [4] [5] [6]   row 1
        [7] [8] [9]   row 2
    Cell N → row = (N-1)//3, col = (N-1)%3
    """
    row_group, col_group = DIMENSION_GRID_PARAMS[dim_name]

    # Single-axis dimension (e.g. subject: only 晕影, no col group)
    if not col_group:
        param = row_group[0]
        vals = _axis_values(param, initial_params.get(param, _neutral(param)), n=9)
        cells = [{param: v} for v in vals]
        return cells, {param: vals}, {}

    # Two-axis dimension: generate 3 values for every param in each group
    row_val_map: dict[str, list[float]] = {
        p: _axis_values(p, initial_params.get(p, _neutral(p)), n=3)
        for p in row_group
    }
    col_val_map: dict[str, list[float]] = {
        p: _axis_values(p, initial_params.get(p, _neutral(p)), n=3)
        for p in col_group
    }

    cells: list[dict] = []
    for r in range(3):
        for c in range(3):
            cell: dict = {}
            for p in row_group:
                cell[p] = row_val_map[p][r]
            for p in col_group:
                cell[p] = col_val_map[p][c]
            cells.append(cell)

    return cells, row_val_map, col_val_map


def build_3x3_montage(
    original_path: Path,
    base_params: dict,
    dim_name: str,
    cells: list[dict],
) -> np.ndarray:
    """
    Render 9 images (one per grid cell) and stitch into a 3×3 grid.

    Each cell fully overrides all dim params on top of base_params (which carries
    params from already-optimised dimensions only).
    Label shows [N] plus the primary axes' values (first param of each group).
    Returns an RGB uint8 numpy array.
    """
    assert len(cells) == 9
    row_group, col_group = DIMENSION_GRID_PARAMS[dim_name]
    primary_row = row_group[0]
    primary_col = col_group[0] if col_group else None

    rendered = []
    for i, overrides in enumerate(cells):
        trial_params = {**base_params, **overrides}
        img = render_params(original_path, trial_params)
        # Label: cell index + primary axis values only (secondary params omitted for readability)
        label = f"[{i + 1}]  {primary_row}={_fmt_val(primary_row, overrides[primary_row])}"
        if primary_col is not None:
            label += f"  {primary_col}={_fmt_val(primary_col, overrides[primary_col])}"
        rendered.append(_add_label(img, label))

    H, W = rendered[0].shape[:2]
    grid = np.full((H * 3, W * 3, 3), 20, dtype=np.uint8)
    for i, img in enumerate(rendered):
        r, c = i // 3, i % 3
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
    return grid


# ============================================================
# MLLM calls
# ============================================================

def _call_mllm(messages: list[dict], max_tokens: int = 1200, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


SYSTEM_PROMPT = (
    "You are an expert photo retouching artist specializing in Lightroom color science. "
    "Analyze photos carefully and provide precise, technically grounded parameter recommendations."
)

# ---------- Phase 1: One-shot aesthetic analysis ----------

ONESHOT_USER_TEMPLATE = """\
[Original Photo]
Analyze this photo across five aesthetic dimensions and generate a complete set of Lightroom adjustment parameters.

Five dimensions to diagnose:
1. Subject (主体): subject prominence, vignette
2. Exposure (曝光): overall brightness, highlights, shadows, white/black point
3. Color (色彩): white balance (色温/色调), vibrance (自然饱和度), saturation (饱和度)
4. Contrast (对比度): global contrast, dehaze (去朦胧)
5. Texture (质感): texture (纹理), clarity (清晰度), sharpening (锐化), noise reduction (噪点消除)

Available parameters with valid ranges:
- 曝光度: -5.0 to +5.0 (float)
- 对比度: -100 to +100 (int)
- 高光: -100 to +100 (int)
- 阴影: -100 to +100 (int)
- 白色色阶: -100 to +100 (int)
- 黑色色阶: -100 to +100 (int)
- 色温: 2000 to 10000 (int, absolute value; 4500 is neutral)
- 色调: -150 to +150 (int)
- 自然饱和度: -100 to +100 (int)
- 饱和度: -100 to +100 (int)
- 去朦胧: -100 to +100 (int)
- 清晰度: -100 to +100 (int)
- 纹理: -100 to +100 (int)
- 晕影: -100 to 0 (int)
- 锐化: 0 to 150 (int)
- 噪点消除: 0 to 100 (int)

Output ONLY a JSON object with parameter names as keys (Chinese) and numeric values.
For 色温 output an absolute value (e.g. 5500), for all others output deltas from neutral (0).
Example format:
{
  "曝光度": 0.5,
  "对比度": 15,
  "高光": -30,
  "阴影": 40,
  "白色色阶": 10,
  "黑色色阶": -5,
  "色温": 5200,
  "色调": 5,
  "自然饱和度": 20,
  "饱和度": 0,
  "去朦胧": 10,
  "清晰度": 15,
  "纹理": 20,
  "晕影": -15,
  "锐化": 30,
  "噪点消除": 10
}
"""


def phase1_oneshot(original_path: Path) -> dict:
    """Call MLLM to generate a complete parameter set from a single look at the image."""
    print("\n[Phase 1] Sending original image for one-shot aesthetic analysis...")
    orig_b64 = path_to_b64(original_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}", "detail": "high"}},
                {"type": "text", "text": ONESHOT_USER_TEMPLATE},
            ],
        },
    ]
    response = _call_mllm(messages, max_tokens=600)
    print(f"  MLLM response:\n{response}\n")

    # Extract JSON from response
    json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
    if not json_match:
        print("  [Warning] Could not parse JSON from response, using defaults.")
        return {}
    try:
        params = json.loads(json_match.group())
        # Validate and clamp values
        validated = {}
        for k, v in params.items():
            if k in PARAM_RANGES:
                lo, hi, dtype = PARAM_RANGES[k]
                clamped = max(lo, min(hi, float(v)))
                validated[k] = int(round(clamped)) if dtype == 'int' else round(clamped, 2)
        return validated
    except json.JSONDecodeError as e:
        print(f"  [Warning] JSON parse error: {e}")
        return {}


# ---------- Phase 2: Iterative dimension-by-dimension optimization ----------

GRID_SELECT_TEMPLATE = """\
[Original Photo]
[3×3 Candidate Grid — {dim_name_cn} ({dim_name}) dimension]

The grid shows 9 candidate renderings of the same photo, labeled [1] through [9] \
in reading order (left-to-right, top-to-bottom):
  [1] [2] [3]
  [4] [5] [6]
  [7] [8] [9]

{grid_axes_desc}

Previously fixed parameters (other dimensions already optimised):
{context_params}

Dimension being optimised: {dim_desc}

Task: Compare all 9 cells against the [Current Base Photo] (the photo with all
previously confirmed edits applied, but WITHOUT the parameters currently being swept).
Select the cell number [1-9] that produces the best aesthetic result for the \
"{dim_name_cn}" dimension.

Respond with ONLY:
Best candidate: [N]
Reason: <one sentence>
"""


def phase2_iterative(
    original_path: Path,
    output_dir: Path,
    initial_params: dict,
) -> dict:
    """
    Optimise each aesthetic dimension with a single MLLM call (5 calls total).

    For each dimension, DIMENSION_GRID_PARAMS splits ALL dim params into
    (row_group, col_group).  Every param gets 3 candidate values swept around its
    Phase-1 centre, so no param is ever stuck at a fixed (hallucinated) value.

    After MLLM picks a cell, we commit the chosen row-index values for row_group
    AND the chosen col-index values for col_group — all dim params are resolved.
    """
    # Only keep params that belong to already-completed dimensions.
    # Phase-1 values for the current dimension's params are intentionally excluded
    # from current_params until MLLM resolves them — they must not pollute the base image.
    current_params: dict = {}

    for dim in DIMENSIONS:
        dim_name    = dim["name"]
        dim_name_cn = dim["name_cn"]
        row_group, col_group = DIMENSION_GRID_PARAMS[dim_name]
        all_dim_params = row_group + col_group

        print(f"\n[Phase 2] Dimension: {dim_name_cn} ({dim_name})")
        print(f"  Row group: {row_group}")
        print(f"  Col group: {col_group}")

        # Generate 9 cells, each fully specifying all dim params
        cells, row_val_map, col_val_map = generate_grid_candidates(dim_name, initial_params)

        # Debug: print the value ranges
        for p, vals in row_val_map.items():
            print(f"  Row param {p}: {[_fmt_val(p, v) for v in vals]}")
        for p, vals in col_val_map.items():
            print(f"  Col param {p}: {[_fmt_val(p, v) for v in vals]}")

        # Build and save 3×3 montage.
        # base_params = current_params (previous dimensions only, no current-dim params).
        montage = build_3x3_montage(original_path, current_params, dim_name, cells)
        montage_path = output_dir / f"montage_{dim_name}.jpg"
        save_image_cv2(montage, montage_path)
        print(f"  Montage saved: {montage_path}")

        # Describe grid axes for the prompt (show primary param of each group)
        primary_row = row_group[0]
        if col_group:
            primary_col = col_group[0]
            grid_axes_desc = (
                f"Grid structure:\n"
                f"  Rows (top→bottom) primary axis '{primary_row}': "
                f"{', '.join(_fmt_val(primary_row, v) for v in row_val_map[primary_row])}\n"
                f"  Cols (left→right) primary axis '{primary_col}': "
                f"{', '.join(_fmt_val(primary_col, v) for v in col_val_map[primary_col])}\n"
                f"  (Each row/col also jointly varies: "
                f"row→{row_group[1:] or 'none'}, col→{col_group[1:] or 'none'})"
            )
        else:
            grid_axes_desc = (
                f"Grid structure (reading order [1]→[9]):\n"
                f"  '{primary_row}' = "
                f"{', '.join(_fmt_val(primary_row, v) for v in row_val_map[primary_row])}"
            )

        # Context: params already fixed by previous dimensions
        context_lines = [f"  {k}: {v}" for k, v in current_params.items() if v != 0.0]
        context_str = "\n".join(context_lines) if context_lines else "  (none yet)"

        prompt_text = GRID_SELECT_TEMPLATE.format(
            dim_name=dim_name,
            dim_name_cn=dim_name_cn,
            dim_desc=dim["desc"],
            grid_axes_desc=grid_axes_desc,
            context_params=context_str,
        )

        # Base image = current state WITHOUT this dimension's params.
        # This is the correct reference: MLLM sees exactly what changes between
        # the base and each grid cell, isolating this dimension's contribution.
        base_b64    = ndarray_to_b64_jpeg(render_params(original_path, current_params), quality=85)
        montage_b64 = ndarray_to_b64_jpeg(montage, quality=85)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base_b64}", "detail": "low"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{montage_b64}", "detail": "high"}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        response = _call_mllm(messages, max_tokens=200)
        print(f"  MLLM response: {response}")

        # Parse cell selection [1]–[9]
        match = re.search(r'Best candidate:\s*\[?(\d)\]?', response, re.IGNORECASE)
        if match:
            cell_idx = int(match.group(1)) - 1
            cell_idx = max(0, min(8, cell_idx))
        else:
            print("  [Warning] Could not parse selection, defaulting to centre cell [5].")
            cell_idx = 4

        # Decode → commit ALL dim params (row group + col group)
        if col_group:
            row, col = cell_idx // 3, cell_idx % 3
            for p in row_group:
                current_params[p] = row_val_map[p][row]
            for p in col_group:
                current_params[p] = col_val_map[p][col]
            print(f"  Selected cell [{cell_idx + 1}] (row={row}, col={col})")
            for p in all_dim_params:
                print(f"    {p} = {current_params[p]}")
        else:
            primary = row_group[0]
            current_params[primary] = row_val_map[primary][cell_idx]
            print(f"  Selected cell [{cell_idx + 1}]: {primary} = {current_params[primary]}")

    return current_params


# ============================================================
# Main pipeline
# ============================================================

def run(sample_id: str | None, output_dir: Path) -> None:
    # Select sample
    test_dirs = sorted(DATA_DIR.iterdir())
    if not test_dirs:
        print(f"No samples found in {DATA_DIR}")
        sys.exit(1)

    if sample_id:
        sample_dir = DATA_DIR / sample_id
        if not sample_dir.exists():
            print(f"Sample {sample_id} not found.")
            sys.exit(1)
    else:
        sample_dir = test_dirs[0]
        sample_id = sample_dir.name

    original_path = sample_dir / "original_captured_cropped.png"
    if not original_path.exists():
        print(f"Original image not found: {original_path}")
        sys.exit(1)

    print(f"Sample: {sample_id}")
    print(f"Original image: {original_path}")

    # Prepare output directory
    sample_output = output_dir / sample_id
    sample_output.mkdir(parents=True, exist_ok=True)

    # Save original for reference
    orig_ref = render_params(original_path, {})  # no adjustments
    save_image_cv2(orig_ref, sample_output / "00_original.jpg")
    print(f"Original saved: {sample_output / '00_original.jpg'}")

    # ---------- Phase 1: One-shot ----------
    phase1_params = phase1_oneshot(original_path)
    print(f"\n[Phase 1] Generated parameters:\n{json.dumps(phase1_params, ensure_ascii=False, indent=2)}")

    phase1_rendered = render_params(original_path, phase1_params)
    phase1_out = sample_output / "01_phase1_oneshot.jpg"
    save_image_cv2(phase1_rendered, phase1_out)
    print(f"Phase 1 result saved: {phase1_out}")

    (sample_output / "01_phase1_params.json").write_text(
        json.dumps(phase1_params, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # ---------- Phase 2: Iterative ----------
    print("\n" + "=" * 60)
    print("Starting Phase 2: Iterative dimension-by-dimension optimization")
    print("=" * 60)

    phase2_params = phase2_iterative(original_path, sample_output, phase1_params)
    print(f"\n[Phase 2] Final parameters:\n{json.dumps(phase2_params, ensure_ascii=False, indent=2)}")

    phase2_rendered = render_params(original_path, phase2_params)
    phase2_out = sample_output / "02_phase2_iterative.jpg"
    save_image_cv2(phase2_rendered, phase2_out)
    print(f"Phase 2 result saved: {phase2_out}")

    (sample_output / "02_phase2_params.json").write_text(
        json.dumps(phase2_params, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # ---------- Side-by-side comparison ----------
    orig_rgb = orig_ref
    p1_rgb = phase1_rendered
    p2_rgb = phase2_rendered

    H, W = orig_rgb.shape[:2]
    gap = 10
    comparison = np.full((H, W * 3 + gap * 2, 3), 40, dtype=np.uint8)
    comparison[:, :W] = orig_rgb
    comparison[:, W + gap:W * 2 + gap] = p1_rgb
    comparison[:, W * 2 + gap * 2:] = p2_rgb

    # Add labels
    pil_comp = Image.fromarray(comparison)
    draw = ImageDraw.Draw(pil_comp)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for x_pos, text in [(10, "Original"), (W + gap + 10, "Phase 1: One-shot"), (W * 2 + gap * 2 + 10, "Phase 2: Iterative")]:
        draw.text((x_pos, 10), text, fill=(255, 255, 255), font=font)

    comparison_path = sample_output / "03_comparison.jpg"
    save_image_cv2(np.array(pil_comp), comparison_path)
    print(f"\nSide-by-side comparison saved: {comparison_path}")

    print("\n" + "=" * 60)
    print("Done! Results in:", sample_output)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Aesthetic photo optimizer using MLLM")
    parser.add_argument("--sample", type=str, default=None,
                        help="Single sample UUID to process")
    parser.add_argument("--count", type=int, default=None,
                        help="Process the first N samples in the test set")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "aesthetic_output"),
                        help="Output directory (default: adobe_lr/aesthetic_output)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.sample:
        run(args.sample, output_dir)
    elif args.count:
        test_dirs = sorted(DATA_DIR.iterdir())[: args.count]
        for i, d in enumerate(test_dirs):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(test_dirs)}] Sample: {d.name}")
            print(f"{'='*60}")
            run(d.name, output_dir)
    else:
        run(None, output_dir)  # default: first sample


if __name__ == "__main__":
    main()
