#!/usr/bin/env python3
"""
LLM Aesthetic Alignment Evaluator
===================================
Evaluates multiple vision LLMs on their ability to perceive Lightroom photo
editing quality across 4 quantifiable dimensions, using scraped_data_v4.

Usage:
    python evaluate_llms.py --samples 50 --output eval_results/
    python evaluate_llms.py --samples 100 --models gpt-4o claude --dim 1 2
    python evaluate_llms.py --analyze-only --output eval_results/
"""

import os
import json
import base64
import random
import argparse
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import requests
import openai
from datetime import datetime
from dotenv import load_dotenv

# ============================================================
# SECTION 1: CONFIGURATION  — Edit here before running
# ============================================================
load_dotenv()

# 全局常量：main() 和 call_llm_vision() 均引用这两个名字
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    raise ValueError("未找到 API Key，请检查 .env 文件是否配置正确。")

client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# Vision-capable LLMs on OpenRouter to benchmark
# Full list at https://openrouter.ai/models (filter: multimodal)
MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gemini-nano-banana": "google/gemini-3.1-flash-image-preview",
    "qwen-3-vl": "qwen/qwen3-vl-32b-instruct",
    "llama-4": "meta-llama/llama-4-maverick",
    "grok-4": "x-ai/grok-4.20",
    "nova-2": "amazon/nova-2-lite-v1",
    # "claude-sonnet": "anthropic/claude-sonnet-4.6",
    # 'gpt-5-image': "openai/gpt-5-image",
}

DATASET_ROOT = Path("/home/a5000/Work/Chen/adobe_lr/scraped_data_v4")
SPLIT = "test"          # "train" or "test"
NUM_SAMPLES = 1000        # Reduce for cost savings; 50 is a good start
RANDOM_SEED = 42

# Dimension 2: parameters with |delta| >= this threshold trigger a question
# (used only as a fallback when image loading fails; the primary gate is PERCEPTUAL_JND)
DIRECTION_THRESHOLD = 30

# Perceptual JND for Dimension 2 pixel-level gating (CIE76 ΔE).
# 2.3 is the consensus JND under standard reference viewing conditions.
# A parameter edit whose masked-region mean ΔE is below this value produces
# no reliably perceptible visual change and should not be presented to the LLM.
PERCEPTUAL_JND = 2.3

# Dimension 4: parameters with |value| >= this threshold are "extreme"
OUTLIER_THRESHOLD = 60

# Composite score weights (must sum to 1.0)
# D1 is the primary aesthetic measure and carries the most weight.
# D2 is a perceptual prerequisite (can you see the change?), not aesthetic judgment.
# D3 diagnoses what changed; D4 detects quality-destroying over-processing.
DIM_WEIGHTS = {
    "dim1": 0.50,   # Aesthetic Delta  (primary: preference alignment)
    "dim2": 0.15,   # Parameter Direction (perceptual prerequisite)
    "dim3": 0.25,   # Diagnosis Accuracy
    "dim4": 0.10,   # Outlier Detection
}

# API settings
REQUEST_DELAY = 1.5     # Seconds between API calls (be polite to rate limits)
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60

# ============================================================
# SECTION 2: PARAMETER KNOWLEDGE BASE
# ============================================================

# Maps Chinese Lightroom parameter labels to visual effect descriptions.
# Used for Dimensions 2, 3, and 4.
PARAM_KNOWLEDGE = {
    "曝光度": {
        "en": "exposure",
        "category": "brightness",
        "is_delta": True,
        "positive_desc": "画面整体变亮",
        "negative_desc": "画面整体变暗",
        "question": "画面的【整体亮度/曝光】",
    },
    "对比度": {
        "en": "contrast",
        "category": "contrast",
        "is_delta": True,
        "positive_desc": "明暗对比增强，画面更有层次感",
        "negative_desc": "明暗对比减弱，画面更柔和平坦",
        "question": "画面的【明暗对比度】",
    },
    "高光": {
        "en": "highlights",
        "category": "brightness",
        "is_delta": True,
        "positive_desc": "亮部区域更亮",
        "negative_desc": "亮部区域受到压制，高光细节更丰富",
        "question": "画面的【高光/亮部区域】",
    },
    "阴影": {
        "en": "shadows",
        "category": "brightness",
        "is_delta": True,
        "positive_desc": "暗部区域变亮，阴影细节增加",
        "negative_desc": "暗部区域更暗",
        "question": "画面的【阴影/暗部细节】",
    },
    "白色色阶": {
        "en": "whites",
        "category": "brightness",
        "is_delta": True,
        "positive_desc": "最亮区域更亮，白色更白",
        "negative_desc": "最亮区域受到压制",
        "question": "画面的【白色色阶/最亮区域】",
    },
    "黑色色阶": {
        "en": "blacks",
        "category": "brightness",
        "is_delta": True,
        "positive_desc": "最暗区域提亮，黑色不再死黑",
        "negative_desc": "最暗区域更黑，对比更强烈",
        "question": "画面的【黑色色阶/最暗区域】",
    },
    "自然饱和度": {
        "en": "vibrance",
        "category": "color",
        "is_delta": True,
        "positive_desc": "颜色更鲜艳生动（低饱和色彩优先增强）",
        "negative_desc": "颜色更低沉克制",
        "question": "画面的【色彩鲜艳度/自然饱和度】",
    },
    "饱和度": {
        "en": "saturation",
        "category": "color",
        "is_delta": True,
        "positive_desc": "所有颜色均匀增强，更鲜艳",
        "negative_desc": "颜色减弱，趋向去饱和",
        "question": "画面的【整体色彩饱和度】",
    },
    "纹理": {
        "en": "texture",
        "category": "detail",
        "is_delta": True,
        "positive_desc": "中频细节/纹理增强，表面质感更强",
        "negative_desc": "中频细节减弱，画面更平滑",
        "question": "画面的【纹理/表面质感细节】",
    },
    "清晰度": {
        "en": "clarity",
        "category": "detail",
        "is_delta": True,
        "positive_desc": "中间调对比增强，画面更清晰有冲击力",
        "negative_desc": "中间调对比减弱，画面更柔和",
        "question": "画面的【清晰度/中间调对比】",
    },
    "去朦胧": {
        "en": "dehaze",
        "category": "clarity",
        "is_delta": True,
        "positive_desc": "雾霾减少，画面通透度大幅提升",
        "negative_desc": "画面增加朦胧感/雾感",
        "question": "画面的【通透度/去朦胧效果】",
    },
    "晕影": {
        "en": "vignette",
        "category": "style",
        "is_delta": True,
        "positive_desc": "边缘变亮（亮晕影）",
        "negative_desc": "边缘变暗，四角产生暗晕影效果",
        "question": "画面的【边角/晕影效果】",
    },
    "锐化": {
        "en": "sharpening",
        "category": "detail",
        "is_delta": False,  # absolute value 0-150; higher = more sharpening
        "positive_desc": "画面锐度增加，细节边缘更清晰",
        "negative_desc": None,
        "question": "画面的【锐化程度/边缘清晰度】",
    },
    "噪点消除": {
        "en": "noise_reduction",
        "category": "detail",
        "is_delta": False,  # absolute value 0-100; higher = less noise
        "positive_desc": "噪点减少，画面更细腻平滑",
        "negative_desc": None,
        "question": "画面的【噪点/颗粒感】",
    },
    "色温": {
        "en": "color_temperature",
        "category": "color",
        "is_delta": False,        # stored as absolute Kelvin (e.g. 5500), not a delta
        "is_absolute_temp": True, # delta = value − NEUTRAL_KELVIN; positive=warmer, negative=cooler
        "positive_desc": "画面色调偏暖（橙黄色调增强）",
        "negative_desc": "画面色调偏冷（蓝青色调增强）",
        "question": "画面的【冷暖色调】",
        # Weber-Fechner relative threshold: JND ∝ K (linear Weber fraction).
        # 5.5 % of the actual Kelvin value ≈ 300 K at the 5500 K daylight neutral,
        # but scales down to ~154 K at 2800 K (tungsten) and up to ~467 K at 8500 K
        # (overcast sky) — matching the logarithmic adaptation of human colour vision.
        "threshold_jnd_fraction": 0.055,
    },
    "色调": {
        "en": "tint",
        "category": "color",
        "is_delta": True,
        "positive_desc": "色调偏洋红/粉红",
        "negative_desc": "色调偏绿",
        "question": "画面的【色调（洋红-绿轴）】",
        "threshold_override": 10,   # tint ±10 already produces a visible color cast
    },
    "颗粒": {
        "en": "grain",
        "category": "style",
        "is_delta": False,  # absolute value, higher = more grain
        "positive_desc": "画面添加了胶片颗粒感",
        "negative_desc": None,
        "question": "画面是否添加了【胶片颗粒】效果",
    },
    "色彩混合器": {
        "en": "color_mixer",
        "category": "color",
        # Synthetic parameter: max(|S|, |L|) across all color_mixer channels.
        # Always positive — direction is intentionally dropped because a composite
        # of 8 channels can't be reduced to a single meaningful direction.
        # Used only for Dim 3 (category ground truth) and Dim 4 (outlier detection),
        # NOT in Dim 2 (no MCQ direction question generated for this parameter).
        "is_delta": False,
        "positive_desc": "特定颜色通道的饱和度或明度发生了明显的选择性调整",
        "negative_desc": None,
        "question": "画面中某些颜色通道是否被选择性地加强或压制",
    },
}

# Parameters that produce perceptible style changes and are tested in Dimension 2.
# Excludes technical/micro-detail parameters (sharpening, noise reduction, etc.)
# whose effects are hard to judge from a compressed image pair shown to an LLM.
STYLE_SENSITIVE_PARAMS = {
    "色温", "色调",                          # Color tone (highest perceptual impact)
    "高光", "阴影", "黑色色阶", "白色色阶",   # Tonal range / luminosity distribution
    "自然饱和度", "饱和度",                   # Color intensity
    "去朦胧",                                # Clarity / atmospheric haze
}

# Neutral Kelvin reference for 色温 delta computation.
# When the original's color temperature is unknown, deviation from 5500K (daylight)
# is used as a proxy for "how far from neutral the edit moved the image".
NEUTRAL_KELVIN = 5500

# Category groupings for Dimension 3 (Diagnosis)
CATEGORY_LABELS = {
    "brightness": "曝光/亮度调整（曝光度/高光/阴影/黑白色阶）",
    "contrast": "对比度",
    "color": "色彩调整（饱和度/自然饱和度/色调）",
    "detail": "细节处理（锐化/清晰度/纹理/降噪）",
    "clarity": "通透度/去朦胧",
    "style": "风格化（晕影/颗粒）",
}

# ============================================================
# SECTION 3: DATA LOADING
# ============================================================

def load_image_b64(
    img_path: Path,
    max_bytes: int = 4 * 1024 * 1024,
    quality: int = 90,
    max_edge: int = 1536,
) -> "Optional[tuple[str, str]]":
    """Load image and return (base64_string, mime_type).

    Always re-encodes via PIL as JPEG so the declared MIME type matches the
    actual bytes (avoids 'detected JPEG does not match declared PNG' errors
    from strict model providers such as Amazon Nova).

    Quality guidance:
      quality=90, max_edge=1536  — default; preserves subtle tonal/color differences
      quality=85, max_edge=1024  — safe for most models at ~1MB budget
      quality=75, max_edge=768   — last resort; noticeable JPEG artefacts possible;
                                   D2/D4 results at this setting are less reliable
    """
    if not img_path.exists():
        return None
    try:
        from PIL import Image
        import io
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = min(max_edge / w, max_edge / h, 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            # If still over budget, lower quality one more step before giving up
            if len(encoded) * 3 // 4 > max_bytes and quality > 70:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return encoded, "image/jpeg"
    except ImportError:
        # Pillow not installed — send raw bytes with the file's actual extension
        with open(img_path, "rb") as f:
            raw = f.read()
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        return base64.b64encode(raw).decode("utf-8"), mime


def _compute_image_stats(img_path: Path) -> dict:
    """Compute luminance-distribution metrics for content-aware threshold scaling.

    Weber-Fechner tonal adaptation: the perceptual impact of a Shadows/Highlights
    adjustment depends on how many pixels actually fall in that tonal zone.  A +30
    shadow lift on a high-key image (shadow_ratio ≈ 0.02) is nearly invisible;
    the same +30 on a backlit scene (shadow_ratio ≈ 0.35) produces a dramatic lift.

    Returns:
        shadow_ratio    — fraction of pixels with luminance < 51  (bottom 20 %)
        highlight_ratio — fraction of pixels with luminance > 204 (top 20 %)
        mean_lum        — mean luminance (0-255)
    Fallback values (0.25 / 0.25 / 128) are used when PIL or numpy is absent.
    """
    try:
        import numpy as np
        from PIL import Image
        with Image.open(img_path) as img:
            lum = np.asarray(img.convert("L"), dtype=np.float32)
            n = lum.size
            return {
                "shadow_ratio": float((lum < 51).sum() / n),
                "highlight_ratio": float((lum > 204).sum() / n),
                "mean_lum": float(lum.mean()),
            }
    except Exception:
        return {"shadow_ratio": 0.25, "highlight_ratio": 0.25, "mean_lum": 128.0}


# Minimum pixel count inside a luminance zone before we trust the masked ΔE.
# If the zone is sparser than this (e.g. no shadows in a high-key shot), we
# fall back to full-image ΔE rather than reporting an unreliable masked value.
_MIN_MASK_PIXELS = 300

# Per-parameter luminance mask ranges in CIELAB L* space (0–100).
# Only pixels whose L* in the *original* image falls within [L_min, L_max]
# are included when computing the masked ΔE for that parameter.
# Parameters absent from this dict use the full-image ΔE (no mask).
PARAM_LMASK: dict[str, tuple[float, float]] = {
    "阴影":    (0.0,  35.0),   # dark zone  — bottom third of tonal range
    "黑色色阶": (0.0,  18.0),   # deep shadow — crushed blacks territory
    "高光":    (65.0, 100.0),  # bright zone — top third
    "白色色阶": (82.0, 100.0),  # specular / clipping zone
}


def _srgb_to_lab(rgb_u8) -> "np.ndarray":
    """Convert an sRGB uint8 array (H×W×3) to CIELAB float32 (H×W×3).

    Implements the full IEC 61966-2-1 pipeline:
      sRGB u8  →  linear RGB  →  XYZ (D65)  →  L*a*b*
    No external colour-science library required beyond NumPy.
    """
    import numpy as np

    rgb = rgb_u8.astype(np.float32) / 255.0

    # sRGB gamma decode (IEC 61966-2-1)
    linear = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    # Linear RGB → CIE XYZ (D65 illuminant, sRGB primaries)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ M.T  # (H, W, 3)

    # Normalise by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    # CIE f function  (handles both the cube-root and linear segments)
    eps = 0.008856
    kappa = 903.3
    f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def _load_lab_pair(
    original_path: Path, edited_path: Path
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Load original and edited images and return (orig_lab, edit_lab) in CIELAB.

    Both arrays have shape (H, W, 3) with dtype float32.
    Returns None if PIL or NumPy is unavailable or if any I/O error occurs.
    """
    try:
        import numpy as np
        from PIL import Image

        orig_rgb = np.asarray(Image.open(original_path).convert("RGB"))
        edit_rgb = np.asarray(Image.open(edited_path).convert("RGB"))

        # Crop to common dimensions if the images were shot at slightly different sizes
        if orig_rgb.shape != edit_rgb.shape:
            h = min(orig_rgb.shape[0], edit_rgb.shape[0])
            w = min(orig_rgb.shape[1], edit_rgb.shape[1])
            orig_rgb = orig_rgb[:h, :w]
            edit_rgb = edit_rgb[:h, :w]

        return _srgb_to_lab(orig_rgb), _srgb_to_lab(edit_rgb)
    except Exception:
        return None


def _masked_delta_e(
    orig_lab,
    edit_lab,
    l_range: "tuple[float, float] | None" = None,
) -> float:
    """Compute mean CIE76 ΔE between orig_lab and edit_lab.

    If l_range=(L_min, L_max) is given, restrict the computation to pixels
    whose L* in the *original* image falls in that range (luminance masking).
    Falls back to the full-image mean ΔE when the masked zone contains fewer
    than _MIN_MASK_PIXELS pixels (e.g. no real shadows in a high-key shot).

    A return value below PERCEPTUAL_JND (2.3) means the edit is visually
    imperceptible and should not generate a Dimension 2 question.
    """
    import numpy as np

    # Per-pixel ΔE76 = Euclidean distance in L*a*b* space
    delta_e = np.sqrt(np.sum((orig_lab - edit_lab) ** 2, axis=2))

    if l_range is not None:
        L = orig_lab[:, :, 0]
        mask = (L >= l_range[0]) & (L <= l_range[1])
        if mask.sum() >= _MIN_MASK_PIXELS:
            return float(delta_e[mask].mean())
        # Zone too sparse (< _MIN_MASK_PIXELS px) — fall back to full-image ΔE

    return float(delta_e.mean())


def parse_param_value(value_str: str) -> Optional[float]:
    """Parse a Lightroom parameter value string to float.
    Handles Unicode minus sign (−) and signed strings like '+52', '−65'.
    """
    if value_str is None:
        return None
    value_str = str(value_str).strip()
    value_str = value_str.replace('\u2212', '-')  # Unicode minus → ASCII minus
    value_str = value_str.replace('\u2013', '-')  # En-dash → ASCII minus
    try:
        return float(value_str)
    except ValueError:
        return None


def load_sample(sample_dir: Path) -> Optional[dict]:
    """Load a sample group anchored to version_0.

    original_path   → original_captured_cropped.png (cropped to match edit framing)
    edited_path     → version_0/main_image.png       (primary edit; used for Dims 2/3/4)
    extra_versions  → [{version, path}]               (version_1+ edits; used only for D1b)

    D1a assumption: version_0 is aesthetically better than original (photographer-validated).
    D1b: version_1+ are alternative edits with unknown relative quality vs original.
    """
    # Use the cropped original so it matches the edited image's framing
    original_img = sample_dir / "original_captured_cropped.png"
    v0_dir = sample_dir / "version_0"
    v0_edited = v0_dir / "main_image.png"
    v0_details = v0_dir / "details.json"

    if not all(p.exists() for p in [original_img, v0_edited, v0_details]):
        return None

    with open(v0_details, "r", encoding="utf-8") as f:
        details = json.load(f)

    # Parse numeric parameters from version_0 (ground truth for Dims 2/3/4)
    numeric_params = {}
    mixer_max_abs = 0

    for item in details.get("details", []):
        if item.get("type") == "standard":
            label = item.get("label", "")
            value = item.get("value", "")
            parsed = parse_param_value(value)
            if parsed is not None and label in PARAM_KNOWLEDGE:
                numeric_params[label] = parsed

        elif item.get("type") == "color_mixer":
            # Collapse all color channels into one synthetic score.
            # Use max(|S|, |L|) — S (saturation) and L (luminosity) shifts drive
            # the visible color transformation; H (hue) is on a different perceptual
            # scale and is intentionally excluded from the magnitude estimate.
            for hsl_key in ("S", "L"):
                raw = item.get("values", {}).get(hsl_key, "")
                val = parse_param_value(raw)
                if val is not None:
                    mixer_max_abs = max(mixer_max_abs, abs(val))

    if mixer_max_abs > 0:
        numeric_params["色彩混合器"] = mixer_max_abs

    # Collect extra version paths for D1b (version_1, version_2, version_3)
    extra_versions = []
    for v in ["version_1", "version_2", "version_3"]:
        vpath = sample_dir / v / "main_image.png"
        if vpath.exists():
            extra_versions.append({"version": v, "path": vpath})

    return {
        "sample_id": sample_dir.name,
        "original_path": original_img,   # original_captured_cropped.png
        "edited_path": v0_edited,         # version_0 — primary edit
        "extra_versions": extra_versions, # version_1+ — alternative edits
        "details": details.get("details", []),
        "numeric_params": numeric_params,
        "step_count": details.get("step_count", 0),
        "image_stats": _compute_image_stats(original_img),  # for adaptive thresholding
    }


def collect_samples(split: str = "test", n: int = 50, seed: int = 42) -> list[dict]:
    """Collect valid samples from the dataset split.

    Only includes groups that have version_0 and original_captured_cropped.png,
    since version_0 is required as the D1a ground-truth edit.
    """
    split_dir = DATASET_ROOT / split
    sample_dirs = sorted(split_dir.iterdir())

    random.seed(seed)
    random.shuffle(sample_dirs)

    samples = []
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue
        sample = load_sample(sample_dir)
        if sample is not None:
            samples.append(sample)
        if len(samples) >= n:
            break

    multi = sum(1 for s in samples if s["extra_versions"])
    print(f"Collected {len(samples)} valid samples from '{split}' split "
          f"({multi} with extra versions for D1b).")
    return samples


# ============================================================
# SECTION 4: OPENROUTER API CLIENT
# ============================================================

def call_llm_vision(
    model_id: str,
    prompt: str,
    image_paths: list[Path],
    max_tokens: int = 512,
    api_key: str = None,
) -> Optional[str]:
    """Call a vision LLM via OpenRouter with one or more images.

    Images are sent as base64-encoded data URIs (no external uploads needed).
    """
    api_key = api_key or OPENROUTER_API_KEY
    if api_key == "YOUR_OPENROUTER_API_KEY_HERE":
        raise ValueError("Please set OPENROUTER_API_KEY in the config section.")

    # Fallback ladder: (max_bytes, quality, max_edge)
    # Step 0 — full quality, suitable for all dimensions
    # Step 1 — modest reduction; D2/D3/D4 still reliable
    # Step 2 — last resort; subtle tonal/color cues may be lost;
    #           D2 and D4 results at this tier are less trustworthy
    IMAGE_LADDER = [
        (4 * 1024 * 1024, 90, 1536),
        (1 * 1024 * 1024, 85, 1024),
        (512 * 1024,       75,  768),
    ]

    def _build_content(max_bytes: int, quality: int, max_edge: int) -> list:
        parts = []
        for img_path in image_paths:
            b64 = load_image_b64(img_path, max_bytes=max_bytes,
                                 quality=quality, max_edge=max_edge)
            if b64 is None:
                print(f"  [WARN] Image not found: {img_path}")
                continue
            b64_data, mime_type = b64
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            })
        parts.append({"type": "text", "text": prompt})
        return parts

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/adobe-lr-eval",  # Optional attribution
        "X-Title": "LR Aesthetic LLM Eval",
    }

    ladder_idx = 0  # start at highest quality tier

    for attempt in range(MAX_RETRIES):
        max_bytes, quality, max_edge = IMAGE_LADDER[ladder_idx]
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": _build_content(max_bytes, quality, max_edge)}],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for reproducibility
        }
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data:
                print(f"\n  [DEBUG] Unexpected response: {data}")
                raise KeyError("choices")
            content = data["choices"][0]["message"]["content"]
            if content is None:
                # Model returned null content — step down image quality and retry
                next_idx = ladder_idx + 1
                if next_idx < len(IMAGE_LADDER):
                    ladder_idx = next_idx
                    _, q, e = IMAGE_LADDER[ladder_idx]
                    print(f"  [WARN] {model_id}: null content, retrying at quality={q} max_edge={e}px"
                          f" (D2/D4 results may be less reliable at this setting)")
                    continue
                print(f"  [WARN] {model_id}: null content even at lowest image quality")
                return None
            return content.strip()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            print(f"  [HTTP {status}] {model_id}: {e}")
            if status == 429:  # Rate limited
                time.sleep(5 * (attempt + 1))
            elif status in (400, 422):
                # Model may not support vision or request format issue
                return None
            else:
                time.sleep(2)
        except Exception as e:
            print(f"  [ERR] {model_id} attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


# ============================================================
# SECTION 5: DIMENSION 1 — Pairwise Aesthetic Preference
# ============================================================
# Design rationale: pairwise comparison is more reliable than two independent
# absolute scores because it eliminates scale bias (a model that scores everything
# 8/10 can't game this) and halves API calls while reducing inter-call noise.
#
# Position randomization: we randomly swap which image appears first/second and
# record the mapping, so we can later check for position bias (LLMs often
# prefer whichever image is shown first).

def _make_pairwise_prompt(left_label: str, right_label: str) -> str:
    return f"""You are a professional photography reviewer. Please carefully compare the aesthetic quality of the two photos below.

Image order: First image = {left_label}, Second image = {right_label}

Please evaluate the images from the following four dimensions, providing a judgment for each dimension individually, and finally giving an overall conclusion:

【Dimension 1: Light and Shadow Layers】Is the distribution of light and dark natural? What is the quality of the handling of highlight details and shadow layers?
PREFER_LIGHT: <A (First image is better) / B (Second image is better) / C (Similar)>

【Dimension 2: Color Representation】Is the color temperature appropriate? How well does the color saturation harmonize with the overall tone?
PREFER_COLOR: <A / B / C>

【Dimension 3: Detail and Texture】Texture clarity, sharpness, and image delicacy
PREFER_DETAIL: <A / B / C>

【Dimension 4: Overall Visual Impact】First impression, visual tension, and emotional expression
PREFER_IMPACT: <A / B / C>

Based on the above four dimensions, give your overall aesthetic judgment:

PREFER: <A (First image) / B (Second image) / C (equivalent, difficult to distinguish) > REASON: <A one-sentence explanation of the core reason for the overall judgment>"""


def _parse_pairwise(response: str) -> dict:
    """Parse pairwise response into per-sub-dimension and overall answers.

    Returns a dict with keys: 'light', 'color', 'detail', 'impact', 'overall',
    each mapped to 'A', 'B', 'C', or None (unparseable).
    """
    import re

    if response is None:
        return {k: None for k in ("light", "color", "detail", "impact", "overall")}

    def _extract_field(tag: str) -> Optional[str]:
        """Extract the A/B/C answer for a given field tag (e.g. 'PREFER_LIGHT')."""
        for line in response.splitlines():
            if line.strip().upper().startswith(tag + ":"):
                val = line.split(":", 1)[1].strip().upper()
                m = re.match(r'^([ABC])', val)
                if m:
                    return m.group(1)
        # Fallback: find tag anywhere in the response
        m = re.search(rf'\b{tag}\b[^:]*:\s*([ABC])', response.upper())
        return m.group(1) if m else None

    # 'PREFER:' must be matched after 'PREFER_*' fields to avoid false hits.
    # The sub-dimension tags are unambiguous; for overall we look for a bare PREFER:.
    overall = _extract_field("PREFER")
    # _extract_field("PREFER") may pick up PREFER_LIGHT etc. if the bare PREFER line
    # is missing; scan lines explicitly for a bare "PREFER:" to be safe.
    for line in response.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("PREFER:") and not any(
            stripped.startswith(t) for t in ("PREFER_LIGHT", "PREFER_COLOR",
                                              "PREFER_DETAIL", "PREFER_IMPACT")
        ):
            val = line.split(":", 1)[1].strip().upper()
            m = re.match(r'^([ABC])', val)
            overall = m.group(1) if m else None
            break

    return {
        "light":  _extract_field("PREFER_LIGHT"),
        "color":  _extract_field("PREFER_COLOR"),
        "detail": _extract_field("PREFER_DETAIL"),
        "impact": _extract_field("PREFER_IMPACT"),
        "overall": overall,
    }


def _run_pairwise(model_id: str, original_path: Path, edited_path: Path,
                  salt: str, api_key: str) -> dict:
    """Single pairwise call with sub-dimension scoring.

    Position is randomized using `salt` for reproducibility.
    Returns per-sub-dimension correctness and an aggregate accuracy score.

    Ground truth: edited photo is better on every sub-dimension.
    Correctness per field: answer == edited_option → correct (1), 'C' (tie) or
    wrong option → incorrect (0).  accuracy = mean across all 5 fields.
    """
    rng = random.Random(salt)
    edited_is_first = rng.random() < 0.5

    if edited_is_first:
        images = [edited_path, original_path]
        left_label, right_label = "修图后照片", "原始照片"
        edited_option = "A"
    else:
        images = [original_path, edited_path]
        left_label, right_label = "原始照片", "修图后照片"
        edited_option = "B"

    prompt = _make_pairwise_prompt(left_label, right_label)
    resp = call_llm_vision(model_id, prompt, images, max_tokens=1500, api_key=api_key)
    time.sleep(REQUEST_DELAY)

    parsed = _parse_pairwise(resp)

    # Per-sub-dimension results
    subdim_keys = ("light", "color", "detail", "impact")
    subdim = {}
    for key in subdim_keys:
        ans = parsed[key]
        subdim[key] = {
            "answer": ans,
            "correct": (ans == edited_option) if ans is not None else None,
            "tie": ans == "C",
        }

    overall_ans = parsed["overall"]
    preferred_edited = (overall_ans == edited_option) if overall_ans is not None else None

    # accuracy: fraction of all 5 questions (4 sub-dims + overall) answered correctly
    all_answers = [subdim[k]["correct"] for k in subdim_keys] + [preferred_edited]
    answered = [c for c in all_answers if c is not None]
    accuracy = sum(answered) / len(answered) if answered else None

    return {
        "edited_position": "first" if edited_is_first else "second",
        "edited_option": edited_option,
        "subdim": subdim,
        "answer": overall_ans,
        "preferred_edited": preferred_edited,
        "tie": overall_ans == "C",
        "accuracy": accuracy,
        "raw": resp,
    }


def evaluate_dim1(model_id: str, sample: dict, api_key: str) -> dict:
    """Dimension 1: Multi-version pairwise aesthetic preference with sub-dimension scoring.

    D1a — original vs version_0 (discover/featured edit)
        Ground truth: version_0 is better (Adobe-featured photographer's edit).
        Scored on 4 sub-dimensions (light, color, detail, impact) + overall.

    D1b — original vs version_1 / version_2 / version_3 (remix edits)
        Ground truth: all remix edits are better than the unprocessed original
        (every version_n is a published professional LR edit from Adobe Discover).
        Same 5-question scoring as D1a.

    All pairwise positions are randomized per (sample_id, version) salt pair to
    allow per-version position-bias auditing.

    Composite D1 accuracy = mean of per-test accuracy across all versions tested.
    """
    sid = sample["sample_id"]

    d1a = _run_pairwise(
        model_id,
        original_path=sample["original_path"],
        edited_path=sample["edited_path"],   # version_0
        salt=f"{sid}_version_0",
        api_key=api_key,
    )

    d1b = []
    for ev in sample.get("extra_versions", []):
        result = _run_pairwise(
            model_id,
            original_path=sample["original_path"],
            edited_path=ev["path"],
            salt=f"{sid}_{ev['version']}",
            api_key=api_key,
        )
        result["version"] = ev["version"]
        d1b.append(result)

    # Aggregate accuracy across all pairwise tests (d1a + d1b)
    all_tests = [d1a] + d1b
    test_accuracies = [t["accuracy"] for t in all_tests if t.get("accuracy") is not None]
    mean_accuracy = sum(test_accuracies) / len(test_accuracies) if test_accuracies else None

    return {
        "d1a": d1a,
        "d1b": d1b,
        "n_tests": len(all_tests),
        "mean_accuracy": mean_accuracy,
    }


# ============================================================
# SECTION 6: DIMENSION 2 — Parameter Direction Consistency
# ============================================================

def build_direction_questions(sample: dict) -> list[dict]:
    """Build MCQ direction-perception questions for style-sensitive parameters.

    Only parameters in STYLE_SENSITIVE_PARAMS are tested — these are the ones
    whose effects are perceptually prominent enough for reliable LLM judgment.

    Special cases:
    - 色温: stored as absolute Kelvin; direction is computed against NEUTRAL_KELVIN (5500K).
            positive delta → warmer (higher K), negative delta → cooler (lower K).
    - All others: direction is determined by the sign of the JSON delta value.

    Perceptual gate (pixel-level ΔE):
    The primary visibility criterion is the mean CIE76 ΔE between the original and
    edited images, computed within the luminance zone that the parameter affects
    (see PARAM_LMASK).  A question is generated only when ΔE ≥ PERCEPTUAL_JND (2.3).
    This makes the gate image-content-aware:
    - A +30 shadow lift on a high-key image (no dark pixels) → masked ΔE ≈ 0 → skipped.
    - The same +30 on a backlit scene → large ΔE in the dark zone → question generated.
    - A 300 K colour temperature shift at 8500 K → small relative Δa*b* → may be skipped.

    Fallback: if image loading fails (PIL / NumPy unavailable), the old numeric
    threshold (DIRECTION_THRESHOLD or threshold_override / threshold_jnd_fraction)
    is used so the function remains operational without those libraries.
    """
    # Load both images as CIELAB arrays once; reuse for every parameter in this sample.
    lab_pair = _load_lab_pair(sample["original_path"], sample["edited_path"])

    questions = []
    for label, value in sample["numeric_params"].items():
        # Filter: only test style-sensitive parameters
        if label not in STYLE_SENSITIVE_PARAMS:
            continue

        info = PARAM_KNOWLEDGE.get(label)
        if info is None:
            continue

        # ── Step 1: determine the edit direction from the JSON parameter value ──
        # (The direction logic is unchanged; only the *gate* moves to ΔE.)
        if info.get("is_absolute_temp"):
            delta = value - NEUTRAL_KELVIN
            if delta == 0:
                continue
            direction = "positive" if delta > 0 else "negative"
            effective_value = delta

        elif info["is_delta"]:
            if value == 0:
                continue
            direction = "positive" if value > 0 else "negative"
            effective_value = value

        else:
            if value <= 0:
                continue
            direction = "positive"
            effective_value = value

        # ── Step 2: perceptual gate — is the edit actually visible? ─────────────
        if lab_pair is not None:
            orig_lab, edit_lab = lab_pair
            l_range = PARAM_LMASK.get(label)  # None → full-image ΔE
            delta_e = _masked_delta_e(orig_lab, edit_lab, l_range)
            if delta_e < PERCEPTUAL_JND:
                continue  # edit is below the human JND; do not ask the LLM
        else:
            # ── Fallback: numeric threshold (no image available) ─────────────
            if info.get("is_absolute_temp"):
                jnd_fraction = info.get("threshold_jnd_fraction")
                threshold = (jnd_fraction * value) if jnd_fraction else info.get(
                    "threshold_override", DIRECTION_THRESHOLD)
                if abs(effective_value) < threshold:
                    continue
            elif info["is_delta"]:
                threshold = info.get("threshold_override", DIRECTION_THRESHOLD)
                if abs(effective_value) < threshold:
                    continue
            else:
                threshold = info.get("threshold_override", DIRECTION_THRESHOLD)
                if effective_value < threshold:
                    continue

        # ── Step 3: build the question ────────────────────────────────────────
        effect_desc = info["positive_desc"] if direction == "positive" else info["negative_desc"]
        expected_options = ["A", "B"] if direction == "positive" else ["D", "E"]

        questions.append({
            "label": label,
            "value": effective_value,
            "direction": direction,
            "effect_desc": effect_desc,
            "question_topic": info["question"],
            "expected_options": expected_options,
        })

    return questions


def make_direction_prompt(question_topic: str, effect_desc: str) -> str:  # noqa: ARG001
    """Build a multi-stage CoT prompt for Dimension 2 (parameter direction perception).

    Three-stage reasoning guides the model from raw pixel observation → directional
    judgment → option selection, preventing the shortcut of guessing without grounding
    the answer in explicit visual evidence.

    effect_desc is accepted for API compatibility but not exposed to the model
    (it encodes the ground-truth direction, which would reveal the answer).
    """
    return f"""Please carefully compare the left image (original photo) and the right image (edited photo).

This task focuses on the image's: **{question_topic}**

Please strictly follow the three stages below, completing each stage completely before moving to the next:

【Stage 1 - Underlying Perception】
Focus only on the image area directly related to **{question_topic}**, objectively describing the original visual differences between the left and right images in that area. Do not make judgments; simply describe the phenomena you actually observe (e.g., changes in brightness, detail, and color).

OBSERVE: <1-2 sentences of pure description>

【Stage 2 - Perceptual Organization】
Based on the above observations, determine: Does this change "enhance/improve" or "weaken/reduce" this attribute? Is the change significant or subtle? Please explain your judgment.

ASSESS: <1 statement of direction + magnitude judgment>

【Phase Three · Option Consolidation】
Based on the reasoning from the first two phases, select the best match from the following options:

A. Significantly enhanced/brightened/improved (large change)
B. Slightly enhanced/brightened/improved (small change)
C. Virtually no change
D. Slightly weakened/darkened/decreased (small change)
E. Significantly weakened/darkened/decreased (large change)

ANSWER: <Single letter A/B/C/D/E>"""


def _extract_cot_field(response: str, field: str) -> "str | None":
    """Extract the text content of a named CoT field (e.g. 'OBSERVE', 'PERCEIVE').

    Handles both single-line and short multi-line values by collecting all text
    between the field marker and the next ALL-CAPS field marker or end-of-string.
    Used to archive reasoning stages alongside the final answer for offline analysis.
    """
    import re
    if not response:
        return None
    # Match FIELD: <content> up to next WORD: header or end of string
    m = re.search(
        rf"(?m)^{field}:\s*(.+?)(?=\n[A-Z]{{2,}}[^a-z\n]*:|$)",
        response,
        re.DOTALL,
    )
    if m:
        text = m.group(1).strip()
        return text[:400] if len(text) > 400 else text  # cap stored length
    return None


def parse_mcq_answer(response: str) -> Optional[str]:
    """Extract MCQ answer letter from response."""
    if response is None:
        return None
    for line in response.splitlines():
        if line.strip().startswith("ANSWER:"):
            ans = line.split(":", 1)[1].strip().upper()
            if ans and ans[0] in "ABCD":
                return ans[0]
    # Fallback
    import re
    m = re.search(r'\b([ABCD])\b', response.upper())
    return m.group(1) if m else None


def evaluate_dim2(model_id: str, sample: dict, api_key: str) -> dict:
    """Dimension 2: Test if LLM perceives the direction of large param adjustments."""
    questions = build_direction_questions(sample)
    if not questions:
        return {"skipped": True, "reason": "no params above threshold"}

    results = []
    for q in questions:
        prompt = make_direction_prompt(q["question_topic"], q["effect_desc"])
        resp = call_llm_vision(
            model_id,
            prompt,
            [sample["original_path"], sample["edited_path"]],
            max_tokens=400,   # increased from 150 to accommodate 3-stage CoT
            api_key=api_key,
        )
        time.sleep(REQUEST_DELAY)

        answer = parse_mcq_answer(resp)
        correct = answer in q["expected_options"] if answer else False

        results.append({
            "label": q["label"],
            "param_value": q["value"],
            "expected": q["expected_options"],
            "got": answer,
            "correct": correct,
            # CoT stages stored verbatim for offline reasoning-quality analysis
            "cot_observe": _extract_cot_field(resp, "OBSERVE"),
            "cot_assess":  _extract_cot_field(resp, "ASSESS"),
            "raw": resp,
        })

    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results) if results else 0.0

    return {
        "questions_asked": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "per_param": results,
    }


# ============================================================
# SECTION 7: DIMENSION 3 — Multi-parameter Diagnosis Accuracy
# ============================================================

DIM3_PROMPT = """Please carefully observe these two photos (left: original, right: edited).

Please strictly follow the three stages below, reasoning step by step, completing each stage before moving to the next, and finally providing an attribution conclusion:

【Stage One - Underlying Perception】 Objectively describe the significant visual differences between the two photos in the following three basic dimensions. Do not make attributions, only state the phenomena:

① Global brightness and exposure distribution (Overall brighter/darker? What changes occur in highlights or shadows?)

② Shadow detail and tonal range (Are shadows more transparent or more subdued? Is the tonal range richer?)

③ Color warmth/coolness and saturation (Overall tone cooler/warmer? Are colors more vibrant or more restrained?)

PERCEIVE: <3-5 objective descriptions, at least one for each dimension>

【Stage Two - Perceptual Organization】 How do the above underlying differences collectively affect the image structure? Please analyze the combined impact of these visual changes on the following aspects: the overall contrast structure of the image, the sense of spatial depth and layering, the prominence and visual weight of the subject.

ORGANIZE: <2-3 sentences of comprehensive analysis, demonstrating the correlation between various perceptual dimensions>

【Phase Three · Higher-Order Attribution】
Based on the visual evidence accumulated in the previous two phases, infer the three Lightroom adjustment categories most likely to produce this overall visual effect, listed in descending order of visual contribution.

Optional Categories:
1. Exposure/Brightness Adjustment (Exposure/Highlights/Shadows/Black and White Levels)
2. Contrast
3. Color Adjustment (Saturation/Vibrance/Hue)
4. Detail Processing (Sharpening/Clarity/Texture/Noise Reduction)
5. Clarity/Dehazing
6. Stylization (Vignetting/Grain)

TOP 3: <1st>, <2nd>, <3rd>"""

CATEGORY_MAP = {
    "1": "brightness",
    "2": "contrast",
    "3": "color",
    "4": "detail",
    "5": "clarity",
    "6": "style",
}

PARAM_TO_CATEGORY = {
    label: info["category"]
    for label, info in PARAM_KNOWLEDGE.items()
}

# Dimension 3: per-parameter contribution weights across the six diagnosis categories.
# Based on Lightroom colour-science priors; used by _compute_category_scores_v2 to
# augment pixel-level ablation scores with semantic parameter knowledge.
PARAM_FEATURE_WEIGHTS: dict[str, dict[str, float]] = {
    '去朦胧': {
        'color': 0.35,
        'contrast': 0.45,
        'clarity': 0.20,
    },
    '清晰度': {
        'contrast': 0.70,
        'clarity': 0.20,
        'brightness': 0.10,
    },
    '曝光度': {
        'brightness': 0.90,
        'contrast': 0.10,
    },
    '高光': {
        'brightness': 0.55,
        'contrast': 0.45,
    },
    '阴影': {
        'brightness': 0.65,
        'contrast': 0.35,
    },
    '对比度': {
        'contrast': 0.80,
        'brightness': 0.20,
    },
    '色温': {
        'color': 1.00,
    },
    '自然饱和度': {
        'color': 0.85,
        'clarity': 0.15,
    },
    '晕影': {
        'style': 0.70,
        'brightness': 0.30,
    },
    '颗粒': {
        'style': 0.80,
        'detail': 0.20,
    },
}


# ── Visual Ablation helpers ───────────────────────────────────────────────────
# Ground truth for Dim3 is derived by measuring the perceptual distance between
# the fully-edited image and a "category-ablated" approximation for each of the
# six adjustment categories.  Since we cannot re-render through Lightroom, each
# category's contribution is estimated in its most diagnostically relevant
# feature sub-space (see _compute_category_scores for the full mapping).

def _lum_weighted_L_delta(dL: "np.ndarray", L_orig: "np.ndarray") -> float:
    """Luminance-masked weighted mean |ΔL*|.

    Weights each pixel by sin(π·L*/100) so mid-tone changes (most perceptible
    to human vision per Weber-Fechner) contribute more than changes in pure
    black or specular-white zones.  This approximates the intent of a
    luminance-mask-weighted MS-SSIM without requiring a multi-scale pyramid.
    """
    import numpy as np
    w = np.sin(np.pi * np.clip(L_orig, 0, 100) / 100.0).clip(0)
    w_sum = w.sum()
    if w_sum == 0:
        return float(np.abs(dL).mean())
    return float((np.abs(dL) * w).sum() / w_sum)


def _rms_contrast(L: "np.ndarray") -> float:
    """Root-mean-square (Michelson) contrast of the L* channel."""
    import numpy as np
    return float(L.std())


def _laplacian_energy(L: "np.ndarray") -> float:
    """Mean absolute 4-connected Laplacian — high-frequency detail energy proxy."""
    import numpy as np
    lap = (L[1:-1, 2:] + L[1:-1, :-2]
           + L[2:, 1:-1] + L[:-2, 1:-1]
           - 4.0 * L[1:-1, 1:-1])
    return float(np.abs(lap).mean())


def _multiscale_gradient_dist(
    orig_lab: "np.ndarray", edit_lab: "np.ndarray", scales: int = 3
) -> float:
    """Multi-scale gradient magnitude distance on the L* channel.

    Serves as the LPIPS fallback.  Computes finite-difference gradient magnitudes
    at three resolution levels (full, ½, ¼) and takes a geometrically weighted
    sum, producing a perceptual distance that correlates with deep-feature
    distances for natural images without requiring PyTorch.
    """
    import numpy as np
    from PIL import Image as PILImage

    orig_L = orig_lab[..., 0].astype(np.float32)
    edit_L = edit_lab[..., 0].astype(np.float32)

    total, weight = 0.0, 1.0
    for s in range(scales):
        h, w = orig_L.shape
        # Horizontal and vertical finite differences (stride 2 for isotropy)
        ogx = orig_L[:, 2:] - orig_L[:, :-2]
        ogy = orig_L[2:, :] - orig_L[:-2, :]
        egx = edit_L[:, 2:] - edit_L[:, :-2]
        egy = edit_L[2:, :] - edit_L[:-2, :]
        rh = min(ogx.shape[0], ogy.shape[0], egx.shape[0], egy.shape[0])
        rw = min(ogx.shape[1], ogy.shape[1], egx.shape[1], egy.shape[1])
        orig_mag = np.sqrt(ogx[:rh, :rw] ** 2 + ogy[:rh, :rw] ** 2)
        edit_mag = np.sqrt(egx[:rh, :rw] ** 2 + egy[:rh, :rw] ** 2)
        total += weight * float(np.abs(orig_mag - edit_mag).mean())
        weight *= 0.5
        if s < scales - 1:
            new_w, new_h = max(1, w // 2), max(1, h // 2)
            orig_L = np.asarray(
                PILImage.fromarray(np.clip(orig_L, 0, 255).astype(np.uint8))
                        .resize((new_w, new_h), PILImage.LANCZOS),
                dtype=np.float32,
            )
            edit_L = np.asarray(
                PILImage.fromarray(np.clip(edit_L, 0, 255).astype(np.uint8))
                        .resize((new_w, new_h), PILImage.LANCZOS),
                dtype=np.float32,
            )
    return total


def _try_lpips(original_path: Path, edited_path: Path) -> "float | None":
    """Compute LPIPS (AlexNet) distance.  Returns None if torch/lpips unavailable."""
    try:
        import torch
        import lpips as lpips_lib
        from PIL import Image as PILImage
        import numpy as np

        loss_fn = lpips_lib.LPIPS(net="alex", verbose=False)

        def _to_tensor(p: Path):
            arr = np.asarray(PILImage.open(p).convert("RGB"), dtype=np.float32)
            t = torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            return t

        with torch.no_grad():
            return float(loss_fn(_to_tensor(original_path), _to_tensor(edited_path)).item())
    except Exception:
        return None


def _style_score(orig_lab: "np.ndarray", edit_lab: "np.ndarray") -> float:
    """Estimate the visual contribution of the style category (vignette + grain).

    Vignette — produces an asymmetric ΔL* between image corners and the centre.
    Grain    — increases local L* texture variance in originally smooth regions.

    Both are purely L*-channel phenomena and are computable without re-rendering.
    """
    import numpy as np

    H, W = orig_lab.shape[:2]
    dL = edit_lab[..., 0] - orig_lab[..., 0]

    # Vignette: corner vs centre mean ΔL* differential
    mh = max(1, H // 7)
    mw = max(1, W // 7)
    corners = np.concatenate([
        dL[:mh, :mw].ravel(), dL[:mh, -mw:].ravel(),
        dL[-mh:, :mw].ravel(), dL[-mh:, -mw:].ravel(),
    ])
    center_dL = dL[H // 4: 3 * H // 4, W // 4: 3 * W // 4].mean()
    vignette = abs(float(corners.mean() - center_dL))

    # Grain: increased local std in smooth original regions
    L_orig = orig_lab[..., 0]
    L_edit = edit_lab[..., 0]
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = 5
        o_std = sliding_window_view(L_orig, (win, win)).std(axis=(-2, -1))
        e_std = sliding_window_view(L_edit, (win, win)).std(axis=(-2, -1))
        smooth = o_std < 4.0
        grain = float(max(0.0, (e_std[smooth] - o_std[smooth]).mean())) if smooth.any() else 0.0
    except Exception:
        grain = 0.0

    # Grain numbers are small; ×3 to bring into the same perceptual ballpark as vignette.
    return vignette + grain * 3.0


# Scale factors that bring each raw metric into a common perceptual unit so that
# cross-category ranking is meaningful.  Calibrated so a "moderate but clearly
# visible" edit in any category produces a score in the range 5–20.
_CATEGORY_SCALE: dict[str, float] = {
    "brightness": 1.0,   # mean luminance-weighted |ΔL*| is already in L* units
    "contrast":   1.5,   # |Δσ(L*)| × 1.5 to align with brightness scale
    "color":      1.0,   # mean √(Δa*²+Δb*²) is in CIELAB units
    "clarity":   40.0,   # LPIPS (0–1) or gradient proxy scaled to L* range
    "detail":     6.0,   # Laplacian energy delta scaled up; downweighted ×0.5 below
    "style":      1.0,   # _style_score already returns L*-range values
}

# Laplacian-based detail scores are halved because JPEG compression and LLM
# image downscaling destroy most of the high-frequency signal introduced by
# Lightroom sharpening/noise-reduction adjustments.
_DETAIL_DOWNWEIGHT = 0.5


def _compute_category_scores(
    orig_lab: "np.ndarray",
    edit_lab: "np.ndarray",
    original_path: Path,
    edited_path: Path,
) -> dict[str, float]:
    """Per-category perceptual contribution scores for Dimension 3 ground truth.

    Replaces raw slider-magnitude ranking with feature-space ablation scoring:

    Category   │ Metric                                 │ Rationale
    ───────────┼────────────────────────────────────────┼──────────────────────────────────
    brightness │ luminance-weighted mean |ΔL*|           │ midtone-sensitive luminance change
    contrast   │ RMS-contrast delta |Δσ(L*)|             │ tonal-spread shift
    color      │ mean CIELAB chrominance dist √(Δa²+Δb²) │ hue/sat shift in a*b* plane
    clarity    │ LPIPS-AlexNet (→ gradient pyramid fallback) │ deep perceptual feature distance
    detail     │ Laplacian energy delta (×0.5 downweight)│ high-freq content change
    style      │ vignette asymmetry + grain proxy        │ spatially localised artefacts

    All scores are scaled to a common perceptual unit before ranking.
    """
    import numpy as np

    L_o, L_e = orig_lab[..., 0], edit_lab[..., 0]
    dL  = L_e - L_o
    dab = edit_lab[..., 1:] - orig_lab[..., 1:]

    brightness = _lum_weighted_L_delta(dL, L_o)
    contrast   = abs(_rms_contrast(L_e) - _rms_contrast(L_o))
    color      = float(np.sqrt((dab ** 2).sum(axis=2)).mean())

    # Prefer LPIPS; fall back to multi-scale gradient distance
    lpips_val  = _try_lpips(original_path, edited_path)
    clarity    = lpips_val if lpips_val is not None else _multiscale_gradient_dist(orig_lab, edit_lab)

    detail_raw = max(0.0, _laplacian_energy(L_e) - _laplacian_energy(L_o))
    detail     = detail_raw

    style      = _style_score(orig_lab, edit_lab)

    raw = {
        "brightness": brightness,
        "contrast":   contrast,
        "color":      color,
        "clarity":    clarity,
        "detail":     detail,
        "style":      style,
    }

    return {
        cat: raw[cat] * _CATEGORY_SCALE[cat] * (_DETAIL_DOWNWEIGHT if cat == "detail" else 1.0)
        for cat in raw
    }


def _compute_category_scores_v2(
    orig_lab: "np.ndarray",
    edit_lab: "np.ndarray",
    original_path: Path,
    edited_path: Path,
    numeric_params: dict,
) -> dict[str, float]:
    """Fused ground-truth scores for Dimension 3.

    Combines two complementary signals:
      - pixel_scores : pixel-level feature ablation (visual truth)
      - prior_scores : parameter-knowledge priors (semantic correctness)

    Each signal is normalised to a probability simplex before fusion so that
    differences in absolute magnitude do not bias the blend.  alpha=0.5 gives
    equal weight; adjust toward 0 (pure pixel) or 1 (pure prior) as needed.
    """
    import numpy as np

    pixel_scores = _compute_category_scores(orig_lab, edit_lab, original_path, edited_path)

    prior_scores: dict[str, float] = {cat: 0.0 for cat in pixel_scores}
    for label, value in numeric_params.items():
        if label not in PARAM_FEATURE_WEIGHTS:
            continue
        weights = PARAM_FEATURE_WEIGHTS[label]
        param_magnitude = abs(value) / 100.0  # normalise to [0, 1] (params live in [-100, 100])
        for cat, w in weights.items():
            prior_scores[cat] = prior_scores.get(cat, 0.0) + param_magnitude * w

    def _normalize(d: dict) -> dict:
        total = sum(d.values())
        if total == 0:
            return dict(d)
        return {k: v / total for k, v in d.items()}

    alpha = 0.5
    px_norm = _normalize(pixel_scores)
    pr_norm = _normalize(prior_scores)
    all_cats = set(px_norm) | set(pr_norm)
    return {
        cat: (1 - alpha) * px_norm.get(cat, 0.0) + alpha * pr_norm.get(cat, 0.0)
        for cat in all_cats
    }


def _numeric_category_scores(sample: dict) -> dict[str, float]:
    """Fallback ground truth: largest |parameter value| within each category.

    Used when image loading fails and pixel-level ablation is unavailable.
    Equivalent to the original get_top_categories logic.
    """
    scores: dict[str, float] = {}
    for label, value in sample["numeric_params"].items():
        cat = PARAM_TO_CATEGORY.get(label)
        if cat is None:
            continue
        scores[cat] = max(scores.get(cat, 0.0), abs(value))
    return scores


def get_top_categories(
    sample: dict, top_n: int = 3
) -> "tuple[list[str], dict[str, float]]":
    """Determine top-N categories by fused perceptual + prior score.

    Returns (ranked_category_names, full_score_dict).
    When both images can be loaded, uses _compute_category_scores_v2 which
    blends pixel-level feature ablation with parameter-knowledge priors.
    Falls back to numeric parameter magnitudes when image loading fails.
    """
    lab_pair = _load_lab_pair(sample["original_path"], sample["edited_path"])
    if lab_pair is not None:
        orig_lab, edit_lab = lab_pair
        scores = _compute_category_scores_v2(
            orig_lab, edit_lab,
            sample["original_path"], sample["edited_path"],
            sample["numeric_params"],
        )
    else:
        scores = _numeric_category_scores(sample)

    if not scores:
        return [], {}

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:top_n], scores


def parse_top3(response: str) -> list[str]:
    """Parse 'TOP3: 1,4,2' into category names."""
    if response is None:
        return []
    import re
    m = re.search(r'TOP3:\s*([1-6]),\s*([1-6]),\s*([1-6])', response)
    if m:
        return [CATEGORY_MAP.get(m.group(i), "") for i in range(1, 4)]
    digits = re.findall(r'\b([1-6])\b', response)
    return [CATEGORY_MAP.get(d, "") for d in digits[:3]]


def evaluate_dim3(model_id: str, sample: dict, api_key: str) -> dict:
    """Dimension 3: Measure overlap between LLM's top-3 diagnosis and ground truth.

    Ground truth is now derived from per-category perceptual ablation scores
    (pixel-level feature-space distances) rather than raw slider magnitudes.
    The ablation_scores dict is stored in the result for offline analysis.
    """
    ground_truth, ablation_scores = get_top_categories(sample)
    if not ground_truth:
        return {"skipped": True, "reason": "no numeric params"}

    resp = call_llm_vision(
        model_id,
        DIM3_PROMPT,
        [sample["original_path"], sample["edited_path"]],
        max_tokens=600,   # increased from 200 to accommodate 3-stage CoT
        api_key=api_key,
    )
    time.sleep(REQUEST_DELAY)

    predicted = parse_top3(resp)
    overlap = len(set(predicted) & set(ground_truth))
    overlap_rate = overlap / len(ground_truth) if ground_truth else 0.0

    return {
        "ground_truth_top3": ground_truth,
        "ablation_scores": ablation_scores,
        "predicted_top3": predicted,
        "overlap": overlap,
        "overlap_rate": overlap_rate,
        # CoT stages stored verbatim for reasoning-quality analysis
        "cot_perceive":  _extract_cot_field(resp, "PERCEIVE"),
        "cot_organize":  _extract_cot_field(resp, "ORGANIZE"),
        "raw": resp,
    }


# ============================================================
# SECTION 8: DIMENSION 4 — Destructive Artifact Detection
# ============================================================
# Design rationale: a parameter with |value| >= 60 is a necessary but not
# sufficient condition for a "true outlier."  A +75 shadow lift on a heavily
# backlit scene is corrective and produces no visual damage; the same +75 on an
# already bright scene may crush the highlight headroom and clip texture detail.
# We require *both* a numeric extreme AND objective image quality degradation
# evidence before generating a Dim4 question.  Compensatory edits (extreme
# parameter value but no measurable artifact) are silently skipped so the LLM
# is never penalised for failing to "see" something that isn't there.

# ── Artifact detection helpers ────────────────────────────────────────────────

def _check_histogram_clipping(edit_rgb: "np.ndarray") -> float:
    """Fraction of pixels showing highlight over-exposure or shadow crushing.

    Highlight clipping  — all channels ≥ 250 simultaneously (blown white).
    Shadow crushing     — all channels ≤ 5 simultaneously (dead black).
    Returns the larger of the two fractions as a severity score in [0, 1].
    """
    import numpy as np
    h_clip = float((edit_rgb >= 250).all(axis=2).mean())
    s_clip = float((edit_rgb <= 5).all(axis=2).mean())
    return max(h_clip, s_clip)


def _check_neon_chroma(edit_rgb: "np.ndarray") -> float:
    """Detect chroma/neon artifacts caused by over-saturation.

    Two mechanisms are detected in the edited RGB:
    1. Per-channel clipping (one channel maxed while others aren't) — indicates a
       colour has been pushed out of the sRGB gamut, producing a flat neon block.
    2. Among those chromatic-clip pixels, low local brightness variance confirms
       that the region has lost its tonal gradient (posterization / neon artifact).

    Score: fraction of total image area occupied by confirmed neon-clip regions.
    """
    import numpy as np

    r, g, b = edit_rgb[..., 0], edit_rgb[..., 1], edit_rgb[..., 2]

    # Per-channel clipping at the high end (any single channel ≥ 250)
    any_channel_clip = (r >= 250) | (g >= 250) | (b >= 250)
    # Exclude luminance clipping (all channels maxed = over-exposed white, not chroma)
    all_channel_clip = (r >= 250) & (g >= 250) & (b >= 250)
    chroma_clip_mask = any_channel_clip & ~all_channel_clip  # true chromatic clipping

    if chroma_clip_mask.sum() < 100:
        return 0.0

    # Within chroma-clipped pixels, check for low local brightness variance
    # (flat neon block ≡ high saturation + no texture = low local std of V)
    V = edit_rgb.max(axis=2).astype(np.float32) / 255.0  # HSV Value channel proxy
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = 5
        V_local_std = sliding_window_view(V, (win, win)).std(axis=(-2, -1))
        chroma_crop = chroma_clip_mask[2:-2, 2:-2]
        neon_mask = chroma_crop & (V_local_std < 0.04)
        return float(neon_mask.sum() / (edit_rgb.shape[0] * edit_rgb.shape[1]))
    except Exception:
        return float(chroma_clip_mask.mean())  # simpler fallback


def _check_halo_artifacts(
    orig_lab: "np.ndarray", edit_lab: "np.ndarray"
) -> float:
    """Detect spatial halo / dark-fringe artefacts at high-contrast edges.

    Over-clarity or over-dehaze creates a characteristic signature: luminance
    changes (ΔL*) are concentrated near original high-contrast edges rather than
    being uniformly distributed across the image.

    Algorithm:
      1. Compute gradient magnitude of original L* channel.
      2. Threshold at the 90th percentile to find strong-edge pixels.
      3. Dilate the edge mask by 7 px to create an "edge neighbourhood."
      4. Compare mean |ΔL*| inside vs outside the neighbourhood.
      Score = (edge_zone_delta / non_edge_delta) − 1, clamped to [0, ∞).
      A score > 0.3 indicates that luminance changes are 30 % more concentrated
      at edges than in flat regions, the hallmark of a halo artifact.
    """
    import numpy as np

    L_o = orig_lab[..., 0].astype(np.float32)
    dL  = np.abs(edit_lab[..., 0] - L_o)

    # Gradient magnitude on original L*
    gx = L_o[:, 2:] - L_o[:, :-2]
    gy = L_o[2:, :] - L_o[:-2, :]
    rh = min(gx.shape[0], gy.shape[0])
    rw = min(gx.shape[1], gy.shape[1])
    grad_mag = np.sqrt(gx[:rh, :rw] ** 2 + gy[:rh, :rw] ** 2)

    edge_mask = grad_mag >= np.percentile(grad_mag, 90)

    # Dilate edge mask with a 15×15 sliding-window max (no scipy required)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = 7
        padded = np.pad(edge_mask.astype(np.float32), pad, mode="constant")
        edge_zone = sliding_window_view(padded, (2 * pad + 1, 2 * pad + 1)).max(
            axis=(-2, -1)
        ) > 0
    except Exception:
        edge_zone = edge_mask  # no dilation fallback

    # Align ΔL* to the gradient coordinate space
    dL_crop = dL[1:-1, 1:-1][:rh, :rw]

    edge_delta = float(dL_crop[edge_zone].mean()) if edge_zone.any() else 0.0
    non_edge = ~edge_zone
    non_edge_delta = float(dL_crop[non_edge].mean()) if non_edge.any() else None

    if non_edge_delta is None or non_edge_delta < 0.1:
        return 0.0

    # Normalised ratio: 0 = no bias, 1 = edge delta is 2× non-edge delta
    return float(max(0.0, (edge_delta / non_edge_delta) - 1.0))


# Maps each parameter category to the artifact type it can produce when over-applied.
# None → the category's extremes rarely manifest as objectively detectable image damage.
_CATEGORY_ARTIFACT_TYPE: dict[str, "str | None"] = {
    "brightness": "clipping",
    "contrast":   "clipping",
    "color":      "chroma",
    "clarity":    "halo",
    "detail":     "halo",
    "style":      None,
}

# Minimum artifact severity to confirm that a numeric extreme has caused visual damage.
# Values are calibrated so that clearly visible artefacts exceed the threshold while
# subtle rendering differences (below human JND) remain below it.
_ARTIFACT_THRESHOLDS: dict[str, float] = {
    "clipping": 0.030,  # 3 % of pixels with blown highlights or crushed blacks
    "chroma":   0.010,  # 1 % of image area showing neon/chromatic-clip artefact
    "halo":     0.300,  # edge-zone ΔL* ≥ 30 % larger than non-edge ΔL*
}


def _compute_artifact_scores(
    orig_lab: "np.ndarray",
    edit_lab: "np.ndarray",
    edit_rgb: "np.ndarray",
) -> dict[str, float]:
    """Return raw severity scores for each artifact type."""
    return {
        "clipping": _check_histogram_clipping(edit_rgb),
        "chroma":   _check_neon_chroma(edit_rgb),
        "halo":     _check_halo_artifacts(orig_lab, edit_lab),
    }


# ── Core Dim4 functions ───────────────────────────────────────────────────────

def find_extreme_params(sample: dict, threshold: float = None) -> list[dict]:
    """Find parameters whose |value| ≥ threshold (numeric gate only).

    This is the first pass.  The second pass (_confirm_by_artifact) removes
    entries whose category-specific artifact is below the detection threshold,
    keeping only true over-processing cases with measurable visual damage.
    """
    threshold = threshold or OUTLIER_THRESHOLD
    extremes = []
    for label, value in sample["numeric_params"].items():
        info = PARAM_KNOWLEDGE.get(label)
        if info is None:
            continue
        if abs(value) >= threshold:
            extremes.append({
                "label": label,
                "value": value,
                "category": info["category"],
                "en": info["en"],
            })
    return extremes


def _confirm_by_artifact(
    extremes: list[dict],
    artifact_scores: dict[str, float],
) -> list[dict]:
    """Filter extreme params to those with confirmed image quality degradation.

    For each extreme parameter, look up the artifact type its category produces
    (_CATEGORY_ARTIFACT_TYPE) and check whether the measured artifact score
    exceeds the perceptual damage threshold (_ARTIFACT_THRESHOLDS).

    Params whose category maps to None (e.g. style/晕影) are included as-is
    when no better evidence is available; categories with a recognised artifact
    type are excluded if the corresponding score is below the threshold.
    """
    confirmed = []
    for e in extremes:
        art_type = _CATEGORY_ARTIFACT_TYPE.get(e["category"])
        if art_type is None:
            confirmed.append(e)  # no artifact model → include by default
            continue
        severity = artifact_scores.get(art_type, 0.0)
        if severity >= _ARTIFACT_THRESHOLDS[art_type]:
            e = dict(e, artifact_type=art_type, artifact_severity=severity)
            confirmed.append(e)
    return confirmed


# Artifact-oriented defect-localisation prompt.
# Replaces the original "something is extremely adjusted" framing with a
# specific visual quality inspection task, eliminating cognitive ambiguity
# from compensatory edits that have high slider values but no visual damage.
DIM4_PROMPT = """Please carefully compare the left image (original photo) and the right image (edited photo).

Please determine: Does the edited photo suffer from unnatural visual flaws due to **excessive post-processing**?

Common overprocessing defects include:

• Severe blown-out highlights/shadows, loss of texture detail (pure white or pure black areas)
• Color areas become flat, solid color blocks without gradients, or neon colors/chromatic noise appear
• Unnatural halos or black edges appear at high-contrast edges (such as the boundary between the sky and mountain ridges)

If **the above defects exist**, please reply strictly according to the following format:

ARTIFACT: Yes

CAUSE: <1-2 numbers, separated by commas, select the most likely cause from the following categories>

DESCRIBE: <A one-sentence description of the specific location and manifestation of the defect you observed>

If **no** obvious defects exist, please reply:

ARTIFACT: No

Optional categories:

1. Exposure/Brightness Adjustment (pure white/pure black/banding)
2. Contrast (abnormal tonal gradation)
3. Color Adjustment (neon color blocks/hue breaks/chromatic noise)
4. Detail Refinement (over-sharpening causing edge noise)
5. Clarity/De-hazing (Edge halo/Glowing effect)
6. Stylization (Excessive vignette/Overly grainy)"""


def _parse_artifact_response(response: str) -> dict:
    """Parse the defect-localisation response format.

    Returns:
        artifact_detected  — True / False / None (unparseable)
        predicted_cats     — list of category name strings (may be empty)
        description        — free-text description or None
    """
    import re

    if response is None:
        return {"artifact_detected": None, "predicted_cats": [], "description": None}

    artifact_detected = None
    for line in response.splitlines():
        s = line.strip()
        if s.startswith("ARTIFACT:"):
            # Prompt 要求模型用英文回答 Yes/No；同时兼容中文「是/否」以防模型混用语言
            val = s.split(":", 1)[1].strip().lower()
            if val.startswith("yes") or val.startswith("是"):
                artifact_detected = True
            elif val.startswith("no") or val.startswith("否"):
                artifact_detected = False
            # 其余情况保持 None（无法解析）
            break

    predicted_cats: list[str] = []
    m = re.search(r"CAUSE:\s*([\d,\s]+)", response)
    if m:
        digits = re.findall(r"\b([1-6])\b", m.group(1))
        predicted_cats = [CATEGORY_MAP.get(d, "") for d in digits[:2]]

    description = None
    m = re.search(r"DESCRIBE:\s*(.+)", response)
    if m:
        description = m.group(1).strip()

    return {
        "artifact_detected": artifact_detected,
        "predicted_cats": predicted_cats,
        "description": description,
    }


def evaluate_dim4(model_id: str, sample: dict, api_key: str) -> dict:
    """Dimension 4: Destructive artifact detection and causal attribution.

    Two-gate qualification:
      Gate 1 (numeric)  — at least one parameter |value| ≥ OUTLIER_THRESHOLD.
      Gate 2 (visual)   — the expected artifact for that parameter's category is
                          objectively detectable in the rendered pixel data.

    Samples that pass Gate 1 but fail Gate 2 are compensatory edits (e.g. a
    heavily backlit shot needing +80 shadows) — they are skipped with reason
    'no_confirmed_artifact' so the LLM is never tested against invisible damage.

    Scoring:
      hit = True  if model says ARTIFACT:是 AND names at least one correct category.
      hit = False if model misses the artifact or blames the wrong category.
    """
    import re

    # Gate 1: numeric threshold
    extremes = find_extreme_params(sample)
    if not extremes:
        return {"skipped": True, "reason": "no_extreme_params"}

    # Load image pair for artifact detection; fall back to numeric-only mode
    try:
        import numpy as np
        from PIL import Image as PILImage

        edit_rgb = np.asarray(PILImage.open(sample["edited_path"]).convert("RGB"))
        orig_rgb = np.asarray(PILImage.open(sample["original_path"]).convert("RGB"))
        if orig_rgb.shape != edit_rgb.shape:
            h = min(orig_rgb.shape[0], edit_rgb.shape[0])
            w = min(orig_rgb.shape[1], edit_rgb.shape[1])
            orig_rgb, edit_rgb = orig_rgb[:h, :w], edit_rgb[:h, :w]

        orig_lab = _srgb_to_lab(orig_rgb)
        edit_lab = _srgb_to_lab(edit_rgb)
        artifact_scores = _compute_artifact_scores(orig_lab, edit_lab, edit_rgb)
        confirmed = _confirm_by_artifact(extremes, artifact_scores)
    except Exception:
        # No image access — skip artifact gate, trust numeric threshold
        confirmed = extremes
        artifact_scores = {}

    # Gate 2: confirmed artifact
    if not confirmed:
        return {
            "skipped": True,
            "reason": "no_confirmed_artifact",
            "numeric_extremes": [{"label": e["label"], "value": e["value"]} for e in extremes],
            "artifact_scores": artifact_scores,
        }

    ground_truth_cats = list({e["category"] for e in confirmed})

    resp = call_llm_vision(
        model_id,
        DIM4_PROMPT,
        [sample["original_path"], sample["edited_path"]],
        max_tokens=250,
        api_key=api_key,
    )
    time.sleep(REQUEST_DELAY)

    parsed = _parse_artifact_response(resp)
    predicted_cats = parsed["predicted_cats"]

    # Correct iff model flagged an artifact AND attributed it to the right category
    artifact_correctly_detected = parsed["artifact_detected"] is True
    cat_hit = any(c in ground_truth_cats for c in predicted_cats)
    hit = artifact_correctly_detected and cat_hit

    return {
        "confirmed_extremes": [
            {"label": e["label"], "value": e["value"], "category": e["category"]}
            for e in confirmed
        ],
        "artifact_scores": artifact_scores,
        "ground_truth_categories": ground_truth_cats,
        "artifact_detected_by_model": parsed["artifact_detected"],
        "predicted_categories": predicted_cats,
        "artifact_description": parsed["description"],
        "overlap": len(set(predicted_cats) & set(ground_truth_cats)),
        "hit": hit,
        "raw": resp,
    }


# ============================================================
# SECTION 9: MAIN EVALUATION LOOP
# ============================================================

def _evaluate_model_worker(
    model_name: str,
    model_id: str,
    samples: list[dict],
    all_results: dict,
    results_file: Path,
    dimensions: list[int],
    api_key: str,
    lock: threading.Lock,
    n_samples: int,
):
    """Worker function that evaluates one model across all samples.

    Runs in its own thread. Uses `lock` when updating the shared `all_results`
    dict and writing raw_results.json so concurrent workers don't corrupt the file.
    """
    tag = f"[{model_name}]"

    print(f"\n{'='*60}")
    print(f"{tag} Model: {model_id}")
    print(f"{'='*60}")

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]

        with lock:
            already_done = sid in all_results[model_name]
        if already_done:
            print(f"  {tag} [{i+1}/{n_samples}] {sid[:16]}... (cached, skipping)")
            continue

        parts = []
        sample_result = {
            "sample_id": sid,
            "version": "version_0",
            "params_count": len(sample["numeric_params"]),
        }

        try:
            if 1 in dimensions:
                sample_result["dim1"] = evaluate_dim1(model_id, sample, api_key)
                d1_acc = sample_result["dim1"].get("mean_accuracy")
                d1_n   = sample_result["dim1"].get("n_tests", 1)
                acc_str = f"{d1_acc:.0%}" if d1_acc is not None else "?"
                parts.append(f"D1✓({d1_n}v,{acc_str})")

            if 2 in dimensions:
                sample_result["dim2"] = evaluate_dim2(model_id, sample, api_key)
                n_q = sample_result["dim2"].get("questions_asked", 0)
                acc = sample_result["dim2"].get("accuracy", 0)
                parts.append(f"D2✓({n_q}q,{acc:.0%})")

            if 3 in dimensions:
                sample_result["dim3"] = evaluate_dim3(model_id, sample, api_key)
                ov = sample_result["dim3"].get("overlap_rate", 0)
                parts.append(f"D3✓({ov:.0%})")

            if 4 in dimensions:
                sample_result["dim4"] = evaluate_dim4(model_id, sample, api_key)
                hit = sample_result["dim4"].get("hit", False)
                parts.append(f"D4✓({'hit' if hit else 'miss'})")

        except Exception as e:
            parts.append(f"ERROR:{e}")
            sample_result["error"] = str(e)

        # Print as one atomic line so output from parallel threads stays readable
        print(f"  {tag} [{i+1}/{n_samples}] {sid[:16]}... {' '.join(parts)}")

        with lock:
            all_results[model_name][sid] = sample_result
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)


def run_evaluation(
    models: dict,
    samples: list[dict],
    output_dir: Path,
    dimensions: list[int],
    api_key: str,
):
    """Run all dimensions for all models across all samples.

    Models are evaluated in parallel (one thread per model). Results are saved
    incrementally after each sample so a crash only loses the current sample.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "raw_results.json"

    # Load existing results for resuming interrupted runs
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Resuming from {results_file} ({len(all_results)} existing entries)")
    else:
        all_results = {}

    # Pre-populate model keys so workers can safely check without the lock
    for model_name in models:
        if model_name not in all_results:
            all_results[model_name] = {}

    lock = threading.Lock()
    n_samples = len(samples)

    print(f"\nRunning {len(models)} model(s) in parallel...")

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                _evaluate_model_worker,
                model_name, model_id, samples, all_results,
                results_file, dimensions, api_key, lock, n_samples,
            ): model_name
            for model_name, model_id in models.items()
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
                print(f"\n[{model_name}] finished.")
            except Exception as e:
                print(f"\n[{model_name}] FAILED: {e}")

    print(f"\nAll models done. Results saved to {results_file}")
    return all_results


# ============================================================
# SECTION 10: RESULTS ANALYSIS
# ============================================================

def aggregate_results(all_results: dict, dimensions: list[int]) -> dict:
    """Compute per-model aggregate scores for each dimension."""
    summary = {}

    def _blank_subdim():
        return {"light": [0, 0], "color": [0, 0],
                "detail": [0, 0], "impact": [0, 0], "overall": [0, 0]}

    for model_name, samples in all_results.items():
        # D1: multi-version, multi-sub-dimension pairwise
        d1_accuracies  = []   # all tests combined
        d1a_accuracies = []   # version_0 vs original only
        d1b_accuracies = []   # version_1/2/3 vs original only

        d1_subdim_correct  = _blank_subdim()   # all tests
        d1a_subdim_correct = _blank_subdim()   # D1a only
        d1b_subdim_correct = _blank_subdim()   # D1b only

        d1_first_preferred = 0
        d1_total = 0

        dim2_accuracies = []
        dim2_per_param = {}  # label -> [correct, total]

        dim3_overlaps = []

        dim4_hits = []

        for sid, result in samples.items():
            if "error" in result:
                continue

            # Dim 1 — iterate over all pairwise tests in this sample
            d1 = result.get("dim1")
            if d1:
                tagged = (
                    [("a", d1.get("d1a"))]
                    + [("b", t) for t in d1.get("d1b", [])]
                )
                for tag, t in tagged:
                    if t is None:
                        continue
                    d1_total += 1

                    if t.get("accuracy") is not None:
                        d1_accuracies.append(t["accuracy"])
                        if tag == "a":
                            d1a_accuracies.append(t["accuracy"])
                        else:
                            d1b_accuracies.append(t["accuracy"])

                    # position-bias audit: preferred whichever image was shown first
                    if t.get("answer") == "A":
                        d1_first_preferred += 1

                    # per-sub-dimension counts — update both combined and split trackers
                    subdim_targets = [d1_subdim_correct]
                    if tag == "a":
                        subdim_targets.append(d1a_subdim_correct)
                    else:
                        subdim_targets.append(d1b_subdim_correct)

                    for key in ("light", "color", "detail", "impact"):
                        sd = (t.get("subdim") or {}).get(key, {})
                        if sd.get("correct") is not None:
                            for tracker in subdim_targets:
                                tracker[key][1] += 1
                                if sd["correct"]:
                                    tracker[key][0] += 1
                    # overall
                    if t.get("answer") is not None:
                        for tracker in subdim_targets:
                            tracker["overall"][1] += 1
                            if t.get("preferred_edited"):
                                tracker["overall"][0] += 1

            # Dim 2
            d2 = result.get("dim2")
            if d2 and not d2.get("skipped"):
                acc = d2.get("accuracy", 0)
                if d2.get("questions_asked", 0) > 0:
                    dim2_accuracies.append(acc)
                for pr in d2.get("per_param", []):
                    lbl = pr["label"]
                    if lbl not in dim2_per_param:
                        dim2_per_param[lbl] = [0, 0]
                    dim2_per_param[lbl][1] += 1
                    if pr["correct"]:
                        dim2_per_param[lbl][0] += 1

            # Dim 3
            d3 = result.get("dim3")
            if d3 and not d3.get("skipped") and "overlap_rate" in d3:
                dim3_overlaps.append(d3["overlap_rate"])

            # Dim 4
            d4 = result.get("dim4")
            if d4 and not d4.get("skipped") and "hit" in d4:
                dim4_hits.append(1 if d4["hit"] else 0)

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else None

        def _subdim_rate(tracker, key):
            c, t = tracker[key]
            return c / t if t > 0 else None

        dim1_score = safe_mean(d1_accuracies)
        dim2_score = safe_mean(dim2_accuracies)
        dim3_score = safe_mean(dim3_overlaps)
        dim4_score = safe_mean(dim4_hits)

        # Composite score (only include evaluated dimensions)
        composite = 0.0
        total_weight = 0.0
        dim_scores = {
            "dim1": dim1_score,
            "dim2": dim2_score,
            "dim3": dim3_score,
            "dim4": dim4_score,
        }
        dim_key_map = {1: "dim1", 2: "dim2", 3: "dim3", 4: "dim4"}
        for d in dimensions:
            key = dim_key_map[d]
            score = dim_scores.get(key)
            if score is not None:
                w = DIM_WEIGHTS[key]
                composite += w * score
                total_weight += w
        if total_weight > 0:
            composite /= total_weight  # Normalize to [0,1]

        n_errors = sum(1 for r in samples.values() if "error" in r)
        summary[model_name] = {
            "samples_evaluated": len(samples),
            "samples_with_errors": n_errors,
            "dim1": {
                # ── combined (D1a + D1b) ──────────────────────────────────
                "mean_accuracy": dim1_score,
                "subdim_accuracy": {
                    k: _subdim_rate(d1_subdim_correct, k)
                    for k in ("light", "color", "detail", "impact", "overall")
                },
                "first_image_bias": d1_first_preferred / d1_total if d1_total else None,
                "n_tests": d1_total,
                # ── D1a: version_0 (featured discover edit) vs original ───
                "d1a": {
                    "mean_accuracy": safe_mean(d1a_accuracies),
                    "subdim_accuracy": {
                        k: _subdim_rate(d1a_subdim_correct, k)
                        for k in ("light", "color", "detail", "impact", "overall")
                    },
                    "n_tests": len(d1a_accuracies),
                },
                # ── D1b: version_1/2/3 (remix edits) vs original ──────────
                "d1b": {
                    "mean_accuracy": safe_mean(d1b_accuracies),
                    "subdim_accuracy": {
                        k: _subdim_rate(d1b_subdim_correct, k)
                        for k in ("light", "color", "detail", "impact", "overall")
                    },
                    "n_tests": len(d1b_accuracies),
                },
            },
            "dim2": {
                "mean_accuracy": dim2_score,
                "per_param_accuracy": {
                    lbl: v[0] / v[1] if v[1] > 0 else 0
                    for lbl, v in dim2_per_param.items()
                },
                "n": len(dim2_accuracies),
            },
            "dim3": {
                "mean_overlap_rate": dim3_score,
                "n": len(dim3_overlaps),
            },
            "dim4": {
                "hit_rate": dim4_score,
                "n": len(dim4_hits),
            },
            "composite_score": composite,
        }

    return summary


def print_report(summary: dict):
    """Print a formatted leaderboard."""
    # ── Evaluation progress table ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EVALUATION PROGRESS")
    print("=" * 50)
    print(f"  {'Model':<20} {'Evaluated':>10}  {'Errors':>7}")
    print("-" * 50)
    for model_name, s in sorted(summary.items()):
        n_eval  = s.get("samples_evaluated", 0)
        n_err   = s.get("samples_with_errors", 0)
        print(f"  {model_name:<20} {n_eval:>10}  {n_err:>7}")
    print("=" * 50)

    print("\n" + "=" * 70)
    print("  LLM AESTHETIC ALIGNMENT LEADERBOARD")
    print("=" * 70)

    # Sort by composite score
    ranked = sorted(
        summary.items(),
        key=lambda x: x[1].get("composite_score") or 0,
        reverse=True,
    )

    header = (f"{'Rank':<5} {'Model':<20} {'D1 Aes':>9} "
              f"{'Light':>7} {'Color':>7} {'Detail':>7} {'Impact':>7} "
              f"{'D2 Dir':>8} {'D3 Diag':>8} {'D4 Out':>8} {'Composite':>10}")
    print(header)
    print("-" * 100)

    for rank, (model_name, s) in enumerate(ranked, 1):
        d1   = s["dim1"]["mean_accuracy"]
        sd   = s["dim1"]["subdim_accuracy"]
        d2   = s["dim2"]["mean_accuracy"]
        d3   = s["dim3"]["mean_overlap_rate"]
        d4   = s["dim4"]["hit_rate"]
        comp = s["composite_score"]

        fmt = lambda v: f"{v:.1%}" if v is not None else "  N/A "
        print(
            f"{rank:<5} {model_name:<20} {fmt(d1):>9} "
            f"{fmt(sd.get('light')):>7} {fmt(sd.get('color')):>7} "
            f"{fmt(sd.get('detail')):>7} {fmt(sd.get('impact')):>7} "
            f"{fmt(d2):>8} {fmt(d3):>8} {fmt(d4):>8} {fmt(comp):>10}"
        )

    print("=" * 100)
    print("\nDimension key:")
    print("  D1 Aes  : [PRIMARY] Mean accuracy across all sub-dimensions and all version pairs")
    print("    Light : % where LLM correctly identifies edited as having better light/shadow")
    print("    Color : % correct for colour rendition sub-dimension")
    print("    Detail: % correct for detail/texture sub-dimension")
    print("    Impact: % correct for overall visual impact sub-dimension")
    print("  D2 Dir  : Accuracy in perceiving direction of large parameter adjustments")
    print("  D3 Diag : Overlap rate between top-3 diagnosed categories and ground truth")
    print("  D4 Out  : Hit rate on identifying extremely-adjusted parameters")
    print(f"\nWeights: D1={DIM_WEIGHTS['dim1']:.0%} D2={DIM_WEIGHTS['dim2']:.0%} "
          f"D3={DIM_WEIGHTS['dim3']:.0%} D4={DIM_WEIGHTS['dim4']:.0%}")

    # ── D1 detailed breakdown ─────────────────────────────────────────────────
    subdim_labels = [
        ("light",   "光影"),
        ("color",   "色彩"),
        ("detail",  "细节"),
        ("impact",  "冲击"),
        ("overall", "整体"),
    ]
    subdim_header = "  ".join(f"{lbl:>5}" for _, lbl in subdim_labels)

    def _fmt_subdim_row(sd: dict) -> str:
        fmt = lambda v: f"{v:>5.1%}" if v is not None else "  N/A"
        return "  ".join(fmt(sd.get(k)) for k, _ in subdim_labels)

    for section_title, d1_key in [
        ("Dimension 1 (All versions) — sub-dimension accuracy per model", None),
        ("Dimension 1a (version_0 / featured edit vs original)", "d1a"),
        ("Dimension 1b (version_1/2/3 / remix edits vs original)", "d1b"),
    ]:
        print(f"\n--- {section_title} ---")
        print(f"  {'Model':<20} {'Overall':>8}  {subdim_header}  {'n_tests':>7}  {'pos_bias':>8}")
        print("  " + "-" * 85)
        for model_name, s in sorted(summary.items()):
            block = s["dim1"] if d1_key is None else s["dim1"].get(d1_key, {})
            overall  = block.get("mean_accuracy")
            sd       = block.get("subdim_accuracy", {})
            n        = block.get("n_tests", 0)
            bias     = s["dim1"]["first_image_bias"] if d1_key is None else None
            bias_str = f"{bias:.2f}" if bias is not None else "   —  "
            fmt_ov   = f"{overall:>8.1%}" if overall is not None else "    N/A "
            print(f"  {model_name:<20} {fmt_ov}  {_fmt_subdim_row(sd)}  {n:>7}  {bias_str:>8}")

    # Per-param breakdown for Dim 2
    print("\n--- Dimension 2: Per-parameter Accuracy (aggregated across all models) ---")
    param_totals = {}
    for model_name, s in summary.items():
        for param, acc in s["dim2"]["per_param_accuracy"].items():
            n = summary[model_name]["dim2"]["n"]
            if param not in param_totals:
                param_totals[param] = []
            param_totals[param].append(acc)

    for param in sorted(param_totals, key=lambda p: -sum(param_totals[p])/len(param_totals[p])):
        accs = param_totals[param]
        avg = sum(accs) / len(accs)
        print(f"  {param:<12}: avg {avg:.1%} across {len(accs)} model(s)")


def save_summary(summary: dict, output_dir: Path):
    """Save analysis summary to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_file}")

    # CSV leaderboard
    csv_file = output_dir / "leaderboard.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        # 字段与 aggregate_results 产出的 summary 结构保持一致
        f.write("model,composite_score,"
                "d1_mean_accuracy,d1a_accuracy,d1b_accuracy,d1_first_image_bias,d1_n_tests,"
                "dim2_accuracy,dim3_overlap,dim4_hit_rate,samples\n")
        for model_name, s in sorted(summary.items(), key=lambda x: x[1].get("composite_score") or 0, reverse=True):
            d1 = s["dim1"]
            row = [
                model_name,
                s.get("composite_score"),
                d1.get("mean_accuracy"),
                d1.get("d1a", {}).get("mean_accuracy"),
                d1.get("d1b", {}).get("mean_accuracy"),
                d1.get("first_image_bias"),
                d1.get("n_tests"),
                s["dim2"]["mean_accuracy"],
                s["dim3"]["mean_overlap_rate"],
                s["dim4"]["hit_rate"],
                s["samples_evaluated"],
            ]
            f.write(",".join(str(v) if v is not None else "" for v in row) + "\n")
    print(f"Leaderboard CSV saved to {csv_file}")


# ============================================================
# SECTION 11: CLI ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on Lightroom aesthetic alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES,
                        help=f"Number of samples to evaluate (default: {NUM_SAMPLES})")
    parser.add_argument("--split", default=SPLIT, choices=["train", "test"],
                        help=f"Dataset split (default: {SPLIT})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=Path("eval_results"),
                        help="Output directory for results")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Models to evaluate (default: all). e.g. --models gpt-4o claude-sonnet")
    parser.add_argument("--dim", nargs="+", type=int, default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4],
                        help="Dimensions to evaluate (default: all)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip evaluation, just re-analyze existing raw_results.json")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key (overrides config)")
    args = parser.parse_args()

    api_key = args.api_key or OPENROUTER_API_KEY
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    if args.models is None or "all" in (args.models or []):
        selected_models = MODELS
    else:
        selected_models = {k: MODELS[k] for k in args.models if k in MODELS}

    print(f"LLM Aesthetic Alignment Evaluator")
    print(f"  Models:     {list(selected_models.keys())}")
    print(f"  Dimensions: {args.dim}")
    print(f"  Samples:    {args.samples} from '{args.split}' split")
    print(f"  Output:     {output_dir}")
    print()

    results_file = output_dir / "raw_results.json"

    if not args.analyze_only:
        if api_key == "YOUR_OPENROUTER_API_KEY_HERE":
            print("ERROR: Please set your OpenRouter API key in the script config or via --api-key")
            sys.exit(1)

        samples = collect_samples(args.split, args.samples, args.seed)
        all_results = run_evaluation(
            selected_models, samples, output_dir, args.dim, api_key
        )
    else:
        if not results_file.exists():
            print(f"ERROR: No results file found at {results_file}")
            sys.exit(1)
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {results_file}")

    summary = aggregate_results(all_results, args.dim)
    print_report(summary)
    save_summary(summary, output_dir)


if __name__ == "__main__":
    main()
