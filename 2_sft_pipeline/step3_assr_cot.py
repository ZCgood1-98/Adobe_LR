#!/usr/bin/env python3
"""
step3_assr_cot.py
=================
Generate Chain-of-Thought annotations for the ASSR framework.

Two generation modes:
  - Rule template : zero cost, covers all data (bottom 70% by magnitude)
  - LLM        : high quality, used for the top 30% by magnitude

Per-sample output:
  1. Candidate-value sweep montages for key parameters
  2. ASSR-structured CoT text (Analysis / Sweep & Select / Refine)

Public API (called by run_pipeline.py):
  generate_all_cot(valid_samples, output_cache_path, llm_ratio=0.3)
  → returns {sample_id: {"cot": str, "montage_paths": dict, "method": str}}
"""

import os
import re
import io
import json
import time
import random
import hashlib
import colorsys
import openai
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# ============================================================
# SECTION 0: Global configuration
# ============================================================

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("未找到 API Key，请检查 .env 文件是否配置正确。")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# 换模型只需修改这一行，其余代码无需改动。
# OpenRouter 常用视觉模型 ID：
#   openai/gpt-4o          — GPT-4o 完整版
#   openai/gpt-4o-mini     — GPT-4o mini（当前默认）
#   anthropic/claude-sonnet-4-5  — Claude Sonnet
#   qwen/qwen2.5-vl-72b-instruct — Qwen2.5-VL 开源版
VISION_MODEL = "google/gemini-3.1-flash-image-preview"

SYSTEM_PROMPT_ASSR = (
    "You are a professional photo retouching artist with deep expertise in Lightroom color science. "
    "Describe the retouching reasoning process using the ASSR framework "
    "(Analyze → Sweep & Select → Refine). Be concise and technical."
)

# Key parameters to sweep (most perceptually significant, with corresponding render functions)
SWEEP_PARAMS = [
    # brightness: 曝光/亮度调整（曝光度/高光/阴影/黑白色阶）
    '曝光度', '高光', '阴影', '黑色色阶', '白色色阶',
    # contrast: 对比度
    '对比度',
    # color: 色彩调整（饱和度/自然饱和度/色调/颜色混合器）
    '色温', '自然饱和度', '饱和度', '色调', 'color_mixer',
    # detail: 细节处理（锐化/清晰度/纹理/降噪）
    '锐化', '清晰度', '纹理', '噪点消除',
    # clarity: 通透度/去朦胧
    '去朦胧',
    # style: 风格化（晕影/颗粒）
    '晕影',
]

# 候选图保存尺寸（供人工/模型浏览的完整分辨率）
CANDIDATE_SAVE_SIZE = (960, 720)

# LLM API 调用时拼接 montage 所用的缩略图尺寸（减小 token 消耗）
LLM_THUMB_SIZE = (320, 240)

# Valid range (lo, hi, dtype) for each parameter
PARAM_RANGES = {
    '曝光度':     (-5.0,   5.0,  'float'),
    '对比度':     (-100,   100,  'int'),
    '高光':       (-100,   100,  'int'),
    '阴影':       (-100,   100,  'int'),
    '白色色阶':   (-100,   100,  'int'),
    '黑色色阶':   (-100,   100,  'int'),
    '色温':       (2000,   10000,  'int'),
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

# Visual effect descriptions for too-low / just-right / too-high (used in Select phase)
PARAM_EFFECT = {
    '曝光度':     ("overexposed, losing highlight detail",          "exposure balanced, rich tonal range",              "underexposed, image lacks vitality"),
    '高光':       ("highlights blown out, detail lost",             "highlights natural, retaining detail",             "highlights crushed, image lacks clarity"),
    '阴影':       ("shadows crushed to black, no detail",           "shadow detail and tonal range balanced",           "shadows lifted too much, image looks flat"),
    '黑色色阶':   ("excessive crushed black",                       "black point well balanced",                        "blacks lifted too much"),
    '白色色阶':   ("whites dull, peak brightness clipped low",      "white point well set, highlights clean",           "whites blown out, highlight texture lost"),
    '对比度':     ("contrast too weak, tonal range compressed",     "contrast well-balanced, image has depth",          "contrast too strong, harsh clipped tones"),
    '色温':       ("image noticeably cool",                         "color temperature natural and comfortable",        "image noticeably warm with yellow cast"),
    '色调':       ("green cast visible in neutrals",                "tint neutral, white balance accurate",             "magenta cast visible in neutrals"),
    '自然饱和度': ("colors flat and dull",                          "colors vivid and natural",                         "colors oversaturated and unnatural"),
    '饱和度':     ("colors washed out and lifeless",                "global saturation balanced and pleasing",          "colors garish and oversaturated"),
    '锐化':       ("edges soft and unresolved",                     "sharpening reveals fine detail cleanly",           "over-sharpened, halos and noise amplified"),
    '清晰度':     ("image lacks midtone pop and definition",        "clarity enhances subject presence naturally",      "clarity excessive, looks over-processed"),
    '纹理':       ("fine surface detail feels soft and muted",      "texture reveals natural detail without noise",     "texture excessive, noise becomes distracting"),
    '噪点消除':   ("noise distracting and prominently visible",     "noise smoothed while retaining key detail",        "over-smoothed, fine detail and edges lost"),
    '去朦胧':     ("haze still visible",                            "clarity noticeably improved",                      "dehaze excessive, looks harsh"),
    '晕影':       ("subject lacks prominence",                      "corners darkened, focus on subject",               "vignette too heavy, feels oppressive"),
    'color_mixer': ("hue adjustments too subtle, color imbalance remains",
                    "color mixer corrects hue balance naturally",
                    "color mixer over-applied, unnatural hue shifts appear"),
}


# ============================================================
# SECTION 1: Lightweight parameter renderer (sRGB-native, no Lightroom required)
# ============================================================
#
# 设计原则：整个渲染管线在 float32 sRGB [0,1] 空间内运行。
#
# 背景：数据集图片均为 PNG（已经过 Lightroom tone mapping 的 sRGB 输出），
# 不是 RAW 线性数据。若在线性空间渲染（先 gamma decode，再应用参数，再 encode），
# 大曝光量会把像素值推至远超 1.0，导致高光/阴影的亮度 mask 全部饱和，
# 与负数高光值叠加后乘数变为零，整片亮区变黑 → 严重伪色。
#
# sRGB 原生方案的优点：
#   - 所有 mask 的 lum 输入天然限于 [0,1]，乘法因子永不为负
#   - 无需任何 gamma 往返转换（无精度损失、无伪色）
#   - 对于生成 sweep 对比缩略图来说，感知域的相对差异比物理精度更重要

# ---------- Per-parameter render functions ----------
# 所有函数接收并返回 float32 sRGB [0,1] 数组。
# 中间结果始终 clip 到 [0,1]，避免任何溢出传导。

# 中性色温基准（标准日光 5500 K），与 step2_filter.PARAM_NEUTRAL 保持一致。
TEMP_NEUTRAL = 5500


def apply_exposure(img_srgb: np.ndarray, ev: float) -> np.ndarray:
    """曝光：sRGB 感知空间亮度缩放。
    EV=+1 ≈ 2^(1/2.2)≈1.37× 感知亮度，与线性域的 2× 物理亮度等效感知。
    在 sRGB 空间操作可确保值域始终限于 [0,1]，不会引发后续 mask 溢出。
    """
    return np.clip(img_srgb * (2.0 ** (ev / 2.2)), 0.0, 1.0)


def apply_contrast(img_srgb: np.ndarray, contrast: float) -> np.ndarray:
    """对比度：以 sRGB 感知中灰 0.5 为轴的线性拉伸（对应线性域约 0.214）。"""
    if contrast == 0:
        return img_srgb
    c = contrast * 1.27                                    # [-100,100] → [-127,127]
    factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    return np.clip(factor * (img_srgb - 0.5) + 0.5, 0.0, 1.0)


def apply_shadows_highlights(img_srgb: np.ndarray,
                              shadows: float = 0,
                              highlights: float = 0) -> np.ndarray:
    """阴影/高光：sRGB 亮度分区加法调整。

    分界点为感知中灰 0.5 sRGB：
      - 高光 mask：lum > 0.5，±100 → 该区域 ±0.30 亮度加减
      - 阴影 mask：lum < 0.5，±100 → 该区域 ±0.30 亮度加减
    加法而非乘法：避免 PNG 已压缩高光区域在乘法下产生色相偏移。
    lum 天然限于 [0,1]，mask 永不溢出，彻底消除伪色。
    """
    lum = (0.2126 * img_srgb[:, :, 0]
           + 0.7152 * img_srgb[:, :, 1]
           + 0.0722 * img_srgb[:, :, 2])
    result = img_srgb
    if highlights != 0:
        mask = np.clip((lum - 0.5) / 0.5, 0.0, 1.0)[:, :, np.newaxis]
        result = np.clip(result + mask * (highlights / 100.0) * 0.30, 0.0, 1.0)
    if shadows != 0:
        mask = np.clip((0.5 - lum) / 0.5, 0.0, 1.0)[:, :, np.newaxis]
        result = np.clip(result + mask * (shadows / 100.0) * 0.30, 0.0, 1.0)
    return result


def apply_whites(img_srgb: np.ndarray, whites: float) -> np.ndarray:
    """白色色阶：影响 sRGB lum > 0.75 的高光区域（加法）。"""
    if whites == 0:
        return img_srgb
    lum = (0.2126 * img_srgb[:, :, 0]
           + 0.7152 * img_srgb[:, :, 1]
           + 0.0722 * img_srgb[:, :, 2])
    mask = np.clip((lum - 0.75) / 0.25, 0.0, 1.0)[:, :, np.newaxis]
    return np.clip(img_srgb + mask * (whites / 100.0) * 0.20, 0.0, 1.0)


def apply_blacks(img_srgb: np.ndarray, blacks: float) -> np.ndarray:
    """黑色色阶：影响 sRGB lum < 0.25 的暗部区域（加法）。"""
    if blacks == 0:
        return img_srgb
    lum = (0.2126 * img_srgb[:, :, 0]
           + 0.7152 * img_srgb[:,2

def apply_temperature(img_srgb: np.ndarray, temp: float) -> np.ndarray:
    """色温：sRGB 空间 R/B 通道缩放。
    效果系数 1/500 < 线性域的 1/300，避免 sRGB 已压缩通道被过度放大导致偏色。
    """
    relative = np.clip((temp - TEMP_NEUTRAL) / 50.0, -100.0, 100.0)
    result = img_srgb.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * (1.0 + relative / 500.0), 0.0, 1.0)
    result[:, :, 2] = np.clip(result[:, :, 2] * (1.0 - relative / 500.0), 0.0, 1.0)
    return result


def apply_tint(img_srgb: np.ndarray, tint: float) -> np.ndarray:
    """色调：sRGB 空间绿-品红轴调整。"""
    result = img_srgb.copy()
    shift = tint / 800.0                            # [-150,150] → [-0.19, 0.19]
    result[:, :, 1] = np.clip(result[:, :, 1] * (1.0 - shift), 0.0, 1.0)
    result[:, :, 0] = np.clip(result[:, :, 0] * (1.0 + shift * 0.3), 0.0, 1.0)
    result[:, :, 2] = np.clip(result[:, :, 2] * (1.0 + shift * 0.3), 0.0, 1.0)
    return result


def apply_vibrance(img_srgb: np.ndarray, vibrance: float) -> np.ndarray:
    """自然饱和度：对低饱和区域优先增强，保护已高饱和色调。直接在 sRGB 空间运算。"""
    if vibrance == 0:
        return img_srgb
    r, g, b = img_srgb[:, :, 0], img_srgb[:, :, 1], img_srgb[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-6)
    boost = (vibrance / 100.0) * (1.0 - sat)[:, :, np.newaxis]
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b)[:, :, np.newaxis]
    return np.clip(img_srgb + boost * (img_srgb - lum), 0.0, 1.0)


def apply_saturation(img_srgb: np.ndarray, saturation: float) -> np.ndarray:
    """饱和度：全局饱和度缩放。直接在 sRGB 空间运算。"""
    if saturation == 0:
        return img_srgb
    lum = (0.2126 * img_srgb[:, :, 0]
           + 0.7152 * img_srgb[:, :, 1]
           + 0.0722 * img_srgb[:, :, 2])[:, :, np.newaxis]
    factor = 1.0 + saturation / 100.0
    return np.clip(lum + factor * (img_srgb - lum), 0.0, 1.0)


def apply_dehaze(img_srgb: np.ndarray, dehaze: float) -> np.ndarray:
    """去朦胧：以感知均值为轴增强局部对比度。"""
    if dehaze <= 0:
        return img_srgb
    mean = img_srgb.mean()
    return np.clip(mean + (img_srgb - mean) * (1.0 + dehaze / 150.0), 0.0, 1.0)


def apply_vignette(img_srgb: np.ndarray, vignette: float) -> np.ndarray:
    """晕影：四角压暗，渐变自然。"""
    if vignette >= 0:
        return img_srgb
    H, W = img_srgb.shape[:2]
    xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    dist = np.sqrt(xx ** 2 + yy ** 2)
    mask = np.clip(1.0 - abs(vignette) / 100.0 * dist * 0.6, 0.0, 1.0)[:, :, np.newaxis]
    return img_srgb * mask


def apply_clarity(img_srgb: np.ndarray, clarity: float) -> np.ndarray:
    """清晰度：大半径反锐化蒙版，增强中间调局部对比。直接在 sRGB 空间操作。"""
    if clarity == 0:
        return img_srgb
    blurred = gaussian_filter(img_srgb, sigma=[10.0, 10.0, 0])
    return np.clip(img_srgb + (clarity / 100.0) * 0.6 * (img_srgb - blurred), 0.0, 1.0)


def apply_texture(img_srgb: np.ndarray, texture: float) -> np.ndarray:
    """纹理：小半径反锐化蒙版，增强精细表面细节。直接在 sRGB 空间操作。"""
    if texture == 0:
        return img_srgb
    blurred = gaussian_filter(img_srgb, sigma=[2.0, 2.0, 0])
    return np.clip(img_srgb + (texture / 100.0) * 0.5 * (img_srgb - blurred), 0.0, 1.0)


def apply_sharpening(img_srgb: np.ndarray, sharpening: float) -> np.ndarray:
    """锐化：极小半径反锐化蒙版，增强边缘清晰度。"""
    if sharpening == 0:
        return img_srgb
    blurred = gaussian_filter(img_srgb, sigma=[0.5, 0.5, 0])
    return np.clip(img_srgb + (sharpening / 150.0) * 0.8 * (img_srgb - blurred), 0.0, 1.0)


def apply_noise_reduction(img_srgb: np.ndarray, noise_reduction: float) -> np.ndarray:
    """噪点消除：高斯平滑降噪。"""
    if noise_reduction == 0:
        return img_srgb
    sigma = noise_reduction / 100.0 * 2.0
    return np.clip(gaussian_filter(img_srgb, sigma=[sigma, sigma, 0]), 0.0, 1.0)


# ---------- 颜色混合器（HSL 面板）helper ----------

# Lightroom 8 个固定色相通道（名称 → 中心色相度、高斯半宽）
_MIXER_CHANNELS = {
    'Red':     {'center':   0.0, 'width': 40.0},
    'Orange':  {'center':  30.0, 'width': 30.0},
    'Yellow':  {'center':  60.0, 'width': 35.0},
    'Green':   {'center': 120.0, 'width': 50.0},
    'Aqua':    {'center': 180.0, 'width': 35.0},
    'Blue':    {'center': 225.0, 'width': 40.0},
    'Purple':  {'center': 280.0, 'width': 35.0},
    'Magenta': {'center': 320.0, 'width': 35.0},
}


def _parse_color_style_hue(color_style: str) -> Optional[float]:
    """从 CSS border-color 字符串（如 'border-color: rgb(170, 0, 0); ...'）提取色相（度）。"""
    m = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_style)
    if not m:
        return None
    r, g, b = int(m.group(1)) / 255.0, int(m.group(2)) / 255.0, int(m.group(3)) / 255.0
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0  # 0–360 degrees


def _parse_color_style_to_channel(color_style: str) -> Optional[dict]:
    """
    从 CSS color 字符串解析色相，并匹配到最近的 _MIXER_CHANNELS 通道。
    返回该通道的 {'center': float, 'width': float}，找不到时返回 None。
    """
    hue = _parse_color_style_hue(color_style)
    if hue is None:
        return None
    best_name, best_dist = None, float('inf')
    for name, ch in _MIXER_CHANNELS.items():
        diff = abs(hue - ch['center'])
        diff = min(diff, 360.0 - diff)  # 环形距离
        if diff < best_dist:
            best_dist, best_name = diff, name
    return _MIXER_CHANNELS[best_name]


def _rgb_to_hsl(r: np.ndarray, g: np.ndarray,
                b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sRGB [0,1] → HSL (H in degrees [0,360], S and L in [0,1])."""
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    eps = 1e-7

    L = (max_c + min_c) / 2.0
    S = np.where(delta < eps, 0.0,
                 delta / np.where(L < 0.5,
                                  np.maximum(max_c + min_c, eps),
                                  np.maximum(2.0 - max_c - min_c, eps)))

    mask_r = (delta > eps) & (max_c == r)
    mask_g = (delta > eps) & (max_c == g) & ~mask_r
    mask_b = (delta > eps) & ~mask_r & ~mask_g
    d = np.where(delta > eps, delta, eps)
    H = np.where(mask_r, ((g - b) / d) % 6.0,
         np.where(mask_g, (b - r) / d + 2.0,
         np.where(mask_b, (r - g) / d + 4.0, 0.0)))
    H = (H * 60.0) % 360.0

    return H, S, L


def _hsl_to_rgb(H: np.ndarray, S: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Vectorized HSL → sRGB [0,1]. H in degrees [0,360]."""
    C = (1.0 - np.abs(2.0 * L - 1.0)) * S
    H_prime = H / 60.0
    X = C * (1.0 - np.abs(H_prime % 2.0 - 1.0))
    m = L - C / 2.0
    Z = np.zeros_like(C)
    seg = H_prime.astype(int) % 6
    R = np.select([seg==0, seg==1, seg==2, seg==3, seg==4, seg==5], [C,X,Z,Z,X,C]) + m
    G = np.select([seg==0, seg==1, seg==2, seg==3, seg==4, seg==5], [X,C,C,X,Z,Z]) + m
    B = np.select([seg==0, seg==1, seg==2, seg==3, seg==4, seg==5], [Z,Z,X,C,C,X]) + m
    return np.stack([R, G, B], axis=2)


def apply_color_mixer(img_srgb: np.ndarray, color_mixer: list) -> np.ndarray:
    """
    Lightroom 颜色混合器（HSL 面板）：对 8 个固定色相通道分别做 H/S/L 微调。

    算法：
      1. 向量化 RGB → HSL（输入已是 sRGB，无需转换）
      2. 对每个色相通道用高斯软遮罩加权施加 H/S/L 偏移
         mask = exp(-0.5 * (dist / sigma)^2)，sigma = width / 2
      3. HSL → RGB（直接输出 sRGB）

    参数缩放（近似 Lightroom 行为）：
      H ±100 → ±30° 色相偏移
      S ±100 → 饱和度变化，低饱和度区域影响更强：
             weight * (s_delta/100) * (0.5 + S_pixel * 0.5)
      L ±100 → ±0.5 明度偏移（clamp 到 [0,1]）

    color_mixer 格式（来自 step1_parse_params.extract_params）：
      [{'color': 'border-color: rgb(r,g,b); ...', 'H': float, 'S': float, 'L': float}, ...]
    """
    if not color_mixer:
        return img_srgb

    H, S, L = _rgb_to_hsl(img_srgb[:, :, 0], img_srgb[:, :, 1], img_srgb[:, :, 2])
    H_adj, S_adj, L_adj = H.copy(), S.copy(), L.copy()

    # 饱和度保护权重：S < 0.10 的近中性像素 H 值无意义，必须将其排除在色相调整之外。
    # 若不加此保护，灰色像素会被分配 H=0（Red 通道中心），导致被 Red 通道的 S 调整
    # 大幅拉偏（灰色区域变红），出现严重伪色。
    sat_guard = np.clip(S / 0.10, 0.0, 1.0)   # S=0 → weight=0，S≥0.10 → weight=1

    for entry in color_mixer:
        h_delta = float(entry.get('H') or 0)
        s_delta = float(entry.get('S') or 0)
        l_delta = float(entry.get('L') or 0)
        if h_delta == 0 and s_delta == 0 and l_delta == 0:
            continue

        channel = _parse_color_style_to_channel(entry.get('color', ''))
        if channel is None:
            continue

        center = channel['center']
        sigma  = channel['width'] / 2.0   # 高斯标准差

        # 环形色相距离 → 高斯软遮罩
        dist = np.abs(H - center)
        dist = np.minimum(dist, 360.0 - dist)
        hue_mask = np.exp(-0.5 * (dist / sigma) ** 2)

        # 综合遮罩 = 色相软遮罩 × 饱和度保护
        # 确保近中性（无意义色相）像素不被任何通道的 H/S/L 调整影响
        effective_mask = hue_mask * sat_guard

        H_adj = (H_adj + effective_mask * h_delta * 0.30) % 360.0   # ±100 → ±30°

        # S/L 比例公式：正向向剩余空间推进，负向向 0 缩退。
        # 比平坦相加更接近 Lightroom 行为：高饱和像素不会在第一档就撞顶/撞底，
        # 保证 ×1.5/×2.0 候选与 ×1.0 在视觉上有可区分的差异。
        s_shift = effective_mask * (s_delta / 100.0)
        S_adj = np.where(s_shift >= 0,
                         np.clip(S_adj + s_shift * (1.0 - S_adj), 0.0, 1.0),
                         np.clip(S_adj + s_shift * S_adj,         0.0, 1.0))

        l_shift = effective_mask * (l_delta / 200.0)
        L_adj = np.where(l_shift >= 0,
                         np.clip(L_adj + l_shift * (1.0 - L_adj), 0.0, 1.0),
                         np.clip(L_adj + l_shift * L_adj,         0.0, 1.0))

    return np.clip(_hsl_to_rgb(H_adj, S_adj, L_adj), 0.0, 1.0)


# 渲染函数映射表（所有函数接收并返回 float32 sRGB [0,1]）
RENDER_FUNCTIONS = {
    '曝光度':     lambda arr, v: apply_exposure(arr, v),
    '对比度':     lambda arr, v: apply_contrast(arr, v),
    '高光':       lambda arr, v: apply_shadows_highlights(arr, highlights=v),
    '阴影':       lambda arr, v: apply_shadows_highlights(arr, shadows=v),
    '白色色阶':   lambda arr, v: apply_whites(arr, v),
    '黑色色阶':   lambda arr, v: apply_blacks(arr, v),
    '色温':       lambda arr, v: apply_temperature(arr, v),
    '色调':       lambda arr, v: apply_tint(arr, v),
    '自然饱和度': lambda arr, v: apply_vibrance(arr, v),
    '饱和度':     lambda arr, v: apply_saturation(arr, v),
    '清晰度':     lambda arr, v: apply_clarity(arr, v),
    '纹理':       lambda arr, v: apply_texture(arr, v),
    '锐化':       lambda arr, v: apply_sharpening(arr, v),
    '噪点消除':   lambda arr, v: apply_noise_reduction(arr, v),
    '去朦胧':     lambda arr, v: apply_dehaze(arr, v),
    '晕影':       lambda arr, v: apply_vignette(arr, v),
    'color_mixer': lambda arr, v: apply_color_mixer(arr, v),
}

# 五个渲染阶段（按 Lightroom 真实工作流顺序）
# 扫描某阶段的参数时，以前若干阶段 GT 值的累积渲染结果作为基底
STAGE_PARAM_GROUPS = [
    ['曝光度', '白色色阶', '黑色色阶', '高光', '阴影', '对比度'],  # Stage 1: 基础曝光
    ['色温', '色调'],                                                  # Stage 2: 白平衡
    ['自然饱和度', '饱和度', 'color_mixer'],                             # Stage 3: HSL/色彩混合
    ['晕影'],                                                          # Stage 4: 效果/风格化
    ['纹理', '清晰度', '去朦胧', '锐化', '噪点消除'],                  # Stage 5: 细节/锐化/降噪
]

# 每个参数属于哪个阶段（0-indexed）
PARAM_STAGE = {
    param: stage_idx
    for stage_idx, group in enumerate(STAGE_PARAM_GROUPS)
    for param in group
}


def render_stage(img_srgb: np.ndarray, params: dict, stage_labels: list) -> np.ndarray:
    """在 float32 sRGB [0,1] 空间内依次应用一个阶段所有可渲染参数的 GT 值。
    标量参数强转 float；list 参数（color_mixer）直接传入，由各自的渲染函数处理。
    """
    result = img_srgb
    for label in stage_labels:
        if label not in params:
            continue
        render_fn = RENDER_FUNCTIONS.get(label)
        if render_fn is None:
            continue
        val = params[label]
        result = render_fn(result, val if isinstance(val, list) else float(val))
    return result


def _resize_proportional(img: Image.Image, max_size: tuple) -> Image.Image:
    """等比例缩放：保持宽高比，使图片适配 max_size，不裁剪、不拉伸"""
    w, h = img.size
    scale = min(max_size[0] / w, max_size[1] / h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def render_cumulative_base(original_path: str,
                            params: dict,
                            up_to_stage: int,
                            output_size: tuple = (300, 225)) -> np.ndarray:
    """
    Pipeline 入口：将原图读取为 float32 sRGB [0,1] 数组（不做 gamma 解码），
    然后依次应用第 0..up_to_stage-1 阶段的 GT 参数。
    返回 float32 sRGB [0,1]，作为第 up_to_stage 阶段扫描的基底。
    图片按原始宽高比等比例缩放至 output_size 内，不拉伸。
    """
    img = _resize_proportional(Image.open(original_path).convert('RGB'), output_size)
    arr_srgb = np.array(img).astype(np.float32) / 255.0   # sRGB [0,1]，不做 gamma 解码
    for stage_idx in range(up_to_stage):
        arr_srgb = render_stage(arr_srgb, params, STAGE_PARAM_GROUPS[stage_idx])
    return arr_srgb


def render_single_param(original_path: str,
                         param_label: str,
                         param_value: float,
                         output_size: tuple = (400, 300)) -> Image.Image:
    """将单个参数应用到原图，返回渲染后的 PIL Image。"""
    img = _resize_proportional(Image.open(original_path).convert('RGB'), output_size)
    render_fn = RENDER_FUNCTIONS.get(param_label)
    if render_fn is None:
        return img
    arr_srgb = np.array(img).astype(np.float32) / 255.0
    result = render_fn(arr_srgb, float(param_value))
    return Image.fromarray((np.clip(result, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _render_full_image(original_path: str,
                       params: dict,
                       output_size: tuple = (400, 300)) -> Image.Image:
    """将所有参数按 Lightroom 工作流顺序应用到原图，返回完整渲染结果。"""
    arr = render_cumulative_base(original_path, params, len(STAGE_PARAM_GROUPS), output_size)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _generate_distorted_params(gt_params: dict, rng: random.Random) -> dict:
    """
    生成一套扰动参数：对 SWEEP_PARAMS 中的每个参数随机加减 15~30% 的参数范围。

    边界安全机制：截断后若扰动值与 GT 的绝对差不足 min_diff（参数范围的 10%），
    则翻转符号重试；若两侧均不足（GT 极度靠近边界），强制取离 GT 最远的端点，
    确保视觉差异始终可辨。
    """
    distorted = dict(gt_params)
    for label in SWEEP_PARAMS:
        if label == 'color_mixer':
            continue  # list 类型，单独处理
        if label not in gt_params:
            continue
        lo, hi, dtype = PARAM_RANGES[label]
        span     = hi - lo
        gt       = float(gt_params[label])
        min_diff = span * 0.10   # 最低差异阈值：参数范围的 10%

        magnitude = rng.uniform(0.15, 0.30)
        sign      = rng.choice([-1, 1])

        new_val = max(lo, min(hi, gt + span * magnitude * sign))

        # 截断后差异不足 → 翻转方向重试
        if abs(new_val - gt) < min_diff:
            new_val = max(lo, min(hi, gt + span * magnitude * (-sign)))

        # 两侧均不足（GT 极度靠近某边界）→ 取离 GT 最远的端点
        if abs(new_val - gt) < min_diff:
            new_val = lo if (gt - lo) >= (hi - gt) else hi

        distorted[label] = int(round(new_val)) if dtype == 'int' else round(new_val, 2)

    # color_mixer 单独扰动：将所有通道 H/S/L 乘以随机缩放系数（±30~70% 偏离 GT）
    if 'color_mixer' in gt_params and isinstance(gt_params['color_mixer'], list):
        scale = rng.uniform(0.3, 0.7) if rng.random() < 0.5 else rng.uniform(1.3, 1.8)
        distorted['color_mixer'] = [
            {
                'color': e.get('color', ''),
                'H': round(max(-100.0, min(100.0, (e.get('H') or 0) * scale)), 1),
                'S': round(max(-100.0, min(100.0, (e.get('S') or 0) * scale)), 1),
                'L': round(max(-100.0, min(100.0, (e.get('L') or 0) * scale)), 1),
            }
            for e in gt_params['color_mixer']
        ]

    return distorted


# ============================================================
# SECTION 2: 候选值生成与网格图（Montage）
# ============================================================

def generate_sweep_candidates(label: str,
                               gt_value: float,
                               n_candidates: int = 5) -> tuple[list, int]:
    lo, hi, dtype = PARAM_RANGES.get(label, (-100, 100, 'int'))
    gt = float(gt_value)

    # 色温单独处理：步长固定 250K，不按范围比例算
    if label == '色温':
        step = 250
    else:
        span = hi - lo
        step = span * 0.08 if dtype == 'float' else max(1, int(span * 0.12))

    half = n_candidates // 2
    raw = [gt + i * step for i in range(-half, half + 1)]

    seen, candidates = set(), []
    for v in raw:
        clamped = max(lo, min(hi, v))
        if dtype == 'float':
            val = round(clamped, 2)
        else:
            val = int(round(clamped))
        if val not in seen:
            seen.add(val)
            candidates.append(val)

    gt_key = round(gt, 2) if dtype == 'float' else int(round(gt))
    if gt_key not in candidates:
        candidates.append(gt_key)
        candidates.sort()

    candidates = candidates[:n_candidates]
    random.shuffle(candidates)
    return candidates, candidates.index(gt_key)


def describe_select_reason(label: str,
                            gt_value: float,
                            candidates: list) -> str:
    """生成 Select 阶段的选择理由文字"""
    _, _, dtype = PARAM_RANGES.get(label, (-100, 100, 'int'))
    gt_key = round(float(gt_value), 2) if dtype == 'float' else int(round(float(gt_value)))
    too_low, just_right, too_high = PARAM_EFFECT.get(label, ("效果不足", "效果适中", "效果过度"))

    sorted_c = sorted(candidates)
    rank = sorted_c.index(gt_key)

    parts = []
    if rank > 0:
        parts.append(f"At {sorted_c[rank-1]}: {too_low}. ")
    if rank < len(sorted_c) - 1:
        parts.append(f"At {sorted_c[rank+1]}: {too_high}. ")
    parts.append(f"Selecting {gt_key}: {just_right}.")
    return "".join(parts)


def _add_value_label(img: Image.Image,
                     param_label: str,
                     value: float) -> Image.Image:
    """在图片底部添加参数值文字标签"""
    label_h = 32
    new_img = Image.new('RGB', (img.width, img.height + label_h), (20, 20, 20))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    if param_label == '曝光度':
        display = f"{value:+.2f}"
    else:
        v = int(round(float(value)))
        display = f"{v:+d}" if v != 0 else "0"

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), display, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(
        ((img.width - text_w) // 2, img.height + 6),
        display, fill=(255, 255, 255), font=font
    )
    return new_img


def save_candidate_thumbs(base_img_srgb: np.ndarray,
                           param_label: str,
                           candidates: list,
                           gt_value: float,
                           save_dir: Path) -> list[str]:
    """
    为每个候选值单独渲染并保存到 save_dir/{idx:02d}_{value}.jpg。
    candidates 传入时已随机打乱（由 generate_sweep_candidates 负责），
    此处保持该顺序，GT 不固定在中间位置，增加多样性。
    GT 对应的图片加绿色边框。
    返回与 candidates 同序的图片路径列表。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    _, _, dtype = PARAM_RANGES.get(param_label, (-100, 100, 'int'))
    gt_key = round(float(gt_value), 2) if dtype == 'float' else int(round(float(gt_value)))

    render_fn = RENDER_FUNCTIONS.get(param_label)

    paths = []
    for idx, val in enumerate(candidates):
        rendered = render_fn(base_img_srgb, float(val)) if render_fn is not None else base_img_srgb
        thumb = Image.fromarray((np.clip(rendered, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        labeled = _add_value_label(thumb, param_label, val)

        if val == gt_key:
            ImageDraw.Draw(labeled).rectangle(
                [0, 0, labeled.width - 1, labeled.height - 1],
                outline=(0, 220, 80), width=4
            )

        val_str = f"{val:+.2f}" if dtype == 'float' else f"{int(val):+d}"
        fpath = save_dir / f"{idx:02d}_{val_str}.jpg"
        labeled.save(str(fpath), quality=85)
        paths.append(str(fpath))

    return paths


def generate_sweep_candidates_color_mixer(
        gt_color_mixer: list, n_candidates: int = 5) -> tuple[list, list, int]:
    """
    为 color_mixer 生成 n_candidates 个候选强度等级。

    通过将所有通道的 H/S/L 值乘以缩放系数来模拟不同调整强度。

    自适应区间策略：
      - 普通参数（所有通道 abs(S) < 70）：以 GT 为中心，[0.0, 0.5, 1.0, 1.5, 2.0]
        → 5 张图覆盖欠矫到过矫的完整范围
      - 极端参数（任意通道 abs(S) ≥ 70）：收缩到 [0.0, 0.25, 0.5, 0.75, 1.0]
        → 1.5×/2.0× 对 S=±100 无增量效果，改为展示从无到 GT 的渐变

    返回：
      (candidates, scales, gt_idx)
        candidates : list of color_mixer configs (list of dicts)，已随机打乱
        scales     : list of float scale factors（与 candidates 一一对应，用于显示标签）
        gt_idx     : GT (scale=1.0) 在打乱后 candidates 列表中的新索引
    """
    max_abs_s = max((abs(float(e.get('S') or 0)) for e in gt_color_mixer), default=0.0)
    if max_abs_s >= 70.0:
        # 极端参数：[0, 0.25, 0.5, 0.75, 1.0]
        steps = n_candidates - 1 or 1
        scales = [round(i / steps, 2) for i in range(n_candidates)]
    else:
        # 普通参数：[0.0, 0.5, 1.0, 1.5, 2.0]
        half = n_candidates // 2
        step = 1.0 / half if half > 0 else 1.0
        scales = [round(max(0.0, 1.0 + i * step), 2) for i in range(-half, half + 1)]

    def _scale_entry(entry: dict, factor: float) -> dict:
        return {
            'color': entry.get('color', ''),
            'H': round(max(-100.0, min(100.0, (entry.get('H') or 0) * factor)), 1),
            'S': round(max(-100.0, min(100.0, (entry.get('S') or 0) * factor)), 1),
            'L': round(max(-100.0, min(100.0, (entry.get('L') or 0) * factor)), 1),
        }

    candidates = [[_scale_entry(e, s) for e in gt_color_mixer] for s in scales]

    # 打乱顺序，GT 不固定在中间位置
    paired = list(zip(candidates, scales))
    random.shuffle(paired)
    candidates, scales = zip(*paired)
    candidates, scales = list(candidates), list(scales)
    gt_idx = scales.index(1.0)   # 找到 GT (scale=1.0) 的新位置

    return candidates, scales, gt_idx


def save_candidate_thumbs_color_mixer(base_img_srgb: np.ndarray,
                                       candidates: list,
                                       scales: list,
                                       gt_idx: int,
                                       save_dir: Path) -> list[str]:
    """
    为每个 color_mixer 候选强度等级渲染并保存到 save_dir。
    candidates/scales 已由 generate_sweep_candidates_color_mixer 随机打乱，GT 不固定在中间。
    文件名格式：{idx:02d}_x{scale:.1f}.jpg。
    GT 对应的图片加绿色边框。
    返回与 candidates 同序的路径列表。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    label_h = 32
    paths = []

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()

    for idx, (mixer, scale) in enumerate(zip(candidates, scales)):
        rendered = apply_color_mixer(base_img_srgb, mixer)
        thumb = Image.fromarray((np.clip(rendered, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))

        # 底部加强度标签
        labeled = Image.new('RGB', (thumb.width, thumb.height + label_h), (20, 20, 20))
        labeled.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(labeled)
        display = f"\u00d7{scale:.1f}"   # e.g. "×1.0"
        bbox = draw.textbbox((0, 0), display, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(((thumb.width - text_w) // 2, thumb.height + 6),
                  display, fill=(255, 255, 255), font=font)

        # GT 加绿色边框
        if idx == gt_idx:
            ImageDraw.Draw(labeled).rectangle(
                [0, 0, labeled.width - 1, labeled.height - 1],
                outline=(0, 220, 80), width=4
            )

        fpath = save_dir / f"{idx:02d}_x{scale:.1f}.jpg"
        labeled.save(str(fpath), quality=85)
        paths.append(str(fpath))

    return paths


def _stitch_for_llm(image_paths: list[str]) -> Image.Image:
    """
    将已保存的大图路径缩小至 LLM_THUMB_SIZE 后横向拼接，仅供 LLM API 调用。
    保存到磁盘的是完整分辨率（CANDIDATE_SAVE_SIZE）；
    发给 LLM 的 montage 缩小以减少 token 消耗，不额外落盘。
    """
    imgs = [_resize_proportional(Image.open(p), LLM_THUMB_SIZE) for p in image_paths]
    total_w = sum(img.width for img in imgs)
    total_h = max(img.height for img in imgs)
    montage = Image.new('RGB', (total_w, total_h), (30, 30, 30))
    x = 0
    for img in imgs:
        montage.paste(img, (x, 0))
        x += img.width
    return montage


# ============================================================
# SECTION 3: CoT 文字生成（规则模板）
# ============================================================

# ── 同义词池 ──────────────────────────────────────────────────
# 每次从中随机抽取，使文本形态足够多样，迫使模型关注视觉特征而非文本规律。

_W_TOO_LOW: list[str] = [
    "insufficient",
    "lacks impact",
    "fails to address the issue",
    "too subtle to make a difference",
    "undercooked",
    "leaves the problem unresolved",
    "doesn't go far enough",
    "barely perceptible",
    "underwhelming",
    "underdone",
    "misses the mark",
    "too timid",
    "not enough to correct the issue",
    "falls short",
    "achieves little at this level",
    "hardly noticeable",
    "ineffective at this value",
    "too conservative",
    "a negligible improvement",
    "leaves the image looking largely untouched",
]

_W_TOO_HIGH: list[str] = [
    "excessive",
    "overdone",
    "pushes too far",
    "creates a distracting artifact",
    "goes beyond what the image needs",
    "damages the tonal balance",
    "looks unnatural",
    "too aggressive",
    "destroys fine detail",
    "overcooked",
    "harsh and unflattering",
    "loses the original character of the scene",
    "feels heavily processed",
    "introduces clipping",
    "overshoots the intended target",
    "takes the adjustment too far",
    "causes unwanted side effects",
    "looks artificial",
    "compromises overall image quality",
    "sacrifices realism for drama",
]

_W_JUST_RIGHT: list[str] = [
    "strikes the right balance",
    "achieves the intended look",
    "corrects the issue without overcorrecting",
    "feels natural and deliberate",
    "lands in the sweet spot",
    "exactly what the image calls for",
    "subtle yet effective",
    "completes the edit cleanly",
    "looks intentional and polished",
    "hits the mark",
    "achieves the desired visual effect",
    "brings the image to life",
    "produces the most convincing result",
    "well-calibrated for this scene",
    "reads as intentional and controlled",
    "resolves the issue elegantly",
    "yields the most pleasing result",
    "the image responds well at this value",
    "feels consistent with the overall edit direction",
    "the ideal compromise between the extremes",
]

_W_FINAL_SEL: list[str] = [
    "Final selection: {val}.",
    "Settled on {val}.",
    "Chose {val} as the final value.",
    "{val} selected.",
    "Going with {val}.",
    "The retoucher landed on {val}.",
    "{val} was chosen.",
    "Locked in at {val}.",
    "The edit lands at {val}.",
    "Ended up at {val}.",
]


def _rw_low()      -> str: return random.choice(_W_TOO_LOW)
def _rw_high()     -> str: return random.choice(_W_TOO_HIGH)
def _rw_ok()       -> str: return random.choice(_W_JUST_RIGHT)
def _rw_sel(val)   -> str: return random.choice(_W_FINAL_SEL).format(val=val)


# ── Analysis 诊断语句变体池 ───────────────────────────────────

_A_EXP_UNDER = [
    "Overall exposure appears underexposed by ~{ev} EV.",
    "The image is ~{ev} EV underexposed and needs brightening.",
    "Base exposure is low (~{ev} EV); overall brightness needs lifting.",
    "Shot roughly {ev} EV short of proper exposure.",
    "Underexposure of ~{ev} EV makes the image feel heavy and dark.",
]
_A_EXP_OVER = [
    "Overall exposure appears overexposed by ~{ev} EV.",
    "The image runs ~{ev} EV too bright; highlights need reining in.",
    "Base exposure runs hot (~{ev} EV above target).",
    "Roughly {ev} EV overexposed; the tonal range needs compressing.",
    "Overexposure (~{ev} EV) risks washing out highlight detail.",
]
_A_HL_CLIP = [
    "Highlights at risk of clipping; detail needs to be pulled back.",
    "Highlight recovery is necessary — clipping is imminent.",
    "The bright areas are blown; highlights need pulling down.",
    "Highlight detail is at risk of being lost.",
]
_A_SH_HEAVY = [
    "Insufficient shadow detail; shadows are too heavy.",
    "Shadow areas lack detail and feel crushed.",
    "The dark tones are blocking up and need lifting.",
    "Shadows are underexposed; detail recovery needed.",
]
_A_BK_CRUSHED = [
    "Crushed blacks present; blacks need lifting to recover detail.",
    "The black point is too low; shadow detail is lost.",
    "Blacks are clipped; the base needs raising.",
    "Lifting blacks will help recover the lowest tonal range.",
]
_A_COOL_STRONG = [
    "Image is noticeably cool; lacks warmth.",
    "Color temperature skews cold, giving the image a clinical feel.",
    "The scene reads too cool — a warmer cast would feel more natural.",
    "Noticeable blue-green cast; white balance needs warming.",
]
_A_COOL_SLIGHT = [
    "Image is slightly cool.",
    "A mild cool cast is present.",
    "Color temperature is a touch on the cold side.",
    "Slight blue bias in the white balance.",
]
_A_WARM_STRONG = [
    "Image is noticeably warm with a yellow cast.",
    "Excessive warmth makes the scene look unnatural.",
    "Strong orange-yellow cast; white balance needs cooling.",
    "The image skews very warm and needs correction.",
]
_A_WARM_SLIGHT = [
    "Image is slightly warm.",
    "A mild warm cast is present.",
    "Color temperature leans a bit warm.",
    "Slight orange bias in the white balance.",
]
_A_VIB_FLAT = [
    "Colors appear flat and lack vibrancy.",
    "The palette looks desaturated; colors need a boost.",
    "Vibrance is low — the image lacks color energy.",
    "Colors feel muted and subdued.",
]
_A_VIB_HOT = [
    "Overall saturation is too high; needs to be reined in.",
    "Colors are oversaturated and look unnatural.",
    "The palette is overly vivid — pulling saturation back will help.",
    "Vibrance is excessive; tones need to breathe.",
]
_A_HAZE = [
    "Image appears hazy; clarity is insufficient.",
    "Atmospheric haze reduces overall contrast and detail.",
    "The scene lacks crisp clarity — dehaze will help.",
    "A veil of haze softens the image more than intended.",
]
_A_VIG = [
    "Corners are too bright; subject lacks prominence.",
    "The frame needs vignetting to draw the eye inward.",
    "Without a vignette, the corners compete with the subject.",
    "Bright edges distract from the central subject.",
]
_A_OK = [
    "Base image is in good condition; applying refined adjustments.",
    "Exposure and color are well-balanced; fine-tuning from here.",
    "The image is technically sound; the edit focuses on polish.",
    "No major issues; adjustments will refine rather than correct.",
]


def _pick(pool: list[str], **kw) -> str:
    """Select a random entry from pool, formatting with kw if provided."""
    return random.choice(pool).format(**kw) if kw else random.choice(pool)


# ── Refine 耦合描述变体池 ─────────────────────────────────────

_R_SH_BK = [
    "Shadows +{sh} and Blacks +{bk} both lifted; verify the image doesn't look too flat.",
    "With both Shadows (+{sh}) and Blacks (+{bk}) raised, overall contrast may need checking.",
    "Lifting shadows (+{sh}) and blacks (+{bk}) together risks a flat midtone response.",
    "The dual lift of Shadows +{sh} / Blacks +{bk} softens the dark end — watch for flatness.",
]
_R_DH_VIB = [
    "Dehaze +{dh} and Vibrance +{vib} combined; watch that colors don't become oversaturated.",
    "With Dehaze +{dh} and Vibrance +{vib} both active, saturation can build up quickly.",
    "Dehaze (+{dh}) boosts local contrast and color simultaneously with Vibrance (+{vib}); keep an eye on saturation.",
    "The pairing of Dehaze +{dh} and Vibrance +{vib} risks oversaturation in already-vivid hues.",
]
_R_EXP_HL = [
    "Exposure +{exp:.2f} lifts overall brightness; Highlights {hl:.0f} pulled back to protect detail.",
    "Raising Exposure (+{exp:.2f}) and recovering Highlights ({hl:.0f}) is a classic tonal rescue.",
    "Exposure +{exp:.2f} brightens the scene while Highlights {hl:.0f} guards against clipping.",
    "The push-pull of Exposure +{exp:.2f} and Highlights {hl:.0f} widens the tonal range cleanly.",
]
_R_OK = [
    "Parameter combination is well-balanced; no further correction needed.",
    "No significant coupling interactions to address.",
    "The adjustments work together cleanly — no compensation required.",
    "Parameters are mutually consistent; the edit holds together.",
]


def _build_analysis(params: dict) -> list[str]:
    """Analysis phase: diagnose the main visual issues in the image."""
    lines = []
    exp  = params.get('曝光度', 0)
    hl   = params.get('高光', 0)
    sh   = params.get('阴影', 0)
    bk   = params.get('黑色色阶', 0)
    temp = params.get('色温', 0)
    vib  = params.get('自然饱和度', 0)
    dh   = params.get('去朦胧', 0)
    vig  = params.get('晕影', 0)

    if abs(exp) > 0.2:
        pool = _A_EXP_UNDER if exp > 0 else _A_EXP_OVER
        lines.append(_pick(pool, ev=f"{abs(exp):.1f}"))
    if hl < -20:
        lines.append(_pick(_A_HL_CLIP))
    if sh > 20:
        lines.append(_pick(_A_SH_HEAVY))
    if bk > 30:
        lines.append(_pick(_A_BK_CRUSHED))
    deviation = temp - TEMP_NEUTRAL
    if deviation < -800:
        lines.append(_pick(_A_COOL_STRONG))
    elif deviation < -300:
        lines.append(_pick(_A_COOL_SLIGHT))
    elif deviation > 800:
        lines.append(_pick(_A_WARM_STRONG))
    elif deviation > 300:
        lines.append(_pick(_A_WARM_SLIGHT))
    if vib > 10:
        lines.append(_pick(_A_VIB_FLAT))
    elif vib < -10:
        lines.append(_pick(_A_VIB_HOT))
    if dh > 20:
        lines.append(_pick(_A_HAZE))
    if vig < -20:
        lines.append(_pick(_A_VIG))
    if not lines:
        lines.append(_pick(_A_OK))
    return lines


def _build_refine(params: dict) -> list[str]:
    """Refine phase: correct parameter coupling interactions."""
    lines = []
    sh  = params.get('阴影', 0)
    bk  = params.get('黑色色阶', 0)
    dh  = params.get('去朦胧', 0)
    vib = params.get('自然饱和度', 0)
    hl  = params.get('高光', 0)
    exp = params.get('曝光度', 0)

    if sh > 20 and bk > 20:
        lines.append(_pick(_R_SH_BK, sh=int(sh), bk=int(bk)))
    if dh > 30 and vib > 10:
        lines.append(_pick(_R_DH_VIB, dh=int(dh), vib=int(vib)))
    if hl < -20 and exp > 0.3:
        lines.append(_pick(_R_EXP_HL, exp=exp, hl=hl))
    if not lines:
        lines.append(_pick(_R_OK))
    return lines


def describe_select_reason(label: str,
                            gt_value: float,
                            candidates: list) -> str:
    """
    Generate a randomized Select-phase justification sentence.

    Five sentence structure styles are chosen at random:
      0 — Standard: reject neighbors → select GT
      1 — Lead with selection, then justify
      2 — Only explain one extreme (randomly high or low)
      3 — Narrative / story form
      4 — Minimal: just the conclusion
    """
    _, _, dtype = PARAM_RANGES.get(label, (-100, 100, 'int'))
    gt_key   = round(float(gt_value), 2) if dtype == 'float' else int(round(float(gt_value)))
    sorted_c = sorted(candidates)
    rank     = sorted_c.index(gt_key)
    has_low  = rank > 0
    has_high = rank < len(sorted_c) - 1
    low_val  = sorted_c[rank - 1] if has_low  else None
    high_val = sorted_c[rank + 1] if has_high else None

    style = random.randint(0, 4)

    if style == 0 or (not has_low and not has_high):
        # Standard: reject neighbors → final selection
        parts = []
        if has_low:
            parts.append(f"At {low_val}: {_rw_low()}.")
        if has_high:
            parts.append(f"At {high_val}: {_rw_high()}.")
        parts.append(_rw_sel(gt_key))
        return " ".join(parts)

    elif style == 1:
        # Lead with selection, then justify
        sel  = _rw_sel(gt_key)
        just = []
        if has_low:
            just.append(f"{low_val} was {_rw_low()}")
        if has_high:
            just.append(f"{high_val} {_rw_high()}")
        if just:
            return f"{sel} {_rw_ok()}. {'; '.join(just).capitalize()}."
        return f"{sel} {_rw_ok()}."

    elif style == 2:
        # Only explain one extreme — randomly pick which
        if has_low and has_high:
            if random.random() < 0.5:
                return f"Values like {high_val} are {_rw_high()}. {_rw_sel(gt_key)}"
            else:
                return f"{low_val} proved {_rw_low()}, making {gt_key} the better pick."
        elif has_high:
            return f"Going higher to {high_val} would be {_rw_high()}. {_rw_sel(gt_key)}"
        else:
            return f"Dropping to {low_val} left things {_rw_low()}. {_rw_sel(gt_key)}"

    elif style == 3:
        # Narrative / story form
        if has_high and has_low:
            return (f"Pushing to {high_val} {_rw_high()}; "
                    f"pulling back to {low_val} {_rw_low()}. "
                    f"{gt_key} {_rw_ok()}.")
        elif has_high:
            return f"{gt_key} {_rw_ok()} — {high_val} would have been {_rw_high()}."
        else:
            return f"{gt_key} {_rw_ok()} — {low_val} was {_rw_low()}."

    else:  # style == 4
        # Minimal: just the conclusion
        return f"{gt_key}: {_rw_ok()}."


def _build_rule_cot_text(params: dict, montage_paths: dict) -> str:
    """
    Generate a complete ASSR CoT text using the rule template.

    Randomisation applied:
      - All diagnostic / justification phrases drawn from synonym pools.
      - Sweep sentence structure varies across 5 styles per parameter.
      - Each parameter's Sweep entry is independently dropped with p=0.30.
      - The entire Refine section is dropped with p=0.30.
    """
    sections = []

    # Analysis
    sections.append("## Analysis\n" + "".join(_build_analysis(params)))

    # Sweep & Select — with per-parameter 30 % dropout
    sweep_lines = []
    for label in SWEEP_PARAMS:
        if label not in params or label not in montage_paths:
            continue
        if random.random() < 0.30:          # 30 % chance: silently skip this parameter
            continue

        # color_mixer 是 list 类型，特殊处理
        if label == 'color_mixer':
            too_low, just_right, too_high = PARAM_EFFECT['color_mixer']
            sweep_lines.append(
                f"**color_mixer** (see sweep montage):\n"
                f"Swept color mixer at scales \u00d70.0, \u00d70.5, \u00d71.0, \u00d71.5, \u00d72.0. "
                f"At \u00d70.5: {too_low}. "
                f"At \u00d71.5: {too_high}. "
                f"Selecting \u00d71.0 (recorded Lightroom values): {just_right}."
            )
            continue

        gt_value = params[label]
        if abs(float(gt_value)) < 5:
            continue
        candidates, _ = generate_sweep_candidates(label, gt_value)
        reason = describe_select_reason(label, gt_value, candidates)
        sweep_lines.append(
            f"**{label}** (see sweep montage):\n"
            f"Swept candidates {candidates}. {reason}"
        )

    if sweep_lines:
        sections.append("## Sweep & Select\n" + "\n".join(sweep_lines))

    # Refine — 30 % chance to omit the entire section
    if random.random() >= 0.30:
        sections.append("## Refine\n" + "".join(_build_refine(params)))

    return "\n\n".join(sections)


# ============================================================
# SECTION 4: LLM CoT 生成
# ============================================================

def _generate_llm_cot(original_path: str,
                         params: dict,
                         montage_paths: dict,
                         sample_id: str = "",
                         max_retries: int = 3) -> tuple[Optional[str], bool]:
    """
    向 LLM 出示 3 套参数（1 套 GT + 2 套随机扰动），让它通过对比原图和渲染结果
    判断哪套最合理，再为选出的那套写 ASSR CoT。

    返回值：(cot_text, is_hard_sample)
      - (str,  False) — LLM 成功识别 GT，返回 CoT 文本
      - (None, True)  — LLM 选错 → 难样本，调用方应丢弃该条数据
      - (None, False) — API 调用失败 → 调用方回退到规则模板
    """
    import base64
    import re

    _seed = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(_seed)

    # 生成 2 套扰动参数，并渲染全图
    distorted_a = _generate_distorted_params(params, rng)
    distorted_b = _generate_distorted_params(params, rng)

    render_size = (400, 300)
    rendered_gt = _render_full_image(original_path, params,       render_size)
    rendered_da = _render_full_image(original_path, distorted_a,  render_size)
    rendered_db = _render_full_image(original_path, distorted_b,  render_size)

    # 随机打乱 A/B/C 与 GT 的对应关系
    options = [('gt', params, rendered_gt),
               ('da', distorted_a, rendered_da),
               ('db', distorted_b, rendered_db)]
    rng.shuffle(options)
    labels   = ['A', 'B', 'C']
    gt_label = labels[[opt[0] for opt in options].index('gt')]

    def pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def path_to_b64(path: str) -> str:
        return base64.b64encode(open(path, 'rb').read()).decode()

    def fmt_params(p: dict) -> str:
        core = {k: v for k, v in p.items()
                if k != 'color_mixer' and isinstance(v, (int, float))}
        return json.dumps(core, ensure_ascii=False, indent=2)

    user_content = []

    # 原图
    user_content.append({"type": "text", "text": "[Original Photo — before any edits]"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{path_to_b64(original_path)}", "detail": "low"}
    })

    # 3 套渲染结果 + 对应参数
    for letter, (_, p, rendered) in zip(labels, options):
        user_content.append({
            "type": "text",
            "text": f"[Option {letter} — rendered result]"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(rendered)}", "detail": "low"}
        })
        user_content.append({
            "type": "text",
            "text": f"[Option {letter} — parameter set]\n{fmt_params(p)}"
        })

    # Sweep montages（GT 的候选扫描图，供写 CoT 时参考）
    # 每个参数的候选图已按升序存为独立文件；此处临时拼接后发送，避免超过 API 图片数上限。
    for label_param in SWEEP_PARAMS:
        if label_param not in montage_paths:
            continue
        stitched = _stitch_for_llm(montage_paths[label_param])
        user_content.append({
            "type": "text",
            "text": (f"[{label_param} candidate sweep (low→high) — "
                     f"green border marks the selected value in the correct parameter set]")
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{pil_to_b64(stitched)}",
                "detail": "low"
            }
        })

    # 任务指令（选择 + CoT 合并为一次调用）
    user_content.append({"type": "text", "text": (
        "Three parameter sets (A, B, C) are shown above. "
        "Exactly one of them is the genuine edit applied to this photo.\n\n"
        "Task:\n"
        "1. Compare the original photo with each of the three rendered results. "
        "Identify which option (A, B, or C) most plausibly produced a well-crafted, "
        "intentional edit.\n"
        "2. On the very first line, write only: Best match: [letter]\n"
        "3. Then write the ASSR CoT for the parameter set you selected, "
        "referencing the sweep montages above as visual evidence.\n\n"
        "Output format (no extra commentary before or after):\n"
        "Best match: [A/B/C]\n\n"
        "## Analysis\n...\n\n## Sweep & Select\n...\n\n## Refine\n..."
    )})

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ASSR},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=900,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()

            # 解析第一行：严格匹配 Prompt 要求的 "Best match: [A/B/C]" 前缀，
            # 不使用宽泛的 \b[ABC]\b，避免正文中任意大写字母被误抓
            first_line = text.split('\n')[0]
            match = re.search(r'Best\s+match:\s*([ABC])', first_line, re.IGNORECASE)
            selected = match.group(1).upper() if match else None

            if selected != gt_label:
                # LLM 没有选中 GT（或格式不符）→ 难样本，直接丢弃
                return None, True

            # 提取 CoT 部分：匹配 ## Analysis / ### Analysis 等变体，
            # 若完全缺失则视为格式不合规，丢弃该条，不把含 "Best match" 的原始文本存入训练集
            cot_match = re.search(r'(#{1,4}\s*Analysis.*)', text, re.DOTALL | re.IGNORECASE)
            if cot_match is None:
                print("    LLM response missing ## Analysis section, discarding")
                return None, True
            cot_text = cot_match.group(1).strip()
            return cot_text, False

        except openai.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"    Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    LLM error (attempt {attempt + 1}): {e}")
            time.sleep(2)

    return None, False  # API 全部失败 → 回退规则模板


# ============================================================
# SECTION 5: 主生成函数
# ============================================================

def generate_visual_assr_cot(original_path: str,
                              params: dict,
                              montage_output_dir: str,
                              sample_id: str,
                              n_candidates: int = 5,
                              use_llm: bool = False) -> tuple[Optional[str], dict, bool]:
    """
    Generate a complete ASSR CoT (text + sweep montages) for one sample.

    Args:
      original_path      : path to the original image
      params             : parameter dict returned by extract_params()
      montage_output_dir : directory to save sweep montages
      sample_id          : sample ID used for file naming
      n_candidates       : number of candidate values per parameter (default 5)
      use_llm          : if True, call LLM for text; otherwise use rule template

    Returns:
      (cot_text, montage_paths, discarded)
      - cot_text     : ASSR-structured CoT string，LLM 选错时为 None
      - montage_paths: {param_label: absolute image path}
      - discarded    : True 表示 LLM 选错 GT → 难样本，调用方应跳过该条数据
    """
    Path(montage_output_dir).mkdir(parents=True, exist_ok=True)
    montage_paths = {}

    # Step A：按阶段累积渲染，为每个关键参数生成并保存候选图。
    # 保存分辨率为 CANDIDATE_SAVE_SIZE（供人工浏览/模型训练）；
    # LLM API 调用时由 _stitch_for_llm 另行缩小。
    # 阶段间：以前序阶段 GT 值的累积渲染结果为基底。
    # 阶段内：每扫完一个参数，立即把其 GT 值叠到基底上，下一个参数在此基础上扫描。
    for stage_idx, stage_labels in enumerate(STAGE_PARAM_GROUPS):
        # 计算当前阶段入口基底：将 0..stage_idx-1 阶段的 GT 参数依次应用到原图
        base_arr = render_cumulative_base(original_path, params, stage_idx, CANDIDATE_SAVE_SIZE)

        for label in stage_labels:
            gt_value = params.get(label)

            # color_mixer 是 list 类型，需要单独处理
            if label == 'color_mixer':
                if gt_value and isinstance(gt_value, list) and 'color_mixer' in SWEEP_PARAMS:
                    total_strength = sum(
                        abs(e.get('H') or 0) + abs(e.get('S') or 0) + abs(e.get('L') or 0)
                        for e in gt_value
                    )
                    if total_strength >= 5:
                        cm_candidates, cm_scales, cm_gt_idx = generate_sweep_candidates_color_mixer(
                            gt_value, n_candidates
                        )
                        param_dir = Path(montage_output_dir) / 'color_mixer'
                        montage_paths['color_mixer'] = save_candidate_thumbs_color_mixer(
                            base_arr, cm_candidates, cm_scales, cm_gt_idx, param_dir
                        )
                if gt_value is not None:
                    base_arr = RENDER_FUNCTIONS['color_mixer'](base_arr, gt_value)
                continue

            # 标量参数：判断是否需要生成扫描网格图
            should_sweep = (
                label in SWEEP_PARAMS
                and gt_value is not None
                and abs(float(gt_value)) >= 5
                and label in RENDER_FUNCTIONS
            )

            if should_sweep:
                candidates, _ = generate_sweep_candidates(label, gt_value, n_candidates)
                param_dir = Path(montage_output_dir) / label
                montage_paths[label] = save_candidate_thumbs(
                    base_arr, label, candidates, gt_value, param_dir
                )

            # 无论是否扫描，都把该参数的 GT 值叠到基底，供同阶段后续参数使用
            if gt_value is not None and label in RENDER_FUNCTIONS:
                base_arr = RENDER_FUNCTIONS[label](base_arr, float(gt_value))

    # Step B：生成 CoT 文字
    if use_llm:
        cot_text, is_hard = _generate_llm_cot(original_path, params, montage_paths, sample_id=sample_id)
        if is_hard:
            # LLM 选错 GT → 难样本，丢弃
            return None, montage_paths, True
        if cot_text is None:
            # API 调用失败 → 回退规则模板
            print("    LLM failed, falling back to rule template")
            cot_text = _build_rule_cot_text(params, montage_paths)
    else:
        cot_text = _build_rule_cot_text(params, montage_paths)

    return cot_text, montage_paths, False


# ============================================================
# SECTION 6: 批量生成入口（供 run_pipeline.py 调用）
# ============================================================

def generate_all_cot(valid_samples: list,
                     output_cache_path: str,
                     llm_ratio: float = 0.3) -> dict:
    """
    Generate CoT for all valid samples, sorted by magnitude descending.
    The top llm_ratio fraction uses LLM; the rest use the rule template.

    Supports resume: existing cache entries are skipped automatically.

    Args:
      valid_samples      : output of Step2 collect_valid_samples()
      output_cache_path  : path to the cache JSON file
      llm_ratio        : fraction of samples to process with LLM (default 0.3)

    Returns:
      {sample_id: {"cot": str, "montage_paths": dict, "method": str, "magnitude": float}}
    """
    cache_path = Path(output_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache for resume support
    if cache_path.exists():
        with open(cache_path, encoding='utf-8') as f:
            cot_dict = json.load(f)
        print(f"Loaded cache: {len(cot_dict)} entries")
    else:
        cot_dict = {}

    # Sort by magnitude descending to decide which samples use LLM
    sorted_samples = sorted(valid_samples, key=lambda x: x['magnitude'], reverse=True)
    llm_count = int(len(sorted_samples) * llm_ratio)

    print(f"\nTotal samples: {len(sorted_samples)}")
    print(f"LLM: top {llm_count} samples")
    print(f"Rule template: remaining {len(sorted_samples) - llm_count} samples\n")

    for i, sample in enumerate(sorted_samples):
        sid = sample['sample_id']

        # Skip already-processed entries
        if sid in cot_dict:
            continue

        use_llm = (i < llm_count)
        method_label = "LLM" if use_llm else "rule"

        if use_llm or (i + 1) % 100 == 0:
            print(
                f"[{i+1}/{len(sorted_samples)}] {method_label}  "
                f"{sid[:24]}  magnitude={sample['magnitude']:.3f}",
                end=" ... " if use_llm else "\n",
                flush=True
            )

        cot_text, montage_paths, discarded = generate_visual_assr_cot(
            original_path=sample['original'],
            params=sample['params'],
            montage_output_dir=f"sft_data/montages/{sid}",
            sample_id=sid,
            n_candidates=5,
            use_llm=use_llm,
        )

        if use_llm:
            if discarded:
                print("✗ hard sample (GT not identified), discarded")
            else:
                print("✓")
            time.sleep(1.0)   # throttle to avoid rate limit

        cot_dict[sid] = {
            "cot":           cot_text,
            "montage_paths": montage_paths,
            "method":        ("llm_discarded" if discarded else "llm") if use_llm else "rule",
            "magnitude":     sample['magnitude'],
            "discarded":     discarded,
        }

        # Save every 10 entries to guard against mid-run crashes
        if (i + 1) % 10 == 0:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cot_dict, f, ensure_ascii=False, indent=2)

    # Final save
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cot_dict, f, ensure_ascii=False, indent=2)

    llm_n    = sum(1 for v in cot_dict.values() if v['method'] == 'llm')
    discarded_n = sum(1 for v in cot_dict.values() if v.get('discarded'))
    rule_n     = sum(1 for v in cot_dict.values() if v['method'] == 'rule')
    print(f"\nDone: LLM {llm_n} entries, "
          f"discarded (hard samples) {discarded_n}, "
          f"rule template {rule_n} entries")
    print(f"Cache saved to {cache_path}")

    return cot_dict


# ============================================================
# SECTION 7: 独立测试入口
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from step1_parse_params import extract_params
    from step2_filter import collect_valid_samples

    # Quick smoke test: process only 2 samples, no LLM API calls
    test_samples = collect_valid_samples(
        dataset_root='scraped_data_v4',
        split='test',
        min_magnitude=0.5,
    )[:2]

    result = generate_all_cot(
        valid_samples=test_samples,
        output_cache_path='sft_data/cot_cache_test.json',
        llm_ratio=0.0,   # 测试时全用规则模板，不花 API 费用
    )

    for sid, entry in result.items():
        print(f"\n{'='*60}")
        print(f"Sample: {sid}")
        print(f"Method: {entry['method']}")
        print(f"Montages: {list(entry['montage_paths'].keys())}")
        print(f"CoT preview:\n{entry['cot'][:300]}...")