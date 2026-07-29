#!/usr/bin/env python3
"""v2_direction_probe.py — 方向语义标定探针（诊断课过滤器的前置校准）
=================================================================

背景：渲染器的 温度 方向与 LR 滑块相反（README 明示）、晕影/色调等方向语义从未验证。
诊断课的一致性过滤器要拿「老师说 warm」对照「GT 参数符号」，映射错了会成批误杀。

做法：取 3 张不同题材原图，对每个待校准轴渲 (−值 / 中性 / +值) 三张，
用图像统计量判定「参数增大」对应的视觉方向，三图一致才收录；
产出 sft_data_v2/direction_calibration.json + 对照拼图（人眼复核）。

用法（仓库根目录，Chen 环境即可）：python 5_training/v2_direction_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from differentiable_renderer.opencv_renderer import render  # noqa: E402

OUT_JSON = REPO_ROOT / "sft_data_v2" / "direction_calibration.json"
OUT_PNG = REPO_ROOT / "sft_data_v2" / "direction_probe_montage.png"

# 探针图：挑 3 个不同题材的组原图（存在性在 main 里兜底）
PROBE_IMAGES = [
    "fit_results/train/058e9667-da16-4361-bf10-c0b253341468/original_captured_cropped.png",
    "fit_results/train/af18d741-a22f-4ee2-bcc4-0ab1d812efba/original_captured_cropped.png",
    "fit_results/train/03996bc3-6909-45ff-bf5f-637aad099550/original_captured_cropped.png",
]

# (轴名, 渲染参数键, 低值, 高值, 度量函数名, 过滤器轴名, {度量增大: 方向词, 度量减小: 方向词})
# 度量统一定义为「高值渲染的度量 − 低值渲染的度量」>0 时 → param_up_means 取 up_word
PROBES = [
    ("exposure",   "曝光度",   -1.5,  1.5,  "mean_luma",    "exposure",   ("brighten", "darken")),
    ("wb_temp",    "色温",     3500, 8000,  "warm_index",   "wb",         ("warm", "cool")),
    ("tint",       "色调",      -60,   60,  "green_index",  "tint",       ("green", "magenta")),
    ("contrast",   "对比度",    -60,   60,  "luma_std",     "contrast",   ("up", "down")),
    ("highlights", "高光",      -70,   70,  "hi_region",    "highlights", ("raise", "lower")),
    ("shadows",    "阴影",      -70,   70,  "lo_region",    "shadows",    ("lift", "lower")),
    ("saturation", "饱和度",    -60,   60,  "mean_sat",     "saturation", ("up", "down")),
    ("vibrance",   "自然饱和度", -60,   60,  "mean_sat",     "saturation", ("up", "down")),
    ("vignette",   "晕影",      -90,   90,  "corner_ratio", "vignette",   ("brighten", "darken")),
    ("dehaze",     "去朦胧",    -60,   60,  "luma_std",     "dehaze",     ("up", "down")),
]


def _luma(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])


def metrics(img: np.ndarray, ref_luma: np.ndarray) -> dict:
    """一次算全部度量；高/低光区掩码取自中性渲染(ref)，保证三张图同一掩码。"""
    y = _luma(img)
    hi_mask = ref_luma >= np.percentile(ref_luma, 75)
    lo_mask = ref_luma <= np.percentile(ref_luma, 25)
    h, w = y.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    corner, center = r2 > 0.85, r2 < 0.3
    sat = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1]
    return {
        "mean_luma": float(y.mean()),
        "warm_index": float(img[..., 0].mean() - img[..., 2].mean()),          # R−B
        "green_index": float(img[..., 1].mean() - (img[..., 0].mean() + img[..., 2].mean()) / 2),
        "luma_std": float(y.std()),
        "hi_region": float(y[hi_mask].mean()),
        "lo_region": float(y[lo_mask].mean()),
        "mean_sat": float(sat.mean()),
        "corner_ratio": float(y[corner].mean() / max(y[center].mean(), 1e-6)),
    }


def main():
    imgs = []
    for rel in PROBE_IMAGES:
        p = REPO_ROOT / rel
        if p.exists():
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            s = 768 / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                                 interpolation=cv2.INTER_AREA)
            imgs.append(img)
    if len(imgs) < 3:  # 兜底：随便再抓几张
        for gdir in sorted((REPO_ROOT / "fit_results" / "train").iterdir()):
            p = gdir / "original_captured_cropped.png"
            if p.exists() and len(imgs) < 3:
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                s = 768 / max(img.shape[:2])
                imgs.append(cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
                            if s < 1 else img)
    print(f"探针图 {len(imgs)} 张")

    calib, rows_for_montage, inconsistent = {}, [], []
    for name, zh_key, lo, hi, metric, axis, (up_word, down_word) in PROBES:
        signs = []
        for i, img in enumerate(imgs):
            neutral = render(img, {})
            ref_y = _luma(neutral)
            m_lo = metrics(render(img, {zh_key: float(lo)}), ref_y)[metric]
            m_hi = metrics(render(img, {zh_key: float(hi)}), ref_y)[metric]
            signs.append(np.sign(m_hi - m_lo))
            if i == 0:  # 第一张图进拼图
                rows_for_montage.append((name, render(img, {zh_key: float(lo)}),
                                         neutral, render(img, {zh_key: float(hi)})))
        agree = len(set(signs)) == 1 and signs[0] != 0
        param_up_means = up_word if signs[0] > 0 else down_word
        calib[name] = {
            "render_key": zh_key, "filter_axis": axis,
            "param_up_means": param_up_means,
            "param_down_means": down_word if param_up_means == up_word else up_word,
            "probe_values": [lo, hi], "metric": metric,
            "signs_per_image": [int(s) for s in signs], "consistent": bool(agree),
        }
        flag = "" if agree else "  ⚠ 三图不一致!"
        if not agree:
            inconsistent.append(name)
        print(f"{name:11} ({zh_key}): 参数增大 → {param_up_means}{flag}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(calib, ensure_ascii=False, indent=2))
    print(f"\n校准写入 {OUT_JSON}")
    if inconsistent:
        print(f"⚠ 不一致轴：{inconsistent} —— 过滤器将跳过这些轴，需人工定夺")

    # 拼图：每轴一行 [低值 | 中性 | 高值]，行首写轴名
    tiles = []
    for name, im_lo, im_mid, im_hi in rows_for_montage:
        h = 200
        def rs(x):
            return cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h))
        row = np.hstack([rs(im_lo), np.full((h, 4, 3), 255, np.uint8),
                         rs(im_mid), np.full((h, 4, 3), 255, np.uint8), rs(im_hi)])
        label = np.full((h, 170, 3), 30, np.uint8)
        cv2.putText(label, name, (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(np.hstack([label, row]))
    W = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles]
    cv2.imwrite(str(OUT_PNG), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))
    print(f"对照拼图写入 {OUT_PNG}（每行: 低值|中性|高值）")


if __name__ == "__main__":
    main()
