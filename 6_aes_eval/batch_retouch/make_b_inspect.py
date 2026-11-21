#!/usr/bin/env python3
"""make_b_inspect.py — 生成「生图模型改动了什么」检查页的图片与数据（qwen-image、muse-image）
================================================================================

  · 依赖 align_b_outputs.py 产出的 b_alignment_metrics.csv（先运行它）
  · 选图：每个模型「对齐后仍差异最大」10 张（最可能有真实内容 / 细节改动）
    ∪「未对齐差异大、对齐后接近原图（≥0.95）」5 张（整体位移的典型）∪ 其余照片随机 8 张（RNG 固定）
  · 导出（长边 1024；原图与 qwen 是原生分辨率，muse 原生约 1950 缩小）→ <out>/img/：
      {id}_orig.jpg     原图（各模型收到的输入 q95）
      {id}_{m}.jpg      模型原始输出，缩放到原图尺寸（与未对齐指标的算法一致）
      {id}_{m}_al.jpg   用对齐矩阵搬到原图位置的输出（露出的窄边用边缘像素填充）
      {id}_t.jpg        原图缩略图（长边 360）
    数据 → <out>/data.json

用法（aes 环境）：python 6_aes_eval/batch_retouch/make_b_inspect.py --out <页面目录>
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
RUNS = REPO / "7_baselines" / "runs"
MET = bq.bg_dir(REPO) / "retouch_final" / "final_model_test"
MODELS = {"qwen": ("qwen-image-2.1-turbo", "B_821e477950/qwen__qwen-image-2.1-turbo", "Qwen-Image 2.1 Turbo"),
          "muse": ("muse-image", "B_821e477950/meta__muse-image", "Muse-Image")}
LONG, Q = 1024, 90


def page_matrix(row, w, h):
    """512 尺度的对齐矩阵换算到页面图尺寸：W_page = S·W·S⁻¹，S = diag(w/w512, h/h512)。"""
    W = np.array([[row.w00, row.w01, row.w02], [row.w10, row.w11, row.w12], [0, 0, 1]], np.float64)
    S = np.diag([w / row.w512, h / row.h512, 1.0])
    return (S @ W @ np.linalg.inv(S))[:2].astype(np.float32)


def export(job):
    cv2.setNumThreads(1)
    uid, orig, outs, out_dir = job
    o = cv2.imread(orig)
    s = min(1.0, LONG / max(o.shape[:2]))
    if s < 1:
        o = cv2.resize(o, (round(o.shape[1] * s), round(o.shape[0] * s)), interpolation=cv2.INTER_AREA)
    h, w = o.shape[:2]
    p = out_dir / "img"
    cv2.imwrite(str(p / f"{uid[:8]}_orig.jpg"), o, [cv2.IMWRITE_JPEG_QUALITY, Q])
    t = 360 / max(w, h)
    cv2.imwrite(str(p / f"{uid[:8]}_t.jpg"), cv2.resize(o, (round(w * t), round(h * t)), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    info = {}
    for m, (path, row) in outs.items():
        x = cv2.imread(path)
        aspect = (x.shape[1] / x.shape[0]) / (w / h)
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(p / f"{uid[:8]}_{m}.jpg"), x, [cv2.IMWRITE_JPEG_QUALITY, Q])
        al = cv2.warpAffine(x, page_matrix(row, w, h), (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(p / f"{uid[:8]}_{m}_al.jpg"), al, [cv2.IMWRITE_JPEG_QUALITY, Q])
        info[m] = round(float(aspect), 3)
    return uid, w, h, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    al = pd.read_csv(MET / "b_alignment_metrics.csv")
    summ = json.loads((MET / "b_alignment_summary.json").read_text())
    ann = json.loads((REPO / "fit_annotations.json").read_text())
    cat_of = {k.split("/", 1)[1]: (v.get("category") or "其他") for k, v in ann.items()}

    by = {m: al[al.arm == arm].set_index("uuid") for m, (arm, _, _) in MODELS.items()}
    common = sorted(set(by["qwen"].index) & set(by["muse"].index))
    pick, reason = [], {}

    def add(u, why):
        if u not in reason:
            pick.append(u)
        reason.setdefault(u, []).append(why)

    for m in MODELS:
        g = by[m].loc[common]
        for u in g.nsmallest(10, "aligned_edge").index:
            add(u, f"{m}:对齐后仍有差异")
        for u in g[g.aligned_edge >= 0.95].nsmallest(5, "raw_edge").index:
            add(u, f"{m}:整体位移")
    rest = sorted(set(common) - set(reason))
    for u in np.random.default_rng(20261121).choice(rest, 8, replace=False):
        add(str(u), "随机对照")

    jobs = []
    for u in pick:
        outs = {m: (str(RUNS / sub / "test" / f"{u}.jpg"), by[m].loc[u]) for m, (_, sub, _) in MODELS.items()}
        jobs.append((u, str(RUNS / "A/_inputs_jpg95/test" / f"{u}.jpg"), outs, out))
    with mp.get_context("fork").Pool(12) as pool:
        done = {r[0]: r[1:] for r in pool.map(export, jobs)}

    items = []
    for u in pick:
        w, h, aspect = done[u]
        it = {"id": u[:8], "uuid": u, "category": cat_of.get(u, "其他"), "reason": reason[u], "w": w, "h": h}
        for m in MODELS:
            r = by[m].loc[u]
            it[m] = {"raw_edge": round(float(r.raw_edge), 3), "aligned_edge": round(float(r.aligned_edge), 3),
                     "aligned_ssim": round(float(r.aligned_ssim), 3), "disp_pct": round(float(r.disp_pct), 2),
                     "scale": round(float(r.scale), 3), "rotation_deg": round(float(r.rotation_deg), 2),
                     "method": r.method, "aspect_change": aspect[m]}
        items.append(it)
    data = {"generated": "2026-11-21", "test_photos_both": len(common),
            "models": {m: name for m, (_, _, name) in MODELS.items()},
            "summary": summ, "items": items}
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print("选出", len(items), "张；原因计数",
          pd.Series([r for u in pick for r in reason[u]]).value_counts().to_dict())


if __name__ == "__main__":
    main()
