#!/usr/bin/env python3
"""
scan_render_consistency.py — 全量扫描「标签渲染一致性」
======================================================

对每个入选 version 计算：用构建器同款路径（输入图 + params.json 舍入参数，
4 阶段累积一次性渲染）得到的终图，与拟合产物 rendered.png 的平均逐像素差。

差异大的 version 说明 (输入图, 参数标签) 与拟合几何/渲染不一致，典型原因：
  - 拟合时 crop 对齐过，但 make_cropped_originals 判定未生成 original_cropped.png
    （宽高比/覆盖率阈值边界情况）→ 输入取景错位；
  - 差拟合（final_loss 贴质量门）叠加 torch/numpy 渲染分歧。

产出 sft_data_v2/render_consistency.json：{"train/<gid>/vN": mean_abs_diff}，
供 build_sft_dataset.py --max_render_diff 过滤（只重组 JSON，不重渲图片）。

用法（仓库根目录）：python 5_training/scan_render_consistency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))

from differentiable_renderer.opencv_renderer import render  # noqa: E402
from build_sft_dataset import (collect_jobs, extract_stages,  # noqa: E402
                               _imread_rgb, _resize_max_side)


def check_one(job: dict) -> tuple[str, float | None, str]:
    key, vn = job["key"], job["vn"]
    vid = f"{key}/v{vn}"
    try:
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        p = json.loads((vdir / "params.json").read_text())
        in_png = vdir / "original_cropped.png"
        if not in_png.exists():
            in_png = REPO_ROOT / "fit_results" / key / "original_captured_cropped.png"
        img0 = _resize_max_side(_imread_rgb(str(in_png)), job["max_side"])
        cum: dict = {}
        for zh, _ in extract_stages(p):
            cum.update(zh)
        mine = render(img0, cum).astype(np.float64)
        ref = _imread_rgb(str(vdir / "rendered.png"))
        ref = cv2.resize(ref, (mine.shape[1], mine.shape[0]),
                         interpolation=cv2.INTER_AREA).astype(np.float64)
        return vid, float(np.abs(mine - ref).mean()), ""
    except Exception as e:  # noqa: BLE001
        return vid, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "sft_data_v2")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--max_final_loss", type=float, default=8.0)
    ap.add_argument("--loss_whitelist", type=Path, default=None,
                    help="同构建器：白名单 version 豁免 loss 门、纳入扫描"
                         "（默认 <out>/loss_whitelist.txt，文件存在才生效）")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jpeg_q", type=int, default=92)  # collect_jobs 需要，无实际作用
    args = ap.parse_args()
    if args.loss_whitelist is None:
        args.loss_whitelist = args.out / "loss_whitelist.txt"

    jobs, _ = collect_jobs(args)
    print(f"[scan] {len(jobs)} versions")
    t0, diffs, errs = time.time(), {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(check_one, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            vid, d, err = fut.result()
            if d is None:
                errs[vid] = err
            else:
                diffs[vid] = round(d, 3)
            if n % 500 == 0 or n == len(jobs):
                rate = n / (time.time() - t0)
                print(f"  [{n}/{len(jobs)}] {rate:.1f} it/s  "
                      f"eta {(len(jobs) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)

    out_path = args.out / "render_consistency.json"
    out_path.write_text(json.dumps({"diffs": diffs, "errors": errs},
                                   ensure_ascii=False, indent=1))
    vals = np.array(sorted(diffs.values()))
    print(f"[scan] 写入 {out_path}  errors={len(errs)}")
    for q in [50, 90, 95, 98, 99, 99.5]:
        print(f"  p{q}: {np.percentile(vals, q):.2f}/255")
    for th in [6, 8, 10, 15]:
        print(f"  >{th}: {(vals > th).sum()} 个 ({(vals > th).mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
