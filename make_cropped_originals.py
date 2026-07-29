#!/usr/bin/env python3
"""为带裁剪的 version 生成像素对齐的「裁剪原图」（训练只学参数、不学裁剪）。

背景：
    同一组的不同 version 可能各自带不同的 Lightroom 裁剪。若训练对
    (组级原图 -> version 图) 几何不对齐，模型会把"裁剪"也当成要学的行为。
    本脚本为每个判定为「裁剪过」的保留 version 生成一张与它逐像素对齐的裁剪原图，
    写入训练数据集 fit_results 的对应 version 目录（--out-root 可改）：

        fit_results/{split}/{ID}/version_N/original_cropped.png
        （尺寸与该 version 的 main_image.png 一致，即拟合时的对齐几何，
         与同目录 rendered.png / params.json 配套）

    注意：原图与 version 编辑图仍从源数据集 try_new_dataset1 读取（--src-root），
    因为 main_image.png 只存在于源数据集；fit_results 只是输出位置。

    训练/评测数据加载约定：fit_results 的 version 目录里存在 original_cropped.png
    就用它当输入，否则回退到组级 original_captured_cropped.png。

判定与对齐（复用 differentiable_renderer/crop_align.py，与拟合管线同源）：
    - 宽高比差 ≥2%（needs_crop）→ 必是裁剪：SIFT 找框；SIFT 失败退中心裁剪；
    - 宽高比相同 → 仍跑 SIFT 甄别「等比例缩放式裁剪」：裁剪框线性覆盖率
      max(w/W, h/H) < --coverage（默认 0.95）才算真裁剪，否则视为未裁剪、不生成文件
      （SIFT 角点噪声通常 <1%，真实裁剪一般 ≥5%，0.95 足够区分）；
    - SIFT 在长边 ≤ --sift-side 的缩图上跑、框坐标映射回原尺寸（提速数倍）；
    - 只处理有效组（非整组舍弃）的「保留」version；裁剪用全分辨率原图，
      最后 LANCZOS 缩放到该 version 的尺寸。

用法：
    python make_cropped_originals.py --limit 60     # 冒烟测试（只跑前 60 个配对）
    python make_cropped_originals.py                # 全量（已存在的输出跳过，可断点续跑）
    python make_cropped_originals.py --overwrite    # 忽略已存在、全部重新生成
    可选：--workers 16  --coverage 0.95  --sift-side 1200
结果清单写入 cropped_originals_manifest.json（每个配对的判定、裁剪框、覆盖率）。
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from differentiable_renderer.crop_align import find_crop_box, needs_crop, _centre_crop_to_ratio  # noqa: E402

OUT_NAME = "original_cropped.png"


def _worker_init():
    """限制每个 worker 内 OpenCV 的线程数，避免多进程 × 多线程过订阅。"""
    import cv2
    cv2.setNumThreads(1)


def _gray_small(pil_img, max_side):
    """转灰度并缩到长边 ≤ max_side，返回 (ndarray, 缩放比)。"""
    w, h = pil_img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        g = pil_img.convert("L").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    else:
        g = pil_img.convert("L")
    return np.asarray(g), s


def process_one(task):
    """处理一个 (原图, version) 配对；返回判定记录（供 manifest 汇总）。"""
    key, v, orig_path, edited_path, out_path, coverage_th, sift_side, overwrite = task
    rec = {"key": key, "version": v}
    try:
        if not overwrite and Path(out_path).is_file():
            rec["status"] = "已存在跳过"
            return rec
        orig = Image.open(orig_path).convert("RGB")
        edited = Image.open(edited_path).convert("RGB")
        ow, oh = orig.size
        ew, eh = edited.size
        aspect_diff = needs_crop((ow, oh), (ew, eh))

        # 缩图跑 SIFT，框坐标按原图缩放比映射回全分辨率
        g_orig, s_orig = _gray_small(orig, sift_side)
        g_edit, _ = _gray_small(edited, sift_side)
        box = find_crop_box(g_orig, g_edit)
        if box is not None:
            x, y, w, h = (round(c / s_orig) for c in box)
            x, y = max(0, x), max(0, y)
            w, h = min(ow - x, w), min(oh - y, h)
            box = (x, y, w, h) if w >= 10 and h >= 10 else None
        if box is not None:
            rec["box"] = list(box)
            rec["coverage"] = round(max(box[2] / ow, box[3] / oh), 4)  # 线性覆盖率（取两轴较大者）

        if aspect_diff:
            # 宽高比不同：必是裁剪
            if box is not None:
                cropped, rec["method"] = orig.crop((box[0], box[1], box[0] + box[2], box[1] + box[3])), "sift"
            else:
                cropped, rec["method"] = _centre_crop_to_ratio(orig, ew / eh), "center_fallback"
        elif box is not None and rec["coverage"] < coverage_th:
            # 同比例但框明显小于全幅：等比例缩放式裁剪
            cropped, rec["method"] = orig.crop((box[0], box[1], box[0] + box[2], box[1] + box[3])), "sift_same_aspect"
        else:
            rec["status"] = "未裁剪"
            return rec

        cropped.resize((ew, eh), Image.LANCZOS).save(out_path)
        rec["status"] = "已生成"
        rec["out_size"] = [ew, eh]
        return rec
    except Exception as e:  # 单个失败不拖垮全量，记录后继续
        rec["status"] = "出错"
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec


def main():
    ap = argparse.ArgumentParser(description="为带裁剪的保留 version 生成像素对齐的裁剪原图")
    ap.add_argument("--src-root", type=Path, default=REPO_ROOT / "try_new_dataset1",
                    help="源数据集根目录（读原图与 main_image）")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "fit_results",
                    help="输出根目录（训练数据集，写 version_N/original_cropped.png）")
    ap.add_argument("--anno", type=Path, default=REPO_ROOT / "fit_annotations.json", help="标注 JSON")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "cropped_originals_manifest.json",
                    help="结果清单输出路径")
    ap.add_argument("--coverage", type=float, default=0.95,
                    help="同比例配对判定为裁剪的线性覆盖率阈值（小于该值才生成）")
    ap.add_argument("--sift-side", type=int, default=1200, help="SIFT 缩图长边")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="并行进程数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个配对（冒烟测试用，0=全量）")
    ap.add_argument("--overwrite", action="store_true", help="已存在的输出也重新生成")
    args = ap.parse_args()

    anno = json.load(open(args.anno, encoding="utf-8"))

    # ---- 组装任务：有效组的保留 version ----
    tasks, missing = [], []
    for key, rec in sorted(anno.items()):
        if rec.get("discarded"):
            continue
        split, gid = key.split("/", 1)
        gdir = args.src_root / split / gid
        orig = gdir / "original_captured_cropped.png"
        if not orig.is_file():
            missing.append({"key": key, "status": "缺组级原图"})
            continue
        for v, va in sorted((rec.get("versions") or {}).items(), key=lambda kv: int(kv[0])):
            d = va.get("decision") if isinstance(va, dict) else va
            if d == "discard":
                continue
            edited = gdir / f"version_{v}" / "main_image.png"
            if not edited.is_file():
                missing.append({"key": key, "version": v, "status": "缺main_image"})
                continue
            out_dir = args.out_root / split / gid / f"version_{v}"
            if not out_dir.is_dir():
                missing.append({"key": key, "version": v, "status": "输出version目录不存在"})
                continue
            tasks.append((key, v, str(orig), str(edited),
                          str(out_dir / OUT_NAME),
                          args.coverage, args.sift_side, args.overwrite))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"待处理配对: {len(tasks)}（缺文件跳过 {len(missing)}）， workers={args.workers}", flush=True)

    # ---- 并行处理 ----
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            results.append(fu.result())
            if i % 250 == 0 or i == len(futs):
                el = time.time() - t0
                eta = el / i * (len(futs) - i)
                print(f"  {i}/{len(futs)}  用时 {el:.0f}s  预计剩余 {eta:.0f}s", flush=True)

    # ---- 汇总 + 写清单 ----
    from collections import Counter
    by_status = Counter(r["status"] for r in results)
    by_method = Counter(r.get("method") for r in results if r["status"] == "已生成")
    print(f"\n状态统计: {dict(by_status)}")
    print(f"生成方式: {dict(by_method)}")
    errs = [r for r in results if r["status"] == "出错"]
    for r in errs[:10]:
        print("  出错:", r["key"], "v" + str(r["version"]), r["error"])

    manifest = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "coverage_threshold": args.coverage, "sift_side": args.sift_side,
        "out_name": OUT_NAME, "src_root": str(args.src_root), "out_root": str(args.out_root),
        "n_tasks": len(tasks), "by_status": dict(by_status), "by_method": dict(by_method),
        "missing_files": missing,
        "results": sorted(results, key=lambda r: (r["key"], int(r["version"]))),
    }
    tmp = args.manifest.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.manifest)
    print(f"清单已写入: {args.manifest}")


if __name__ == "__main__":
    main()
