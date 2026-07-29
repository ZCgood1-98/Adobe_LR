#!/usr/bin/env python3
"""
rescue_geometry.py — 救回「输入取景错位」被质量门②过滤的 version
================================================================

背景（见 README 质量门②）：826 个 version 因「累积渲染 vs rendered.png 平均|Δ|>8」
被弃用，其中 ~72% 是纯几何问题——拟合时 crop 对齐过，但 make_cropped_originals 的
判定阈值（宽高比≥2% / 覆盖率<0.95）是为「明显裁剪」设计的，边界 case 没生成
original_cropped.png → 构建器用了整幅组原图，取景与拟合输入错位。

修法（只处理「无 original_cropped.png」的问题 version）：
  1. 复用 make_cropped_originals.process_one，coverage 阈值放开到 2.0
     （SIFT 找到框就裁，不再做"是不是真裁剪"的门控）→ 强制生成对齐原图；
  2. 逐个重测渲染一致性（scan_render_consistency.check_one 同一把尺子）：
       新 diff ≤ 阈值 → 获救，保留生成的图、更新 render_consistency.json；
       仍 > 阈值   → 回滚（删除生成的图，diff 维持旧值）——防止坏框留在盘上
                     污染后续 GRPO 的输入约定；
  3. 删除获救 version 在 sft_data_v2/images 的旧中间图（从错位输入渲的，已失效），
     由重建自动补渲。

有对齐图但仍 >8 的那批（~28%）病根是拟合质量（final_loss 中位 4.1），本脚本不碰；
真修需重拟合，另行决策。

用法（仓库根目录）：
  python 5_training/rescue_geometry.py            # 全量救援
  python 5_training/rescue_geometry.py --limit 20 # 冒烟
  # 跑完后重建数据集（幂等，只补渲获救 version）：
  python 5_training/build_sft_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))

import make_cropped_originals as mco  # noqa: E402
from scan_render_consistency import check_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=REPO_ROOT / "sft_data_v2")
    ap.add_argument("--src_root", type=Path, default=REPO_ROOT / "try_new_dataset1")
    ap.add_argument("--threshold", type=float, default=8.0, help="与质量门②一致")
    ap.add_argument("--sift_side", type=int, default=1200)
    ap.add_argument("--max_side", type=int, default=1024, help="一致性复测分辨率，与扫描一致")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cons_path = args.data_dir / "render_consistency.json"
    cons = json.loads(cons_path.read_text())

    # ---- 圈定救援对象：diff>阈值 且 无 original_cropped.png 且 源文件齐全 ----
    targets, skipped = [], {"有oc(拟合质量问题,不碰)": 0, "缺源文件": 0}
    for vid, d in cons["diffs"].items():
        if d <= args.threshold:
            continue
        key, vn = vid.rsplit("/v", 1)
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        if (vdir / "original_cropped.png").exists():
            skipped["有oc(拟合质量问题,不碰)"] += 1
            continue
        split, gid = key.split("/", 1)
        orig = args.src_root / split / gid / "original_captured_cropped.png"
        edited = args.src_root / split / gid / f"version_{vn}" / "main_image.png"
        if not (orig.is_file() and edited.is_file()):
            skipped["缺源文件"] += 1
            continue
        targets.append({"vid": vid, "key": key, "vn": vn,
                        "old_diff": d, "vdir": vdir,
                        "orig": str(orig), "edited": str(edited)})
    targets.sort(key=lambda t: t["vid"])
    if args.limit:
        targets = targets[: args.limit]
    print(f"[rescue] 救援对象 {len(targets)} 个；跳过 {skipped}")

    # ---- 阶段1：强制生成对齐原图（coverage 阈值放开，找到框就裁） ----
    t0 = time.time()
    gen_ok, gen_fail = {}, []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=mco._worker_init) as ex:
        futs = {ex.submit(mco.process_one,
                          (t["key"], t["vn"], t["orig"], t["edited"],
                           str(t["vdir"] / mco.OUT_NAME),
                           2.0, args.sift_side, True)): t for t in targets}
        for n, fut in enumerate(as_completed(futs), 1):
            t, rec = futs[fut], fut.result()
            if rec.get("status") == "已生成":
                gen_ok[t["vid"]] = rec
            else:
                gen_fail.append({**rec, "vid": t["vid"]})
            if n % 100 == 0 or n == len(futs):
                print(f"  [SIFT {n}/{len(futs)}] {n / (time.time() - t0):.1f} it/s", flush=True)
    print(f"[rescue] 生成对齐原图 {len(gen_ok)}，SIFT 无框/失败 {len(gen_fail)}")

    # ---- 阶段2：一致性复测，成功者更新 json、失败者回滚删图 ----
    t0, recovered, reverted = time.time(), [], []
    tasks = [t for t in targets if t["vid"] in gen_ok]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(check_one, {"key": t["key"], "vn": t["vn"],
                                      "max_side": args.max_side}): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            _, new_diff, err = fut.result()
            if new_diff is not None and new_diff <= args.threshold:
                cons["diffs"][t["vid"]] = round(new_diff, 3)
                recovered.append({"vid": t["vid"], "old": t["old_diff"],
                                  "new": round(new_diff, 3),
                                  "method": gen_ok[t["vid"]].get("method"),
                                  "coverage": gen_ok[t["vid"]].get("coverage")})
                # 旧中间图从错位输入渲的 → 删除，重建时用新对齐原图补渲
                img_dir = args.data_dir / "images" / t["key"] / f"v{t['vn']}"
                for p in img_dir.glob("s*.jpg"):
                    p.unlink()
            else:
                (t["vdir"] / mco.OUT_NAME).unlink(missing_ok=True)  # 回滚坏框
                reverted.append({"vid": t["vid"], "old": t["old_diff"],
                                 "new": None if new_diff is None else round(new_diff, 3),
                                 "err": err or ""})
            if n % 100 == 0 or n == len(tasks):
                print(f"  [复测 {n}/{len(tasks)}] {n / (time.time() - t0):.1f} it/s", flush=True)

    cons_path.write_text(json.dumps(cons, ensure_ascii=False, indent=1))
    log = {"threshold": args.threshold, "targets": len(targets), "skipped": skipped,
           "generated": len(gen_ok), "sift_no_box": len(gen_fail),
           "recovered": len(recovered), "reverted": len(reverted),
           "recovered_detail": recovered, "reverted_detail": reverted,
           "gen_fail_detail": gen_fail}
    (args.data_dir / "rescue_geometry_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1))

    print(f"\n[rescue] 获救 {len(recovered)} / 回滚 {len(reverted)} / SIFT 无框 {len(gen_fail)}")
    if recovered:
        import numpy as np
        olds = [r["old"] for r in recovered]; news = [r["new"] for r in recovered]
        print(f"  获救 diff: {np.median(olds):.1f} → {np.median(news):.1f} (中位)")
    print(f"[rescue] 明细 → {args.data_dir / 'rescue_geometry_log.json'}")
    print("[rescue] 下一步：python 5_training/build_sft_dataset.py  （重建收回获救 version）")


if __name__ == "__main__":
    main()
