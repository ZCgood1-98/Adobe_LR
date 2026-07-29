#!/usr/bin/env python3
"""
make_refit_list.py — 生成「重拟合清单」（fit.py --only 格式）
============================================================

把被渲染一致性门（|Δ| > 阈值）过滤的 version 导出为两份清单：
  sft_data_v2/refit_train.txt / refit_test.txt   （每行 "<gid>:<version>"）

自动跳过已用新版 fit.py 重拟合过的 version（判据：fit_meta.json 含 crop_box 键，
只有新版拟合会写入）→ 重拟合中断后重新生成清单即可续跑，不重复劳动。

用法（仓库根目录）：
  python 5_training/make_refit_list.py                 # 默认阈值 8（与质量门②一致）
  python 5_training/make_refit_list.py --threshold 6   # 想连边缘拟合一起重拟就调低
  python 5_training/make_refit_list.py --no_skip_done  # 已重拟合的也列入（强制重来）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=REPO_ROOT / "sft_data_v2")
    ap.add_argument("--threshold", type=float, default=8.0, help="与质量门②一致")
    ap.add_argument("--no_skip_done", action="store_true",
                    help="不跳过已重拟合（fit_meta 含 crop_box）的 version")
    args = ap.parse_args()

    cons = json.loads((args.data_dir / "render_consistency.json").read_text())["diffs"]
    by_split: dict[str, list[str]] = {"train": [], "test": []}
    n_done = 0
    for vid, d in sorted(cons.items()):
        if d <= args.threshold:
            continue
        key, vn = vid.rsplit("/v", 1)
        split, gid = key.split("/", 1)
        if not args.no_skip_done:
            meta_p = REPO_ROOT / "fit_results" / key / f"version_{vn}" / "fit_meta.json"
            if meta_p.exists() and "crop_box" in json.loads(meta_p.read_text()):
                n_done += 1
                continue
        by_split[split].append(f"{gid}:{vn}")

    for split, lines in by_split.items():
        out = args.data_dir / f"refit_{split}.txt"
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"[list] {out.name}: {len(lines)} 个 version")
    if n_done:
        print(f"[list] 跳过已重拟合（fit_meta 含 crop_box）: {n_done} 个")
    total = sum(len(v) for v in by_split.values())
    print(f"[list] 合计 {total} 个，预计 ~{total * 78 / 3600:.1f} h GPU（约 78s/个）")


if __name__ == "__main__":
    main()
