#!/usr/bin/env python3
"""把原图复制到 fit_results 对应 ID 目录。

背景：
    - try_new_dataset1/{train,test}/{ID}/original_captured_cropped.png  是原图
    - fit_results/{train,test}/{ID}/                                    是重新拟合渲染的结果，缺原图
    本脚本按 ID 把原图复制一份到 fit_results 对应组里，便于原图/拟合图对比。

用法：
    python copy_originals_to_fit_results.py --dry-run   # 先预览，不实际写盘
    python copy_originals_to_fit_results.py             # 实际复制
    python copy_originals_to_fit_results.py --symlink   # 用软链接代替复制（省磁盘）
    python copy_originals_to_fit_results.py --overwrite # 目标已存在也覆盖
"""
import argparse
import os
import shutil
from pathlib import Path

ORIG_NAME = "original_captured_cropped.png"  # 原图文件名（源目录里已确认全部统一）
SPLITS = ("train", "test")                    # 数据划分子目录

# 默认路径锚定到脚本所在目录，这样在任意 cwd 下运行都正确
REPO_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="把原图复制到 fit_results 对应 ID 目录")
    parser.add_argument("--src", type=Path, default=REPO_ROOT / "try_new_dataset1",
                        help="原图所在数据集目录（默认 try_new_dataset1）")
    parser.add_argument("--dst", type=Path, default=REPO_ROOT / "fit_results",
                        help="拟合结果目录（默认 fit_results）")
    parser.add_argument("--dry-run", action="store_true", help="只预览不写盘")
    parser.add_argument("--overwrite", action="store_true", help="目标已存在时也覆盖")
    parser.add_argument("--symlink", action="store_true", help="用软链接代替复制以节省磁盘")
    args = parser.parse_args()

    copied = skipped = missing = 0
    missing_ids = []  # 在 fit_results 里但源数据集找不到原图的组

    for split in SPLITS:
        dst_split = args.dst / split
        if not dst_split.is_dir():
            print(f"[警告] 目标划分目录不存在，跳过：{dst_split}")
            continue

        # 遍历 fit_results 下每一个 ID 组
        for group in sorted(p for p in dst_split.iterdir() if p.is_dir()):
            gid = group.name
            src_img = args.src / split / gid / ORIG_NAME
            dst_img = group / ORIG_NAME

            # 源原图不存在 —— 记录并跳过（train 里已知有 18 个这种组）
            if not src_img.is_file():
                missing += 1
                missing_ids.append(f"{split}/{gid}")
                continue

            # 目标已存在且未指定覆盖 —— 跳过（幂等）
            if (dst_img.exists() or dst_img.is_symlink()) and not args.overwrite:
                skipped += 1
                continue

            if args.dry_run:
                action = "软链" if args.symlink else "复制"
                print(f"[预览-{action}] {src_img}  ->  {dst_img}")
            elif args.symlink:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                # 用相对路径软链，目录整体移动时仍有效
                os.symlink(os.path.relpath(src_img, dst_img.parent), dst_img)
            else:
                shutil.copy2(src_img, dst_img)  # copy2 保留时间戳等元数据
            copied += 1

    # ===== 汇总 =====
    verb = "待复制" if args.dry_run else "已复制"
    print("\n===== 汇总 =====")
    print(f"{verb}        : {copied}")
    print(f"跳过(已存在) : {skipped}")
    print(f"源缺失原图   : {missing}")
    if missing_ids:
        print(f"\n以下 {len(missing_ids)} 个 fit_results 组在源数据集中找不到原图：")
        for mid in missing_ids:
            print(f"  {mid}")


if __name__ == "__main__":
    main()
