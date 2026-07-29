#!/usr/bin/env python3
"""
run_data_pipline.py
===============
将 Step1 ~ Step4 串联成一条完整的数据准备 Pipeline。

使用方式：
  # 完整运行（默认配置）
  python run_data_pipline.py

  # 自定义参数
  python run_data_pipline.py \
      --dataset_root /data/scraped_data_v4 \
      --output_dir   sft_data \
      --split        train 
      --min_magnitude 0.5 \
      --llm_ratio  0.3 \
      --seed         42

  # 跳过 Step3（只用规则模板，不调 LLM，适合快速测试）
  python run_data_pipline.py --llm_ratio 0.0

  # 从已有缓存继续（Step3 中途中断后恢复）
  python run_data_pipline.py --resume

  # 强制清空缓存并全量重新生成（适合后台 nohup 运行，避免交互式 input）
  python run_data_pipline.py --force_overwrite

  # 后台挂起运行示例
  nohup python run_data_pipline.py --force_overwrite > pipeline.log 2>&1 &
  nohup python run_data_pipline.py --resume         > pipeline.log 2>&1 &

Pipeline 流程：
  Step1  参数解析    extract_params()          → 内嵌在 Step2 中
  Step2  数据过滤    collect_valid_samples()    → valid_samples
                     色温以偏离中性（5500 K）的比例归一化，避免绝对值污染 magnitude 排序
  Step3  CoT生成     generate_all_cot()         → cot_dict + 候选缩略图
                     每个参数的候选图独立保存至 montages/{sid}/{param}/
                     顶部 llm_ratio 样本调用 LLM，其余用规则模板
                     同一 sample_id 的随机扰动结果跨进程可复现（SHA-256 种子）
  Step4  数据组装    build_sft_dataset()        → train.json + val.json
                     图像以 <image>\n<image> 分隔，兼容 LLaMA-Factory tokenizer
"""

import sys
import json
import time
import argparse
from pathlib import Path

# ============================================================
# SECTION 0: 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Photo editing SFT data pipeline (Step1 ~ Step4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root", type=str,
        default="scraped_data_v4",
        help="scraped_data_v4 根目录路径（默认：scraped_data_v4）",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="sft_data",
        help="所有输出文件的根目录（默认：sft_data）",
    )
    parser.add_argument(
        "--split", type=str,
        default="train",
        choices=["train", "test"],
        help="使用数据集的哪个 split（默认：train）",
    )
    parser.add_argument(
        "--min_magnitude", type=float,
        default=0.5,
        help="过滤阈值，归一化修图幅度低于此值的样本会被丢弃（默认：0.5）",
    )
    parser.add_argument(
        "--llm_ratio", type=float,
        default=0.3,
        help="前 N%% 高幅度样本使用 LLM 生成 CoT，其余用规则模板（默认：0.3）",
    )
    parser.add_argument(
        "--n_candidates", type=int,
        default=5,
        help="每个参数生成的候选值数量（默认：5）",
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="随机种子（控制候选值打乱和 train/val 划分，默认：42）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从已有的 cot_cache.json 断点续跑 Step3",
    )
    parser.add_argument(
        "--force_overwrite", action="store_true",
        help="若 cot_cache.json 已存在，强制清空并重新生成（默认：续跑已有缓存）",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=[2, 3, 4],
        choices=[2, 3, 4],
        help="指定只运行哪些步骤（默认：2 3 4）。"
             "Step1 始终嵌入在 Step2 中，无需单独指定。",
    )
    return parser.parse_args()


# ============================================================
# SECTION 1: 工具函数
# ============================================================

def print_banner(title: str):
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_step(step: int, name: str):
    print(f"\n{'─'*60}")
    print(f"  STEP {step}: {name}")
    print(f"{'─'*60}")


def save_samples_cache(samples: list, path: Path):
    """将 valid_samples 保存到 JSON，方便断点后直接加载跳过 Step2"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_samples_cache(path: Path) -> list:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# SECTION 2: 主流程
# ============================================================

def main():
    args = parse_args()

    # 路径准备
    output_dir   = Path(args.output_dir)
    montage_dir  = output_dir / "montages"
    cache_samples = output_dir / "valid_samples_cache.json"
    cache_cot     = output_dir / "cot_cache.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    print_banner("Photo Editing SFT Data Pipeline")
    print(f"  dataset_root  : {args.dataset_root}")
    print(f"  split         : {args.split}")
    print(f"  output_dir    : {output_dir}")
    print(f"  min_magnitude : {args.min_magnitude}")
    print(f"  llm_ratio   : {args.llm_ratio:.0%}")
    print(f"  n_candidates  : {args.n_candidates}")
    print(f"  seed          : {args.seed}")
    print(f"  steps         : {args.steps}")
    print(f"  resume        : {args.resume}")
    print(f"  force_overwrite: {args.force_overwrite}")

    t_total = time.time()

    # ----------------------------------------------------------
    # Step 2: 数据过滤
    # ----------------------------------------------------------
    valid_samples = []

    if 2 in args.steps:
        print_step(2, "数据过滤（含 Step1 参数解析）")
        t = time.time()

        # Step1 的 extract_params 已嵌入 Step2 的 collect_valid_samples 中
        from step2_filter import collect_valid_samples

        valid_samples = collect_valid_samples(
            dataset_root=args.dataset_root,
            split=args.split,
            min_magnitude=args.min_magnitude,
        )

        # 保存缓存，方便下次跳过 Step2 直接从 Step3 继续
        save_samples_cache(valid_samples, cache_samples)
        print(f"  → 缓存保存至 {cache_samples}")
        print(f"  → 耗时 {time.time() - t:.1f}s")

        if not valid_samples:
            print("\n[ERROR] 没有有效样本，请检查 dataset_root 路径和 min_magnitude 设置。")
            sys.exit(1)

    else:
        # 跳过 Step2，从缓存加载
        if not cache_samples.exists():
            print(f"\n[ERROR] 跳过 Step2 但缓存文件不存在：{cache_samples}")
            print("请先运行 Step2 或去掉 --steps 参数。")
            sys.exit(1)
        valid_samples = load_samples_cache(cache_samples)
        print(f"\n[跳过 Step2] 从缓存加载 {len(valid_samples)} 个样本")

    # ----------------------------------------------------------
    # Step 3: CoT 生成
    # ----------------------------------------------------------
    cot_dict = {}

    if 3 in args.steps:
        print_step(3, "CoT 生成（规则模板 + LLM）")
        t = time.time()

        from step3_assr_cot import generate_all_cot

        # 缓存处理策略：
        #   --force_overwrite  → 删除已有缓存，全量重新生成
        #   默认（含 --resume）→ 保留缓存，跳过已处理条目继续续跑
        if args.force_overwrite and cache_cot.exists():
            cache_cot.unlink()
            print(f"  → 已删除旧缓存 {cache_cot}，将全量重新生成")

        cot_dict = generate_all_cot(
            valid_samples=valid_samples,
            output_cache_path=str(cache_cot),
            llm_ratio=args.llm_ratio,
        )

        print(f"  → 耗时 {time.time() - t:.1f}s")

    else:
        # 跳过 Step3，从缓存加载
        if not cache_cot.exists():
            print(f"\n[ERROR] 跳过 Step3 但缓存文件不存在：{cache_cot}")
            print("请先运行 Step3 或去掉 --steps 参数。")
            sys.exit(1)
        with open(cache_cot, encoding='utf-8') as f:
            cot_dict = json.load(f)
        print(f"\n[跳过 Step3] 从缓存加载 {len(cot_dict)} 条 CoT")

    # ----------------------------------------------------------
    # Step 4: 数据集组装
    # ----------------------------------------------------------
    if 4 in args.steps:
        print_step(4, "数据集组装（ShareGPT 格式）")
        t = time.time()

        from step4_build_dataset import build_sft_dataset, validate_dataset

        train_path, val_path = build_sft_dataset(
            valid_samples=valid_samples,
            cot_dict=cot_dict,
            output_path=str(output_dir),
            seed=args.seed,
        )

        # 格式校验
        print("\n格式校验：")
        train_ok = validate_dataset(str(train_path))
        val_ok   = validate_dataset(str(val_path))

        if not (train_ok and val_ok):
            print("\n[WARNING] 部分样本格式异常，请检查上方错误信息。")

        print(f"  → 耗时 {time.time() - t:.1f}s")

    # ----------------------------------------------------------
    # 完成总结
    # ----------------------------------------------------------
    print_banner("Pipeline 完成")
    print(f"  总耗时 : {time.time() - t_total:.1f}s")
    print(f"  输出目录: {output_dir.resolve()}")
    print()

    if 4 in args.steps:
        print("  生成文件：")
        for p in sorted(output_dir.glob("*.json")):
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"    {p.name:<40} {size_mb:.1f} MB")

        montage_count = len(list(montage_dir.rglob("*.jpg"))) if montage_dir.exists() else 0
        if montage_count:
            print(f"\n  Montage 图片：{montage_count} 张（{montage_dir}）")

    print()
    print("  下一步：将生成的 JSON 文件注册到 LLaMA-Factory 的 dataset_info.json，")
    print("  然后运行 SFT 训练。")


# ============================================================
# SECTION 3: 入口
# ============================================================

if __name__ == '__main__':
    main()