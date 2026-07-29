#!/usr/bin/env bash
# 线5 · SFT 启动脚本（单卡）。支持追加 llamafactory 覆盖参数，如：
#   bash 5_training/train_sft.sh max_samples=64 num_train_epochs=0.01   # 冒烟
set -euo pipefail
cd "$(dirname "$0")/.."   # 仓库根目录（数据路径都是相对根目录的）

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-lf}"

llamafactory-cli train 5_training/sft_lora_qwen3vl8b.yaml "$@"
