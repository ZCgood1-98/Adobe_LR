#!/usr/bin/env bash
# 线5 · 训练环境安装：独立 conda env（不动现有 Chen 环境，避免破坏拟合流水线）
# 用法：bash 5_training/setup_env.sh
#       DOWNLOAD_MODEL=1 bash 5_training/setup_env.sh   # 顺带预下载模型(~17GB)
set -euo pipefail

ENV_NAME="${ENV_NAME:-lf}"
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# llamafactory 自带 transformers/accelerate 版本钳制（0.9.5: transformers≤5.6.0,
# accelerate≤1.11.0），装完别再单独 -U 升级这两个包，会破坏兼容性
pip install -U "llamafactory[metrics]>=0.9.4"
# 诊断脚本 score 阶段跑 opencv_renderer，需要 contrib 版（cv2.ximgproc 导向滤波）
pip install -U opencv-contrib-python

# 验证：Qwen3-VL 模型类 + llamafactory qwen3_vl 模板都就位
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration  # noqa: F401
print("Qwen3VLForConditionalGeneration OK")
from llamafactory.data.template import TEMPLATES
assert "qwen3_vl" in TEMPLATES, \
    f"qwen3_vl 模板缺失，装到的版本太旧。可用 qwen 系模板: {sorted(k for k in TEMPLATES if 'qwen' in k)}"
print("llamafactory qwen3_vl 模板 OK")
PY

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
    hf download Qwen/Qwen3-VL-8B-Instruct   # huggingface-cli 已废弃，用 hf
fi
echo "[setup_env] 完成。训练：bash 5_training/train_sft.sh"
