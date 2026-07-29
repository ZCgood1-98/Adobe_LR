#!/usr/bin/env python3
"""
step4_build_dataset.py
======================
将 Step1-3 的产出组装成 LLaMA-Factory ShareGPT 格式的训练文件。

输入：
  - valid_samples : Step2 collect_valid_samples() 的输出列表
  - cot_dict      : Step3 generate_all_cot() 的输出字典
                    格式：{sample_id: {"cot": str, "montage_paths": dict, ...}}

输出：
  - sft_data/photo_editing_train.json
  - sft_data/photo_editing_val.json

数据格式：LLaMA-Factory ShareGPT 多模态格式
  每条样本包含：
    - messages: [system, user, assistant]
    - images  : [原图路径, montage图1路径, montage图2路径, ...]

对外接口（供 run_pipeline.py 调用）：
  build_sft_dataset(valid_samples, cot_dict, output_path, seed=42)
  → 返回 (train_path, val_path)
"""

import json
import random
from pathlib import Path

# 从 step3 导入 SWEEP_PARAMS，保证图片顺序与 CoT 文字顺序一致
from step3_assr_cot import SWEEP_PARAMS

# ============================================================
# SECTION 0: 全局常量
# ============================================================

SYSTEM_PROMPT = """You are a professional photo retouching artist. Given an original photograph, reason through the ASSR framework and output the adjustment parameters:

1. Analysis: diagnose the main visual issues in the image
2. Sweep & Select: sweep candidate values for key parameters and select the visually optimal value
3. Refine: check for coupling interactions between parameters and make corrections

Complete your reasoning inside <think> tags, then output valid JSON parameters. Do not add any extra explanation.

Parameter ranges:
- exposure: [-5, +5] (float)
- contrast / highlights / shadows / whites / blacks: [-100, +100]
- temperature / tint: [-100, +100] / [-150, +150]
- vibrance / saturation / dehaze / clarity / texture: [-100, +100]
- vignette: [-100, 0], sharpening: [0, 150], noise_reduction: [0, 100]"""

# 多种 user 指令，增加多样性（有 montage 时使用专用版本）
USER_INSTRUCTIONS_NO_MONTAGE = [
    "Analyze this original photo and output the optimal Lightroom adjustment parameters.",
    "As a professional retoucher, devise a post-processing plan for this photo and output the parameters.",
    "Diagnose the issues in this photo and provide Lightroom parameter recommendations.",
    "This photo needs post-processing. Analyze it and output the retouching parameters.",
    "Evaluate the tonality and color of this photo and output the optimized parameters.",
]

USER_INSTRUCTION_WITH_MONTAGE = (
    "The first image is the original photo. "
    "The remaining images are individual candidate thumbnails for each key parameter, "
    "grouped by parameter and sorted low-to-high in value "
    "(the green border marks the finally selected value). "
    "Analyze them and output the optimal Lightroom adjustment parameters."
)

# 中文参数名 → 英文 key 的映射（JSON 输出用英文 key）
PARAM_KEY_MAP = {
    '曝光度':    'exposure',
    '对比度':    'contrast',
    '高光':      'highlights',
    '阴影':      'shadows',
    '白色色阶':  'whites',
    '黑色色阶':  'blacks',
    '色温':      'temperature',
    '色调':      'tint',
    '自然饱和度':'vibrance',
    '饱和度':    'saturation',
    '去朦胧':    'dehaze',
    '清晰度':    'clarity',
    '纹理':      'texture',
    '晕影':      'vignette',
    '锐化':      'sharpening',
    '噪点消除':  'noise_reduction',
    '颗粒':      'grain',
}


# ============================================================
# SECTION 1: 参数字典 → 标准化 JSON 字符串
# ============================================================

def params_to_json_str(params: dict) -> str:
    """
    将 extract_params() 返回的中文参数字典转为标准化英文 JSON 字符串。
    - 曝光度保留两位小数
    - 其余参数取整
    - color_mixer 单独处理（暂不纳入输出，避免结构复杂化）
    - sort_keys=True 保证每次输出顺序一致
    """
    output = {}
    for cn, en in PARAM_KEY_MAP.items():
        if cn in params:
            val = params[cn]
            if isinstance(val, (int, float)):
                output[en] = round(val, 2) if en == 'exposure' else int(round(val))

    return json.dumps(output, ensure_ascii=False, sort_keys=True)


# ============================================================
# SECTION 2: 单条训练样本构建
# ============================================================

def _build_entry(sample: dict,
                 cot_text: str,
                 montage_paths: dict,
                 params_json: str,
                 rng: random.Random) -> dict:
    """
    构建单条 ShareGPT 格式训练样本。

    有 montage 时（多图输入）：
      images = [原图, montage_1, montage_2, ...]
      user   = "<image><image>...\n说明文字"

    无 montage 时（单图输入）：
      images = [原图]
      user   = "<image>\n随机指令"

    注意：user 消息中 <image> tag 的数量必须与 images 数组长度完全一致，
    否则 LLaMA-Factory 会在数据加载阶段报错。
    """
    assistant_content = (
        f"<think>\n{cot_text}\n</think>\n\n{params_json}"
        if cot_text else params_json
    )

    # 按 SWEEP_PARAMS 固定顺序展开各参数的候选图路径列表
    # montage_paths[label] 现在是 list[str]（按升序排列的独立缩略图路径）
    valid_montages = []
    for label in SWEEP_PARAMS:
        if label in montage_paths:
            valid_montages.extend(montage_paths[label])

    if valid_montages:
        # 多图模式：原图 + 各参数所有候选缩略图，<image> 之间用换行分隔
        image_list = [sample['original']] + valid_montages
        image_tags = "\n".join("<image>" for _ in image_list)
        user_text = f"{image_tags}\n{USER_INSTRUCTION_WITH_MONTAGE}"
    else:
        # 单图模式（该样本没有任何参数幅度达到渲染阈值）
        image_list = [sample['original']]
        user_text = f"<image>\n{rng.choice(USER_INSTRUCTIONS_NO_MONTAGE)}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": image_list,
    }


# ============================================================
# SECTION 3: 批量构建并保存数据集
# ============================================================

def build_sft_dataset(valid_samples: list,
                      cot_dict: dict,
                      output_path: str,
                      seed: int = 42) -> tuple[Path, Path]:
    """
    将所有有效样本组装成训练/验证集并保存。

    参数：
      valid_samples : Step2 的输出，列表元素含 sample_id / original / params
      cot_dict      : Step3 generate_all_cot() 的完整返回字典
                      格式：{sample_id: {"cot": str, "montage_paths": dict, ...}}
      output_path   : 输出目录路径（不存在则自动创建）
      seed          : 随机种子（控制 train/val 划分和指令多样性）

    返回：
      (train_path, val_path)
    """
    rng = random.Random(seed)
    dataset = []
    skipped = 0

    for sample in valid_samples:
        sid = sample['sample_id']
        params_json = params_to_json_str(sample['params'])

        # 从 cot_dict 取出该样本的 cot 文字和 montage 路径
        cot_entry = cot_dict.get(sid, {})
        cot_text     = cot_entry.get('cot', '')
        montage_paths = cot_entry.get('montage_paths', {})

        # 跳过参数转换失败的样本
        if not params_json or params_json == '{}':
            skipped += 1
            continue

        entry = _build_entry(sample, cot_text, montage_paths, params_json, rng)
        dataset.append(entry)

    if skipped:
        print(f"[step4] 跳过 {skipped} 个参数为空的样本")

    # 打乱顺序（固定 seed 保证可复现）
    rng.shuffle(dataset)

    # 按 9:1 划分 train / val
    split_idx  = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    val_data   = dataset[split_idx:]

    # 写入文件
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / 'photo_editing_train.json'
    val_path   = out_dir / 'photo_editing_val.json'

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    # 统计多图 vs 单图样本数量
    multi_img = sum(1 for e in dataset if len(e['images']) > 1)
    single_img = len(dataset) - multi_img

    print(f"\n[step4] 数据集构建完成")
    print(f"  总样本数  : {len(dataset)}")
    print(f"  多图样本  : {multi_img}（含 montage）")
    print(f"  单图样本  : {single_img}（无 montage）")
    print(f"  训练集    : {len(train_data)} 条 → {train_path}")
    print(f"  验证集    : {len(val_data)} 条 → {val_path}")

    return train_path, val_path


# ============================================================
# SECTION 4: 格式校验（可选，运行前检查数据是否合法）
# ============================================================

def validate_dataset(json_path: str, max_check: int = 20) -> bool:
    """
    快速校验生成的数据集是否满足 LLaMA-Factory 的格式要求。
    主要检查：<image> tag 数量与 images 数组长度是否一致。
    返回 True 表示全部通过。
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    errors = []
    for i, entry in enumerate(data[:max_check]):
        images = entry.get('images', [])
        # 统计 user 消息中 <image> 的数量
        user_content = next(
            (m['content'] for m in entry['messages'] if m['role'] == 'user'), ''
        )
        tag_count = user_content.count('<image>')

        if tag_count != len(images):
            errors.append(
                f"  第 {i} 条：<image> tag 数={tag_count}，"
                f"images 数组长度={len(images)}"
            )

    if errors:
        print(f"[validate] ❌ 发现 {len(errors)} 条格式错误（检查前 {max_check} 条）：")
        for e in errors:
            print(e)
        return False

    print(f"[validate] ✅ 格式校验通过（检查前 {min(max_check, len(data))} 条）")
    return True


# ============================================================
# SECTION 5: 独立测试入口
# ============================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from step1_parse_params import extract_params
    from step2_filter import collect_valid_samples
    from step3_assr_cot import generate_all_cot

    # 用少量样本跑通完整 pipeline
    print("=== Step2: 收集有效样本 ===")
    samples = collect_valid_samples(
        dataset_root='scraped_data_v4',
        split='test',
        min_magnitude=0.5,
    )[:5]   # 只取 5 条做测试

    print("\n=== Step3: 生成 CoT ===")
    cot_dict = generate_all_cot(
        valid_samples=samples,
        output_cache_path='sft_data/cot_cache_test.json',
        gpt4o_ratio=0.0,    # 测试时不调 GPT-4o
    )

    print("\n=== Step4: 构建数据集 ===")
    train_path, val_path = build_sft_dataset(
        valid_samples=samples,
        cot_dict=cot_dict,
        output_path='sft_data_test',
        seed=42,
    )

    print("\n=== 格式校验 ===")
    validate_dataset(str(train_path))