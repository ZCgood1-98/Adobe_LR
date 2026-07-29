from pathlib import Path
import json
from step1_parse_params import extract_params

# 归一化分母：对大多数参数是单侧最大值；色温单独处理（见下方）
PARAM_RANGES = {
    '曝光度': 5.0,    '对比度': 100.0,  '高光': 100.0,
    '阴影': 100.0,    '白色色阶': 100.0, '黑色色阶': 100.0,
    '色调': 150.0,    '自然饱和度': 100.0,
    '饱和度': 100.0,  '去朦胧': 100.0,  '晕影': 100.0,
    '清晰度': 100.0,  '纹理': 100.0,    '锐化': 150.0,
    '噪点消除': 100.0,
}

# 具有非零中性参考值的参数：归一化时以 abs(val - neutral) / max_deviation 计算
# 色温：中性日光 5500 K，最大偏移取两侧较大值 max(5500-2000, 10000-5500) = 4500 K
PARAM_NEUTRAL = {
    '色温': (5500, 4500),   # (neutral_K, max_deviation_K)
}

# 归一化后的重要性权重（反映视觉影响力）
IMPORTANCE = {
    '曝光度': 1.5,    '对比度': 0.8,   '高光': 1.0,
    '阴影': 1.0,      '白色色阶': 0.6, '黑色色阶': 0.6,
    '色温': 1.2,      '色调': 0.5,     '自然饱和度': 0.8,
    '饱和度': 0.8,    '去朦胧': 1.3,   '晕影': 0.5,
    '清晰度': 0.9,    '纹理': 0.6,     '锐化': 0.4,
    '噪点消除': 0.3,
}

def compute_edit_magnitude(params):
    score = 0.0
    for key in list(PARAM_RANGES) + list(PARAM_NEUTRAL):
        val = params.get(key)
        if not isinstance(val, (int, float)):
            continue
        if key in PARAM_NEUTRAL:
            neutral, max_dev = PARAM_NEUTRAL[key]
            normalized = abs(val - neutral) / max_dev   # 偏离中性色温的比例
        else:
            normalized = abs(val) / PARAM_RANGES[key]  # 归一化到 [0,1]
        score += normalized * IMPORTANCE[key]
    return score

def collect_valid_samples(dataset_root, split='train', min_magnitude=0.5):
    root = Path(dataset_root) / split
    valid = []
    skipped = 0

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue

        original = sample_dir / 'original_captured_cropped.png'
        v0_img   = sample_dir / 'version_0' / 'main_image.png'
        v0_json  = sample_dir / 'version_0' / 'details.json'

        if not all(p.exists() for p in [original, v0_img, v0_json]):
            skipped += 1
            continue

        params = extract_params(v0_json)
        magnitude = compute_edit_magnitude(params)

        if magnitude < min_magnitude:
            skipped += 1
            continue

        valid.append({
            'sample_id': sample_dir.name,
            'original': str(original),
            'edited': str(v0_img),
            'params': params,
            'magnitude': magnitude,
        })

    print(f"有效样本: {len(valid)}，跳过: {skipped}")
    return valid