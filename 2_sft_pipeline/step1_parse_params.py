import json
from pathlib import Path

def parse_param_value(value_str):
    """处理 Unicode 负号、百分号等特殊字符"""
    if value_str is None:
        return None
    s = str(value_str).strip()
    s = s.replace('\u2212', '-').replace('\u2013', '-')
    s = s.rstrip('%')
    try:
        return float(s)
    except ValueError:
        return None

def extract_params(details_json_path):
    with open(details_json_path, encoding='utf-8') as f:
        data = json.load(f)

    params = {}

    for item in data.get('details', []):
        t = item.get('type')

        if t == 'standard':
            label = item.get('label', '')
            value = parse_param_value(item.get('value'))
            if value is not None:
                params[label] = value

        elif t == 'color_mixer':
            # 颜色混合器：保存每个颜色通道的 H/S/L
            color_style = item.get('color_style', '')
            values = item.get('values', {})
            parsed = {k: parse_param_value(v) for k, v in values.items()}
            # 用 color_style 作为 key（或者按顺序命名）
            params.setdefault('color_mixer', []).append({
                'color': color_style,
                'H': parsed.get('H'),
                'S': parsed.get('S'),
                'L': parsed.get('L'),
            })

    return params


# 测试一下
if __name__ == '__main__':
    sample = Path('scraped_data_v4/train/00bd4db5-9188-4462-a32f-eb7e3a4c594d/version_0/details.json')
    params = extract_params(sample)
    print(json.dumps(params, ensure_ascii=False, indent=2))