"""为对比网页准备素材：原图 / 各模型输出 / 差异图（ΔL 亮度、ΔC 彩度，发散色图）+ data.json。

用法：python build_viewer.py --out viewer_build --tags PhotoCraft_claude-haiku-5-5 PhotoCraft_gpt-6-luna
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path('/home/chen/Work/adobe_lr/7_baselines/runs/A')
LONG = 1024      # 网页图长边
DL_RANGE = 15.0  # ΔL 色图满量程（Lab L 单位）
DC_RANGE = 15.0  # ΔC 色图满量程


def srgb_to_lab(rgb):
    """rgb: HxWx3 uint8 → Lab（D65）。"""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def diverging(d, rng):
    """有符号差值 → 发散色图：负=蓝，0=中性灰，正=橙。"""
    t = np.clip(d / rng, -1, 1)[..., None]
    neg = np.array([40, 110, 210.0])
    mid = np.array([128, 128, 128.0])
    pos = np.array([235, 120, 30.0])
    out = np.where(t < 0, mid + (neg - mid) * (-t), mid + (pos - mid) * t)
    return out.astype(np.uint8)


def fit(im):
    s = LONG / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--keys', nargs='+', default=['claude', 'codex'], help='网页里各模型的短名')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'img').mkdir(parents=True, exist_ok=True)

    sets = [{p.name for p in (BASE / t / a.split).glob('*.jpg')} for t in a.tags]
    files = sorted(set.intersection(*sets))
    items = []
    for f in files:
        u = Path(f).stem
        src = fit(Image.open(BASE / '_inputs' / a.split / f'{u}.png').convert('RGB'))
        src.save(out / 'img' / f'{u}_in.jpg', quality=90)
        lab0 = srgb_to_lab(np.asarray(src))
        c0 = np.hypot(lab0[..., 1], lab0[..., 2])
        it = {'id': u, 'w': src.width, 'h': src.height, 'models': {}}
        for t, k in zip(a.tags, a.keys):
            im = fit(Image.open(BASE / t / a.split / f).convert('RGB')).resize(src.size, Image.LANCZOS)
            im.save(out / 'img' / f'{u}_{k}.jpg', quality=90)
            lab = srgb_to_lab(np.asarray(im))
            dL = lab[..., 0] - lab0[..., 0]
            dC = np.hypot(lab[..., 1], lab[..., 2]) - c0
            dE = np.linalg.norm(lab - lab0, axis=-1)
            Image.fromarray(diverging(dL, DL_RANGE)).save(out / 'img' / f'{u}_{k}_dL.jpg', quality=88)
            Image.fromarray(diverging(dC, DC_RANGE)).save(out / 'img' / f'{u}_{k}_dC.jpg', quality=88)
            meta = BASE / t / '_meta' / a.split / u
            pj = json.loads((meta / 'params.json').read_text())
            au = json.loads((meta / 'audit.json').read_text())
            it['models'][k] = {
                'params': pj['params'], 'rationale': pj.get('rationale', ''),
                'seconds': au['seconds'], 'filters': au['n_filter'], 'previews': au['n_preview'],
                'dL': round(float(dL.mean()), 2), 'dC': round(float(dC.mean()), 2),
                'dE': round(float(dE.mean()), 2), 'dE_p95': round(float(np.percentile(dE, 95)), 2),
            }
        # 两个模型之间的差异
        if len(a.keys) == 2:
            l1 = srgb_to_lab(np.asarray(Image.open(out / 'img' / f'{u}_{a.keys[0]}.jpg')))
            l2 = srgb_to_lab(np.asarray(Image.open(out / 'img' / f'{u}_{a.keys[1]}.jpg')))
            it['dE_between'] = round(float(np.linalg.norm(l1 - l2, axis=-1).mean()), 2)
        items.append(it)
        print(u, {k: (v['dL'], v['dC'], v['dE']) for k, v in it['models'].items()}, it.get('dE_between'))
    data = {'split': a.split, 'tags': dict(zip(a.keys, a.tags)), 'dL_range': DL_RANGE, 'dC_range': DC_RANGE, 'items': items}
    (out / 'data.json').write_text(json.dumps(data, ensure_ascii=False))
    print('→', out)


if __name__ == '__main__':
    main()
