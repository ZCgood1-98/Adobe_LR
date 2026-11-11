"""汇总 PhotoCraft 智能体基线：对照拼图（原图 | 各模型）+ 运行统计表。

用法：python summarize.py --split test --tags PhotoCraft_claude-haiku-5-5 PhotoCraft_gpt-6-luna --out sheet.jpg
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path('/home/chen/Work/adobe_lr/7_baselines/runs/A')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--row_h', type=int, default=300)
    a = ap.parse_args()

    # 以所有模型都有 audit 的图为准
    sets = [{p.name for p in (BASE / t / '_meta' / a.split).iterdir() if (p / 'audit.json').exists()} for t in a.tags]
    uuids = sorted(set.intersection(*sets))

    # —— 统计表 ——
    for t in a.tags:
        rows = [json.loads((BASE / t / '_meta' / a.split / u / 'audit.json').read_text()) for u in uuids]
        ok = sum(r['status'] == 'ok' for r in rows)
        viol = sum(len(r['violations']) for r in rows)
        sec = [r['seconds'] for r in rows]
        nf = [r['n_filter'] for r in rows]
        print(f'{t}: ok {ok}/{len(rows)}  越界调用 {viol}  耗时 中位 {sorted(sec)[len(sec)//2]:.0f}s 最大 {max(sec):.0f}s  '
              f'filter 次数 {nf}')
        # 主要参数的平均绝对幅度，看是否过于保守
        keys = ['exposure', 'temperature', 'contrast', 'highlights', 'shadows', 'whites', 'blacks', 'vibrance', 'saturation', 'clarity']
        ps = [json.loads((BASE / t / '_meta' / a.split / u / 'params.json').read_text())['params']
              for u in uuids if (BASE / t / '_meta' / a.split / u / 'params.json').exists()]
        print('   平均|参数|: ' + '  '.join(f'{k} {sum(abs(p.get(k, 0)) for p in ps) / max(len(ps), 1):.2f}' for k in keys))

    # —— 对照拼图 ——
    cols = ['input'] + a.tags  # PIL 默认字体不含中文，表头用英文
    tiles = []
    for u in uuids:
        paths = [BASE / '_inputs' / a.split / f'{u}.png'] + [BASE / t / a.split / f'{u}.jpg' for t in a.tags]
        row = []
        for p in paths:
            im = Image.open(p).convert('RGB') if p.exists() else Image.new('RGB', (a.row_h, a.row_h), 'gray')
            row.append(im.resize((int(im.width * a.row_h / im.height), a.row_h)))
        tiles.append(row)
    col_w = [max(r[i].width for r in tiles) for i in range(len(cols))]
    W, H = sum(col_w) + 10 * len(cols), (a.row_h + 10) * len(tiles) + 30
    sheet = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(sheet)
    x = 0
    for i, c in enumerate(cols):
        d.text((x + 5, 8), c.replace('PhotoCraft_', ''), fill='black')
        x += col_w[i] + 10
    for j, row in enumerate(tiles):
        x = 0
        for i, im in enumerate(row):
            sheet.paste(im, (x, 30 + j * (a.row_h + 10)))
            x += col_w[i] + 10
    sheet.save(a.out, quality=90)
    print('拼图 →', a.out)


if __name__ == '__main__':
    main()
