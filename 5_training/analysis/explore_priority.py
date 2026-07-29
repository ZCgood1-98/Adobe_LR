"""每个标量参数:桶用量(探索广度) + 幅度比(保守度) + r(感知),判谁该加强探索。"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '5_training')
from v2_codec import value_to_bin, decode_stage, SLOTS, slot_nbins
from build_sft_dataset import extract_stages, STAGE_NAMES

REPO = Path('.')
scalar = {s.en_path[0]: s for s in SLOTS if len(s.en_path) == 1}   # 标量槽
data = defaultdict(lambda: {'pv': [], 'gv': [], 'pb': [], 'gb': []})
for l in open('sft_data_v2/diag_grpo/predictions.jsonl'):
    p = json.loads(l); st = p['stage']
    key, vn = p['id'].rsplit('/', 1)[0].rsplit('/v', 1)
    pen, _ = decode_stage(st, re.findall(r'<[^>]+>', p['raw']))
    gen = extract_stages(json.loads((REPO / 'fit_results' / key / f'version_{vn}' / 'params.json').read_text()))[STAGE_NAMES.index(st)][1]
    for ek, slot in scalar.items():
        if slot.stage != st or ek not in pen or ek not in gen: continue
        data[ek]['pv'].append(float(pen[ek])); data[ek]['gv'].append(float(gen[ek]))
        data[ek]['pb'].append(value_to_bin(slot, float(pen[ek])))
        data[ek]['gb'].append(value_to_bin(slot, float(gen[ek])))

rows = []
for ek, slot in scalar.items():
    d = data[ek]
    if len(d['pv']) < 10: continue
    pv, gv = np.array(d['pv']), np.array(d['gv'])
    pd_, gd_ = pv - slot.neutral, gv - slot.neutral
    nbp, nbg, nb = len(set(d['pb'])), len(set(d['gb'])), slot_nbins(slot)
    mv = np.abs(gd_) > (0.05 * np.median(np.abs(gd_[gd_ != 0])) if (gd_ != 0).any() else 1)
    mag = np.median(np.abs(pd_[mv])) / max(np.median(np.abs(gd_[mv])), 1e-9) if mv.sum() > 3 else 0.0
    r = float(np.corrcoef(pd_, gd_)[0, 1]) if pd_.std() > 1e-9 and gd_.std() > 1e-9 else 0.0
    rows.append((slot.stage, ek, nbp, nbg, nb, mag, r))

def verdict(nbp, nbg, mag, r):
    if r > 0.85: return 'A 已好·别动(保护)'
    if r < 0.20: return 'C 弱信号·探索难救(感知/低杠杆)'
    if nbp < nbg * 0.6 or mag < 0.6: return 'B ⭐感知有+探索不足→该加强'
    return 'B 感知有·尚可'

print(f"{'stage/参数':20} {'桶 模型/GT/共':>14} {'幅度比':>6} {'r':>6}  判读")
for st, ek, nbp, nbg, nb, mag, r in sorted(rows, key=lambda x: (x[6] < 0.2, x[6] < 0.85, -(x[6] * (1 - x[2] / max(x[3], 1))))):
    print(f"{st + '/' + ek:20} {f'{nbp}/{nbg}/{nb}':>14} {mag:>6.2f} {r:>+6.2f}  {verdict(nbp, nbg, mag, r)}")
