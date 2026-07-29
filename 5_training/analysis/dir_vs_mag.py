import json, sys, re
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '5_training')
from v2_codec import decode_stage
from build_sft_dataset import extract_stages, STAGE_NAMES
from diagnose_stage_error import SCALAR_EN2ZH

NEUTRAL = defaultdict(float); NEUTRAL['temperature'] = 5500.0
preds = [json.loads(l) for l in open('sft_data_v2/diag_v2/predictions.jsonl')]

data = defaultdict(lambda: defaultdict(list))    # [stage][param] -> [(pred_delta, gt_delta)]
for p in preds:
    st = p['stage']
    key, vn = p['id'].rsplit('/', 1)[0].rsplit('/v', 1)
    toks = re.findall(r'<[^>]+>', p['raw'])
    pred_en, _ = decode_stage(st, toks)
    gt_en = extract_stages(json.loads(Path('fit_results', key, f'version_{vn}', 'params.json').read_text()))[STAGE_NAMES.index(st)][1]
    for k in SCALAR_EN2ZH[st]:
        data[st][k].append((float(pred_en[k]) - NEUTRAL[k], float(gt_en[k]) - NEUTRAL[k]))

print(f"{'stage/param':20} {'n动':>5} {'方向一致':>7} {'相关r':>6} {'幅度比':>6} {'斜率':>6}  判读")
agg_dir, agg_r = [], []
for st in STAGE_NAMES:
    for k in SCALAR_EN2ZH[st]:
        pd = np.array([a for a, b in data[st][k]]); gd = np.array([b for a, b in data[st][k]])
        scale = np.median(np.abs(gd[gd != 0])) if (gd != 0).any() else 0
        dz = 0.05 * scale
        mv = np.abs(gd) > max(dz, 1e-9)
        if mv.sum() < 8:
            continue
        dir_agree = float((np.sign(pd[mv]) == np.sign(gd[mv])).mean())
        r = float(np.corrcoef(pd, gd)[0, 1]) if pd.std() > 1e-9 and gd.std() > 1e-9 else float('nan')
        mag = float(np.median(np.abs(pd[mv])) / max(np.median(np.abs(gd[mv])), 1e-9))
        slope = float((pd * gd).sum() / max((gd * gd).sum(), 1e-9))
        verdict = ('方向对/幅度保守' if (dir_agree > .65 and r > .3 and mag < .7)
                   else '方向对/校准可' if (dir_agree > .65 and r > .3)
                   else '弱信号' if (dir_agree > .58 or r > .2)
                   else '~随机(没跟)')
        agg_dir.append(dir_agree); agg_r.append(r)
        print(f"{st + '/' + k:20} {int(mv.sum()):>5} {dir_agree:>6.0%} {r:>6.2f} {mag:>6.2f} {slope:>6.2f}  {verdict}")

print(f"\n[汇总] 主要标量参数: 方向一致均值={np.mean(agg_dir):.0%}  相关r均值={np.mean(agg_r):.2f}  (方向一致 50%=随机, r 0=没跟)")
# 曝光单独看(最高杠杆维,v1 曾 r=0.65)
pd = np.array([a for a, b in data['light']['exposure']]); gd = np.array([b for a, b in data['light']['exposure']])
print(f"[曝光] r={np.corrcoef(pd,gd)[0,1]:.2f}  pred标准差={pd.std():.3f}EV  GT标准差={gd.std():.3f}EV  (pred≪GT=没敢动)")
