"""查色温分桶到底是不是根因:往返误差 + GT分布/桶占用 + 桶空间相关(定位分桶vs感知)。"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '5_training')
from v2_codec import value_to_bin, bin_to_value, _SLOT_BY_TOKEN, decode_stage
from build_sft_dataset import extract_stages, STAGE_NAMES

REPO = Path('.')
slot_t = _SLOT_BY_TOKEN['<temperature>']; NEU = 5500

# 1) 连续 100K 步长 → 桶(直接看用户的质疑:100倍数分得开吗)
print('=== 100K 步长映射到的桶(中性附近) ===')
for k in [5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000]:
    print(f'  {k}K → 桶 {value_to_bin(slot_t, k)}', end='  ')
    if k % 300 == 0: print()
print()

# 2) 从数据抠 GT + 模型预测的色温
preds = [json.loads(l) for l in open('sft_data_v2/diag_grpo/predictions.jsonl')
         if json.loads(l)['stage'] == 'light']
gt_t, pr_t, gt_b, pr_b, rt = [], [], [], [], []
for p in preds:
    key, vn = p['id'].rsplit('/', 1)[0].rsplit('/v', 1)
    gt = float(extract_stages(json.loads((REPO / 'fit_results' / key / f'version_{vn}' / 'params.json').read_text()))[0][1]['temperature'])
    pr = float(decode_stage('light', re.findall(r'<[^>]+>', p['raw']))[0]['temperature'])
    gt_t.append(gt); pr_t.append(pr)
    b = value_to_bin(slot_t, gt); gt_b.append(b); pr_b.append(value_to_bin(slot_t, pr))
    rt.append(abs(bin_to_value(slot_t, b) - gt))
gt_t, pr_t, gt_b, pr_b, rt = map(np.array, (gt_t, pr_t, gt_b, pr_b, rt))

print(f'\n=== GT 色温分布 (n={len(gt_t)}) ===')
print(f'  范围[{gt_t.min():.0f},{gt_t.max():.0f}] 中位{np.median(gt_t):.0f} 中位|Δ中性|={np.median(np.abs(gt_t-NEU)):.0f}K')
print(f'  专家真动WB(|Δ|>200K)占比 {(np.abs(gt_t-NEU)>200).mean():.0%}')
print(f'\n=== 分桶质量 ===')
print(f'  往返误差 encode→decode: 中位{np.median(rt):.1f}K 最大{np.max(rt):.0f}K  (小=编码没丢信息)')
u, c = np.unique(gt_b, return_counts=True)
print(f'  GT 用到 {len(u)}/63 个桶, 中性桶={value_to_bin(slot_t,5500)}')
print(f'  GT 桶Top5(桶:样本数): {sorted(zip(u.tolist(),c.tolist()), key=lambda x:-x[1])[:5]}')
print(f'\n=== 模型预测 vs GT (定位:分桶精度 还是 感知?) ===')
print(f'  值空间相关 r = {np.corrcoef(pr_t-NEU, gt_t-NEU)[0,1]:+.3f}')
print(f'  桶空间相关 r = {np.corrcoef(pr_b, gt_b)[0,1]:+.3f}   ← 桶都对不上=不是分桶精度、是感知/信号')
print(f'  方向一致(GT动了时) = {(np.sign(pr_t[np.abs(gt_t-NEU)>50]-NEU)==np.sign(gt_t[np.abs(gt_t-NEU)>50]-NEU)).mean():.0%}  (50%=瞎猜)')
print(f'  pred 色温:中位{np.median(pr_t):.0f} 中位|Δ|={np.median(np.abs(pr_t-NEU)):.0f}K 用到{len(set(pr_b.tolist()))}/63桶(=模型敢不敢动WB)')
