"""WB 中性一刀:拿 diag_grpo 的 light 预测,把色温(可选色调)清中性重打分,看 light 改善中位是否回正。"""
import json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, '5_training')
from v2_codec import decode_stage
from build_sft_dataset import extract_stages, STAGE_NAMES, _imread_rgb, _resize_max_side
from differentiable_renderer.opencv_renderer import render
from differentiable_renderer.losses import CompositeLoss

REPO = Path('.')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
lc = CompositeLoss(weights={'param_reg': 0.0}).to(dev)
lbw = CompositeLoss(weights={**CompositeLoss.BW_LOSS_WEIGHTS, 'param_reg': 0.0}).to(dev)
to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

def dist(a, b, bw):
    torch.manual_seed(0)
    with torch.no_grad():
        return float((lbw if bw else lc)(to_t(a), to_t(b))[0])

preds = [json.loads(l) for l in open('sft_data_v2/diag_grpo/predictions.jsonl')
         if json.loads(l)['stage'] == 'light']
variants = {'as-is (对照)': lambda z: z,
            'WB色温中性': lambda z: {**z, '色温': 5500.0},
            'WB色温+色调中性': lambda z: {**z, '色温': 5500.0, '色调': 0.0}}
imp = {k: [] for k in variants}
for p in preds:
    key, vn = p['id'].rsplit('/', 1)[0].rsplit('/v', 1)
    vdir = REPO / 'fit_results' / key / f'version_{vn}'
    _, zh = decode_stage('light', re.findall(r'<[^>]+>', p['raw']))
    gt_zh = extract_stages(json.loads((vdir / 'params.json').read_text()))[STAGE_NAMES.index('light')][0]
    in_png = vdir / 'original_cropped.png'
    if not in_png.exists(): in_png = REPO / 'fit_results' / key / 'original_captured_cropped.png'
    img0 = _resize_max_side(_imread_rgb(str(in_png)), 1024)
    bw = bool(json.loads((vdir / 'fit_meta.json').read_text()).get('bw_mode'))
    gt_img = render(img0, gt_zh); Lnoop = dist(render(img0, {}), gt_img, bw)
    for k, f in variants.items():
        imp[k].append(1 - dist(render(img0, f(zh)), gt_img, bw) / max(Lnoop, 1e-6))

print(f'light n={len(preds)}  (as-is 应≈诊断的 -0.044,验证脚本对齐)')
for k in variants:
    v = imp[k]
    print(f'  {k:18} 改善中位 {np.median(v):+.3f}   均 {np.mean(v):+.3f}   >0比例 {np.mean([x>0 for x in v]):.0%}')
