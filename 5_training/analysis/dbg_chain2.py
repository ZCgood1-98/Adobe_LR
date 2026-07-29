"""隔离:全链崩塌是「输入图跑偏(exposure bias)」还是「模型参数组合本身坏」?
GT链 : GT中间图 + GT参数            → D4=0 (天花板参考,也看 D_t 是否单调)
A 自渲: 模型自渲中间图 + 模型参数     → 现状
B 教师图: **GT中间图** + 模型参数(全合成) → 输入分布正确,只测参数组合
A vs B 差 = exposure bias 的净代价;B vs GT 差 = 模型参数组合本身的差距。
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path('/home/chen/Work/adobe_lr')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / '5_training'))
from v2_codec import SLOTS, VALUE_TOKENS_V, VALUE_TOKENS_EV
from v2_grpo import stage_temps, load_policy, sample_rollout
from v2_grpo_chain import load_units, prep_unit, build_stage_inputs, to_pil, chain_rollout
from build_sft_dataset import extract_stages, STAGE_NAMES
from differentiable_renderer.opencv_renderer import render
from differentiable_renderer.losses import CompositeLoss

model, proc = load_policy('Qwen/Qwen3-VL-8B-Instruct', REPO / '5_training/output/sft_v2', with_ref=False)
model.processor_tok = proc.tokenizer
dev = model.device
tid = proc.tokenizer.convert_tokens_to_ids
key_ids = {s.token: tid(s.token) for s in SLOTS}
value_ids = {'v': torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
             'ev': torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
stage_T = {st: stage_temps(st, 0.0) for st in STAGE_NAMES}
lc = CompositeLoss(weights={'param_reg': 0.0}).to(dev)
lbw = CompositeLoss(weights={**CompositeLoss.BW_LOSS_WEIGHTS, 'param_reg': 0.0}).to(dev)
to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)
def dist(a, b, bw):
    torch.manual_seed(0)
    with torch.no_grad(): return float(((lbw if bw else lc)(to_t(a), to_t(b)))[0])
def traj(p, ctxs):
    return [dist(render(p['small'], c), p['gt_img'], p['bw']) for c in [{}] + ctxs]
fmt = lambda D: ' '.join(f'{d:<7.2f}' for d in D) + f'  改善 {1-D[-1]/max(D[0],1e-6):+.3f}'

SHOW = ['曝光度', '色温', '饱和度', '自然饱和度', '晕影', '清晰度', '对比度']
units = load_units('val')[:4]
print(f"{'':24}" + ' '.join(f'D{i:<6}' for i in range(5)))
for u in units:
    p = prep_unit(u, 1024, 512, render)
    gts = extract_stages(json.loads((REPO / 'fit_results' / u['group'] / f"version_{u['version']}" / 'params.json').read_text()))
    nm = u['group'].split('/')[-1][:8] + '/v' + u['version'] + (' [bw]' if u['bw'] else '')
    # GT 链
    c, gt_ctxs = {}, []
    for zh, _ in gts:
        c = {**c, **zh}; gt_ctxs.append(dict(c))
    print(f"\n{nm}")
    print(f"  {'GT链':20} " + fmt(traj(p, gt_ctxs)))
    # A 自渲
    _, ctxA = chain_rollout(model, proc, p, stage_T, key_ids, value_ids, render, 92, greedy=True)
    print(f"  {'A 自渲':19} " + fmt(traj(p, ctxA)))
    # B 教师图 + 模型参数合成
    ctx_gt, cm, ctxB = {}, {}, []
    for k, st in enumerate(STAGE_NAMES):
        img = p['big'] if k == 0 else render(p['big'], ctx_gt)
        inp = build_stage_inputs(proc, to_pil(img, 92), u['category'], u['bw'], st, dev)
        with torch.no_grad():
            _, zh = sample_rollout(model, inp, st, key_ids, value_ids, stage_T[st], greedy=True)
        cm = {**cm, **zh}; ctxB.append(dict(cm))
        ctx_gt.update(gts[k][0])
    print(f"  {'B 教师图+模型参数':15} " + fmt(traj(p, ctxB)))
    # 参数对比
    gtf = gt_ctxs[-1]
    print(f"  {'参数':12}" + ''.join(f'{k[:6]:>9}' for k in SHOW))
    for nm2, d in (('GT', gtf), ('A自渲', ctxA[-1]), ('B教师', ctxB[-1])):
        print(f"    {nm2:10}" + ''.join(f'{d.get(k,0):>9.1f}' for k in SHOW))
