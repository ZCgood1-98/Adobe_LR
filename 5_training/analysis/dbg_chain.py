"""定位全链 greedy 崩塌:是 bug 还是真 exposure bias?
① 校验 build_stage_inputs 与 SFT 路径(读 s 图)是否给出同样的 input_ids
② 对同一批 val 单元,打印 D_0..D_4 轨迹:自渲链 vs 教师强制(GT 中间图)
   —— 若教师强制也崩 = 打分/渲染侧 bug;若只有自渲崩 = 真 exposure bias
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

DATA = REPO / 'sft_data_v2'
model, proc = load_policy('Qwen/Qwen3-VL-8B-Instruct', REPO / '5_training/output/sft_v2', with_ref=False)
model.processor_tok = proc.tokenizer
dev = model.device
tid = proc.tokenizer.convert_tokens_to_ids
key_ids = {s.token: tid(s.token) for s in SLOTS}
value_ids = {'v': torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
             'ev': torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
stage_T = {st: stage_temps(st, 0.0) for st in STAGE_NAMES}     # greedy 用不到温度
lc = CompositeLoss(weights={'param_reg': 0.0}).to(dev)
lbw = CompositeLoss(weights={**CompositeLoss.BW_LOSS_WEIGHTS, 'param_reg': 0.0}).to(dev)
to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)
def dist(a, b, bw):
    torch.manual_seed(0)
    with torch.no_grad(): return float(((lbw if bw else lc)(to_t(a), to_t(b)))[0])

# ---------- ① prompt 一致性:同一 (group,version,light),我的拼装 vs 读 s1.jpg ----------
samples = {(s['_meta']['group'], s['_meta']['version'], s['_meta']['stage']): s
           for s in json.loads((DATA / 'task_v2_taste_val.json').read_text())}
units = load_units('val')
u0 = units[0]
p0 = prep_unit(u0, 1024, 512, render)
mine = build_stage_inputs(proc, to_pil(p0['big'], 92), u0['category'], u0['bw'], 'light', dev)
s = samples[(u0['group'], u0['version'], 'light')]
sys_txt = s['messages'][0]['content']; user_txt = s['messages'][1]['content'].replace('<image>\n', '', 1)
msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': sys_txt}]},
        {'role': 'user', 'content': [{'type': 'image', 'image': str(DATA / s['images'][0])},
                                     {'type': 'text', 'text': user_txt}]}]
ref = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                               return_dict=True, return_tensors='pt').to(dev)
a, b = mine['input_ids'][0], ref['input_ids'][0]
print(f"① prompt: 我的 len={len(a)}  SFT路径 len={len(b)}  完全相同={torch.equal(a, b) if len(a)==len(b) else False}")
if len(a) == len(b) and not torch.equal(a, b):
    d = (a != b).nonzero().flatten()[:5]
    print('   首个差异位:', [(int(i), int(a[i]), int(b[i])) for i in d])
print(f"   pixel_values shape 我的={tuple(mine['pixel_values'].shape)} SFT={tuple(ref['pixel_values'].shape)}")

# ---------- ② D 轨迹:自渲链 vs 教师强制 ----------
print(f"\n② D_0..D_4 轨迹 (D 越小越像目标;改善=1−D4/D0)")
print(f"{'unit':22} {'模式':6} " + ' '.join(f'D{i:<6}' for i in range(5)) + '  改善')
for u in units[:6]:
    p = prep_unit(u, 1024, 512, render)
    # 自渲链
    _, ctxs = chain_rollout(model, proc, p, stage_T, key_ids, value_ids, render, 92, greedy=True)
    Dc = [dist(render(p['small'], c), p['gt_img'], p['bw']) for c in [{}] + ctxs]
    # 教师强制:每 stage 喂 GT 中间图(s 图),参数仍取模型输出,但前序用 GT
    gts = extract_stages(json.loads((REPO / 'fit_results' / u['group'] / f"version_{u['version']}" / 'params.json').read_text()))
    ctx_gt, Dt = {}, [Dc[0]]
    for k, st in enumerate(STAGE_NAMES):
        img = p['big'] if k == 0 else render(p['big'], ctx_gt)
        inp = build_stage_inputs(proc, to_pil(img, 92), u['category'], u['bw'], st, dev)
        with torch.no_grad():
            _, zh = sample_rollout(model, inp, st, key_ids, value_ids, stage_T[st], greedy=True)
        Dt.append(dist(render(p['small'], {**ctx_gt, **zh}), p['gt_img'], p['bw']))
        ctx_gt.update(gts[k][0])                      # 前序换成 GT 继续
    nm = u['group'].split('/')[-1][:8] + '/v' + u['version']
    print(f"{nm:22} {'自渲':6} " + ' '.join(f'{d:<7.2f}' for d in Dc) + f'  {1-Dc[-1]/max(Dc[0],1e-6):+.3f}')
    print(f"{'':22} {'教师':6} " + ' '.join(f'{d:<7.2f}' for d in Dt) + f'  {1-Dt[-1]/max(Dt[0],1e-6):+.3f}')
