"""门② 实测:同图 N 次温度采样,量参数 + reward 的组内方差(GRPO 探索信号)。lf 环境 GPU。"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, '5_training')
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from peft import PeftModel
from v2_codec import (init_new_embeddings, decode_stage, STAGE_SLOTS, SLOTS,
                      VALUE_TOKENS_V, VALUE_TOKENS_EV)
from build_sft_dataset import extract_stages, STAGE_NAMES, _imread_rgb, _resize_max_side
from differentiable_renderer.opencv_renderer import render
from differentiable_renderer.losses import CompositeLoss

BASE = "Qwen/Qwen3-VL-8B-Instruct"; ADAPTER = "5_training/output/sft_v2"; REPO = Path('.')
N, TEMP, MS = 8, 1.0, 512

proc = AutoProcessor.from_pretrained(BASE, min_pixels=65536, max_pixels=1048576)
model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="sdpa")
init_new_embeddings(model, proc.tokenizer)
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()
dev = model.device
tid = proc.tokenizer.convert_tokens_to_ids
KEY = {s.token: tid(s.token) for s in SLOTS}
VAL = {"v": torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
       "ev": torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
lossfn = CompositeLoss(weights={"param_reg": 0.0}).to(dev)
to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

def rollout(inputs, stage, temp):
    out = model(**inputs, use_cache=True); past = out.past_key_values; gen = []
    for slot in STAGE_SLOTS[stage]:
        kid = KEY[slot.token]
        out = model(input_ids=torch.tensor([[kid]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        aid = VAL[slot.group]; vlog = out.logits[0, -1][aid].float()
        if temp <= 0:
            vid = int(aid[int(vlog.argmax())])
        else:
            vid = int(aid[int(torch.multinomial(torch.softmax(vlog / temp, -1), 1))])
        out = model(input_ids=torch.tensor([[vid]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        gen += [kid, vid]
    return proc.tokenizer.convert_ids_to_tokens(gen)

samples = json.load(open('sft_data_v2/task_v2_taste_val.json'))
sel = {}
for s in samples:
    st = s['_meta']['stage']
    if st in ('light', 'tone', 'color') and len(sel.get(st, [])) < 2:
        sel.setdefault(st, []).append(s)
picked = [s for v in sel.values() for s in v]
KEYP = {'light': ['exposure', 'temperature'], 'tone': ['contrast', 'highlights', 'shadows'],
        'color': ['saturation', 'vibrance']}

print(f"门② 实测  T={TEMP}  N={N}  {len(picked)} 图  (reward=−距离,越大越好;组内 std 大=有探索)")
t0 = time.time(); rstds = []
for s in picked:
    st = s['_meta']['stage']; key = s['_meta']['group']; vn = s['_meta']['version']
    sys_txt = s['messages'][0]['content']; user_txt = s['messages'][1]['content'].replace('<image>\n', '', 1)
    msgs = [{"role": "system", "content": [{"type": "text", "text": sys_txt}]},
            {"role": "user", "content": [{"type": "image", "image": 'sft_data_v2/' + s['images'][0]},
                                         {"type": "text", "text": user_txt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(dev)
    params = json.loads((REPO / 'fit_results' / key / f'version_{vn}' / 'params.json').read_text())
    stages = extract_stages(params); k = STAGE_NAMES.index(st)
    ctx = {}; [ctx.update(stages[i][0]) for i in range(k)]
    in_png = REPO / 'fit_results' / key / f'version_{vn}' / 'original_cropped.png'
    if not in_png.exists(): in_png = REPO / 'fit_results' / key / 'original_captured_cropped.png'
    img0 = _resize_max_side(_imread_rgb(str(in_png)), MS)
    gt_ref = to_t(render(img0, {**ctx, **stages[k][0]}))
    ens = []
    with torch.inference_mode():                       # 阶段1:只采样,RNG 自然推进→N 个不同 rollout
        for n in range(N):
            ens.append(decode_stage(st, rollout(inputs, st, TEMP)))
    rews = []                                           # 阶段2:渲染+reward(seed(0)固定 ms_swd,不再污染采样)
    for en, zh in ens:
        pred = to_t(render(img0, {**ctx, **zh}))
        torch.manual_seed(0)
        with torch.no_grad(): rews.append(-float(lossfn(pred, gt_ref)[0]))
    rews = np.array(rews); rstds.append(rews.std())
    pstd = {p: round(float(np.std([e[p] for e, z in ens])), 2) for p in KEYP[st]}
    print(f"  [{st:5}] reward 均{rews.mean():.2f} std{rews.std():.3f} 幅[{rews.min():.2f},{rews.max():.2f}] | 参数组内std {pstd}")
print(f"\n[汇总] reward 组内 std 中位 = {np.median(rstds):.3f}  ({time.time()-t0:.0f}s)")
print("判读: reward 组内 std 明显>0 且参数 std 非零 = 有探索方差, GRPO 组内优势不会全塌成0 → 门②过")
