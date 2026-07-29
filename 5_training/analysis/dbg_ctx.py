"""验证:sample_rollout 的 KV-cache 是否被 gradient_checkpointing+train 模式打断(丢掉 prompt/图像)?
① 直接看 prompt forward 后 past_key_values 是不是空
② 同一 prompt 喂两张差异巨大的图 → 输出是否相同(相同=没在看图)
③ 关掉 grad-ckpt / eval 模式后重测(应恢复差异)
"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path('/home/chen/Work/adobe_lr')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / '5_training'))
from v2_codec import SLOTS, VALUE_TOKENS_V, VALUE_TOKENS_EV
from v2_grpo import load_policy, sample_rollout, stage_temps
from v2_grpo_chain import load_units, prep_unit, build_stage_inputs, to_pil
from build_sft_dataset import STAGE_NAMES
from differentiable_renderer.opencv_renderer import render

model, proc = load_policy('Qwen/Qwen3-VL-8B-Instruct', REPO / '5_training/output/sft_v2', with_ref=False)
model.processor_tok = proc.tokenizer
dev = model.device
tid = proc.tokenizer.convert_tokens_to_ids
key_ids = {s.token: tid(s.token) for s in SLOTS}
value_ids = {'v': torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
             'ev': torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
T = {st: stage_temps(st, 0.0) for st in STAGE_NAMES}

u = load_units('val')[0]
p = prep_unit(u, 1024, 512, render)
img_bright = render(p['big'], {'曝光度': 1.5})       # 亮
img_dark = render(p['big'], {'曝光度': -1.5})        # 暗（差异巨大）

def probe(tag):
    inp = build_stage_inputs(proc, to_pil(p['big'], 92), u['category'], u['bw'], 'light', dev)
    with torch.no_grad():
        out = model(**inp, use_cache=True)
    pkv = out.past_key_values
    n = None
    try: n = pkv.get_seq_length()
    except Exception:
        try: n = len(pkv)
        except Exception: pass
    print(f"  [{tag}] past_key_values type={type(pkv).__name__} seq_len={n}  (应≈{inp['input_ids'].shape[1]})")

def outs(tag):
    r = {}
    for nm, im in (('亮', img_bright), ('暗', img_dark)):
        inp = build_stage_inputs(proc, to_pil(im, 92), u['category'], u['bw'], 'light', dev)
        with torch.no_grad():
            _, zh = sample_rollout(model, inp, 'light', key_ids, value_ids, T['light'], greedy=True)
        r[nm] = zh
    same = r['亮'] == r['暗']
    print(f"  [{tag}] 亮图→{ {k: round(v,1) for k,v in r['亮'].items()} }")
    print(f"  [{tag}] 暗图→{ {k: round(v,1) for k,v in r['暗'].items()} }")
    print(f"  [{tag}] 两图输出相同 = {same}   {'❌ 没在看图!' if same else '✓ 在看图'}")

print("=== ① / ② GRPO 路径 (train + gradient_checkpointing) ===")
probe('grpo'); outs('grpo')

print("\n=== ③ 关掉 grad-ckpt + eval 模式 ===")
model.gradient_checkpointing_disable()
model.eval()
probe('eval'); outs('eval')
