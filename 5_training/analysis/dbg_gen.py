import sys, json, torch
sys.path.insert(0, '.'); sys.path.insert(0, '5_training')
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from v2_codec import init_new_embeddings
from peft import PeftModel

base = "Qwen/Qwen3-VL-8B-Instruct"; adapter = "5_training/output/sft_v2"
proc = AutoProcessor.from_pretrained(base, min_pixels=65536, max_pixels=1048576)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")

# 记录 init 前后 embed 行,看 delta 是否真被叠加
emb = model.get_input_embeddings().weight
row_base = emb[151670].float().norm().item()          # <exposure> 之类键行(init前=基座保留区)
new_ids = init_new_embeddings(model, proc.tokenizer)
row_init = emb[151670].float().norm().item()          # init 后
model = PeftModel.from_pretrained(model, adapter)
model.eval()
# PeftModel 后再看同一行(经 trainable_tokens 前向才叠 delta,权重本身可能没变—打印看看)
emb2 = model.get_input_embeddings()
try:
    w2 = emb2.weight[151670].float().norm().item()
except Exception as e:
    w2 = f"n/a({e})"
print(f"[embed row 151670 norm] 基座={row_base:.3f}  init后={row_init:.3f}  peft后weight={w2}")

tid = proc.tokenizer.convert_tokens_to_ids
print("[token->id]", {t: tid(t) for t in ["<exposure>", "<temperature>", "<tint>", "<ev_61>", "<v_31>", "<contrast>"]})

s = [x for x in json.load(open('sft_data_v2/task_v2_taste_val.json')) if x['_meta']['stage'] == 'light'][0]
sys_txt = s['messages'][0]['content']
user_txt = s['messages'][1]['content'].replace('<image>\n', '', 1)
img = 'sft_data_v2/' + s['images'][0]
msgs = [{"role": "system", "content": [{"type": "text", "text": sys_txt}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_txt}]}]
inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                               return_dict=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inp, max_new_tokens=16, do_sample=False)
gen = out[0, inp['input_ids'].shape[1]:].tolist()
print("[GT标签]", s['messages'][-1]['content'])
print("[生成ID]", gen)
for i in gen:
    print("   ", i, repr(proc.tokenizer.convert_ids_to_tokens(i)))
