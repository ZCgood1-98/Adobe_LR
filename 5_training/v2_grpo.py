#!/usr/bin/env python3
"""v2_grpo.py — SFT-v2 之后的 GRPO（第一版：per-stage 教师强制 + 线路1 参考距离 reward）
================================================================================
· reward 结构：per-stage 教师强制——每 stage 单独 rollout，前序用 GT 参数固定；
  reward = −L_composite(render(原图, GT前序 ∪ 预测本stage), render(原图, GT前序 ∪ GT本stage))。
· reward 定义：线路1（参考图感知距离，不蒸馏、近乎不可 hack），暂不上线路2。
· 算法：最小 GRPO = on-policy REINFORCE + 组内相对优势（组内 z-norm）。单步单更新 → 无需 PPO clip。逐 rollout 梯度累积。
· 逐参数探索温度（--explore_scale）：感知强却保守的维(contrast/texture/clarity 等)升采样温度 T>1，
  采样+logp 同温=on-policy π^T；argmax 与 T 无关 → greedy 评估/部署仍是 T=1 策略。sat/vib 与弱信号维保持 T=1。
· KL-to-SFT（--kl_coef）：PEFT 多适配器——第二个「ref」适配器=冻结的 SFT 副本(不占 2×8B)，
  k3 估计逐 token 罚偏离 SFT，护住已校准维不被跨 token 耦合带偏。设 0 关闭。
· 可训参数 = SFT 那批（LoRA + 247 token 行），从 sft_v2 adapter 续。

用法（lf 环境）：
  python 5_training/v2_grpo.py --smoke
  python 5_training/v2_grpo.py --steps 800 --out 5_training/output/grpo_v1
"""
from __future__ import annotations
import argparse, json, sys, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "5_training"))
from v2_codec import (init_new_embeddings, decode_stage, STAGE_SLOTS, SLOTS,
                      VALUE_TOKENS_V, VALUE_TOKENS_EV)
from build_sft_dataset import extract_stages, STAGE_NAMES, _imread_rgb, _resize_max_side

DATA = REPO / "sft_data_v2"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ============================================================ 逐参数探索温度表
# 判据修订(2026-07-26 grpo_explore 实测)：探索只对「渲染 reward 敏感 + 感知强」的维有效；
# reward 不敏感(detail 在 512px 复合 loss 下)或感知弱(WB r0.17)的维,升 T = 纯加噪、MAE 反升。
# 故只保留 tone/light 里 reward 敏感的强感知维；detail 细节维与 WB 交回 T=1。
EXPLORE_TEMP = {
    "contrast": 1.6, "whites": 1.4,          # tone：reward 敏感+感知强 → 实测有效(改善中位 +0.007→+0.081)
    "exposure": 1.3, "tint": 1.3,            # light：实测 MAE 微降、无害
    # ✋ saturation/vibrance → 1.0(已校准好，靠 KL 守住，实测未漂)
    # ✖ 已移除·实测反噬：temperature(WB 探索纯加噪 MAE 421→474)、
    #    texture/clarity(detail 在 512px reward 下不敏感 → MAE 12.6→14.6 / 15.9→18.3、detail 改善 −0.02→−0.10)
    # ✖ 其余弱信号维(shadows/dehaze/blacks/highlights/sharpen/降噪)→ 1.0
}


def stage_temps(stage, scale, wb_t=1.0):
    """该 stage 各 slot 的采样/logp 温度；scale：0=全1(消融) 1=按表 >1=更激进。
    wb_t：temperature(WB) 专用探索温度(WB 实验用；默认1.0=不探，不污染 EXPLORE_TEMP 表)。"""
    out = []
    for s in STAGE_SLOTS[stage]:
        if s.en_path[-1] == "temperature":
            out.append(wb_t)
        else:
            out.append(1.0 + (EXPLORE_TEMP.get(s.en_path[-1], 1.0) - 1.0) * scale)
    return out


# ============================================================ 模型
def load_policy(base, adapter, with_ref=False, ref_adapter=None):
    """adapter=策略(续训起点)；ref_adapter=KL 锚(默认同 adapter)。
    ⚠ 断点续跑时策略从 checkpoint 起，但 **ref 必须仍指向 SFT**——否则 KL 锚会跟着漂、失去约束。"""
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(base)          # 与 SFT 训练一致的默认分辨率(显存 + 输入分布对齐)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    init_new_embeddings(model, proc.tokenizer)                       # 复现词表手术
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)  # "default"=策略,可训,续 SFT
    if with_ref:                                                   # KL 参考=冻结的 SFT 副本(第二个 LoRA,不占 2×8B)
        model.load_adapter(str(ref_adapter or adapter), adapter_name="ref")
        for n, p in model.named_parameters():                      # 显式冻结 ref:永不进优化器、永不漂移
            if "ref" in n.split("."):
                p.requires_grad_(False)
        model.set_adapter("default")                               # 确保策略适配器在位
    model.enable_input_require_grads()                              # 冻结 base 时梯度能流回
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()                                                  # 梯度检查点仅 train 模式生效(lora_dropout=0→无 dropout 副作用)
    return model, proc


# ============================================================ rollout（T=1 受限采样）
@contextmanager
def sampling_mode(model):
    """采样期必须【关梯度检查点 + eval】。
    ⚠️ 2026-07-28 揪出的致命 bug：train()+gradient_checkpointing 会让 transformers 强制
    use_cache=False → prompt forward 返回的 past_key_values 为 None → 逐 token 增量解码
    丢掉整个 prompt 和图像，rollout 退化成「与输入图无关的常量」(亮图/暗图输出逐位相同)。
    梯度检查点只影响 backward，采样期无关；lora_dropout=0 故 eval 模式无副作用。"""
    was_train = model.training
    model.gradient_checkpointing_disable()
    model.eval()
    try:
        yield
    finally:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if was_train:
            model.train()


def sample_rollout(model, inputs, stage, key_ids, value_ids, temps, greedy=False):
    """强制键、只在桶组内取值 bin(greedy=argmax / 否则按逐参数温度 temps 采样)。返回 (gen_ids, 中文渲染参数 zh)。"""
    dev = model.device
    with sampling_mode(model):
        return _sample_rollout_inner(model, inputs, stage, key_ids, value_ids, temps, greedy)


def _sample_rollout_inner(model, inputs, stage, key_ids, value_ids, temps, greedy):
    dev = model.device
    out = model(**inputs, use_cache=True); past = out.past_key_values
    if past is None or past.get_seq_length() < inputs["input_ids"].shape[1]:
        raise RuntimeError(                                  # 不变量:cache 必须含完整 prompt,否则等于没看图
            f"KV cache 失效(past={type(past).__name__})：rollout 会丢掉 prompt/图像。"
            f"检查是否在 train()+gradient_checkpointing 下采样。")
    gen = []
    for i, slot in enumerate(STAGE_SLOTS[stage]):
        kid = key_ids[slot.token]
        out = model(input_ids=torch.tensor([[kid]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        aid = value_ids[slot.group]; vlog = out.logits[0, -1][aid].float()
        vid = (int(aid[int(vlog.argmax())]) if greedy          # argmax 与温度无关 → greedy 评估不受探索温度影响
               else int(aid[int(torch.multinomial(torch.softmax(vlog / temps[i], -1), 1))]))
        out = model(input_ids=torch.tensor([[vid]], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values
        gen += [kid, vid]
    _, zh = decode_stage(stage, model_tok(model).convert_ids_to_tokens(gen))
    return gen, zh


def model_tok(model):
    return getattr(model, "processor_tok", None)


# ============================================================ logprob（带梯度的单次 forward）
def rollout_logp(model, inputs, gen_ids, stage, value_ids, temps):
    """teacher-forcing 整条 [prompt ++ gen] 单次 forward,取各值 token 在其桶组内(按温度)的 log π(sampled)。
    返回逐 token 的 logp 向量[n_slots](带梯度)——调用方 .sum() 作策略 logp、也可逐 token 算 KL。"""
    dev = model.device
    P = inputs["input_ids"].shape[1]
    gen_t = torch.tensor([gen_ids], device=dev)
    ids = torch.cat([inputs["input_ids"], gen_t], dim=1)
    attn = torch.cat([inputs["attention_mask"], torch.ones_like(gen_t)], dim=1)
    mmtt = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(gen_t)], dim=1)  # gen 均文本 token(类型0)
    out = model(input_ids=ids, attention_mask=attn, pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=mmtt, use_cache=False)  # 让梯度检查点生效
    logits = out.logits
    lps = []
    for i, slot in enumerate(STAGE_SLOTS[stage]):
        j = 2 * i + 1                                 # 值 token 在 gen 里的位置
        vid = gen_ids[j]
        grp = value_ids[slot.group]
        lse = torch.log_softmax(logits[0, P + j - 1, grp].float() / temps[i], -1)   # 同采样温度 → on-policy π^T
        idx = int((grp == vid).nonzero()[0, 0])
        lps.append(lse[idx])
    return torch.stack(lps)


# ============================================================ reward（线路1 参考距离）
def reward_setup(sample):
    """取该 (图,stage,version) 的:模型输入图 s、渲染底图 img0、GT前序 ctx、GT本stage zh、GT参考图。"""
    m = sample["_meta"]; key, vn, stage = m["group"], m["version"], m["stage"]
    vdir = REPO / "fit_results" / key / f"version_{vn}"
    stages = extract_stages(json.loads((vdir / "params.json").read_text()))
    k = STAGE_NAMES.index(stage)
    ctx = {}; [ctx.update(stages[i][0]) for i in range(k)]
    in_png = vdir / "original_cropped.png"
    if not in_png.exists(): in_png = REPO / "fit_results" / key / "original_captured_cropped.png"
    return stage, _resize_max_side(_imread_rgb(str(in_png)), 512), ctx, stages[k][0]


# ============================================================ 训练
def main():
    from differentiable_renderer.opencv_renderer import render
    from differentiable_renderer.losses import CompositeLoss

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=MODEL_ID)
    ap.add_argument("--adapter", type=Path, default=REPO / "5_training/output/sft_v2")
    ap.add_argument("--out", type=Path, default=REPO / "5_training/output/grpo_v1")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--group", type=int, default=8, help="每 prompt 的 rollout 数 N")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--explore_scale", type=float, default=1.0, help="逐参数探索温度缩放:0=全T1(消融) 1=按表 >1=更激进")
    ap.add_argument("--kl_coef", type=float, default=0.02, help="KL-to-SFT 系数(0=关);ref=冻结SFT副本,护住已校准维")
    ap.add_argument("--stage", default=None, choices=STAGE_NAMES, help="只训某个 stage(WB 等专项实验);默认全 stage")
    ap.add_argument("--wb_weight", type=float, default=0.5, help="light reward 里 ab_mean(全局色偏=WB)权重;默认0.5=不boost,专项设1.5~2")
    ap.add_argument("--wb_explore_t", type=float, default=1.0, help="temperature(WB) 专用探索温度;默认1.0=不探,WB实验设1.5")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=24, help="固定 val 评估集大小(每 stage n_eval//4)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.prompts_per_step, args.group, args.n_eval = 3, 1, 4, 4

    model, proc = load_policy(args.base, args.adapter, with_ref=args.kl_coef > 0)
    model.processor_tok = proc.tokenizer
    dev = model.device
    tid = proc.tokenizer.convert_tokens_to_ids
    key_ids = {s.token: tid(s.token) for s in SLOTS}
    value_ids = {"v": torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
                 "ev": torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
    stage_T = {st: stage_temps(st, args.explore_scale, args.wb_explore_t) for st in STAGE_NAMES}   # 逐 stage 的逐参数温度
    lossfn = CompositeLoss(weights={"param_reg": 0.0}).to(dev)
    # light 专用:加重 ab_mean(全图平均色度=全局暖/冷偏=白平衡)→ 去稀释 WB 信号、给 WB 一根锐梯度
    lossfn_light = CompositeLoss(weights={"param_reg": 0.0, "ab_mean": args.wb_weight}).to(dev)
    to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

    samples = json.loads((DATA / "task_v2_taste_train.json").read_text())
    if args.stage:                                       # 专项实验:只留某 stage 的 prompt,算力全砸这一维
        samples = [s for s in samples if s["_meta"]["stage"] == args.stage]
    rng = np.random.default_rng(42)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[grpo] {len(samples)} prompts；可训参数 {sum(p.numel() for p in trainable)/1e6:.1f}M；"
          f"N={args.group} lr={args.lr} steps={args.steps}", flush=True)

    def build_inputs(s):
        sys_txt = s["messages"][0]["content"]
        user_txt = s["messages"][1]["content"].replace("<image>\n", "", 1)
        msgs = [{"role": "system", "content": [{"type": "text", "text": sys_txt}]},
                {"role": "user", "content": [{"type": "image", "image": str(DATA / s["images"][0])},
                                             {"type": "text", "text": user_txt}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to(dev)

    # 固定 val 评估集:每 stage 抽 n_eval//4;greedy reward 随步数上升 = 策略在学
    from collections import defaultdict
    vbyst = defaultdict(list)
    for s in json.loads((DATA / "task_v2_taste_val.json").read_text()):
        vbyst[s["_meta"]["stage"]].append(s)
    ev_rng = np.random.default_rng(0); eval_prep = []
    for st in STAGE_NAMES:
        pool = vbyst[st]
        for i in ev_rng.choice(len(pool), min(args.n_eval // 4, len(pool)), replace=False):
            s = pool[i]; stg, img0, ctx, gt_zh = reward_setup(s)
            eval_prep.append((s, stg, img0, ctx, to_t(render(img0, {**ctx, **gt_zh}))))

    def greedy_eval():
        rr = defaultdict(list)
        with torch.inference_mode():
            for s, stg, img0, ctx, gt_ref in eval_prep:
                _, zh = sample_rollout(model, build_inputs(s), stg, key_ids, value_ids, stage_T[stg], greedy=True)
                torch.manual_seed(0)
                with torch.no_grad():
                    rr[stg].append(-float(lossfn(to_t(render(img0, {**ctx, **zh})), gt_ref)[0]))
        torch.cuda.empty_cache()
        allr = [r for v in rr.values() for r in v]
        return f"greedy均 {np.mean(allr):+.3f} | " + " ".join(f"{st}:{np.mean(rr[st]):+.2f}" for st in STAGE_NAMES if rr[st])

    print(f"[eval step 0] {greedy_eval()}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        batch = [samples[i] for i in rng.integers(0, len(samples), args.prompts_per_step)]
        step_rew, step_adv, step_kl, n_roll = [], [], [], 0
        for s in batch:
            inputs = build_inputs(s)
            stage, img0, ctx, gt_zh = reward_setup(s)
            gt_ref = to_t(render(img0, {**ctx, **gt_zh}))
            # --- N rollouts（采样,无梯度） ---
            rolls = []
            with torch.inference_mode():
                for _ in range(args.group):
                    gen, zh = sample_rollout(model, inputs, stage, key_ids, value_ids, stage_T[stage])
                    rolls.append((gen, zh))
            torch.cuda.empty_cache()                   # 释放采样 KV cache,给梯度 forward 腾地方
            # --- reward ---
            rews = []
            lf = lossfn_light if stage == "light" else lossfn   # light 用 WB 加重的尺子
            rng_state = torch.get_rng_state()          # 存采样 RNG,别被 loss 的 seed 污染
            for gen, zh in rolls:
                pred = to_t(render(img0, {**ctx, **zh}))
                torch.manual_seed(0)                    # 组内 ms_swd 投影一致→相对优势不带投影噪
                with torch.no_grad(): rews.append(-float(lf(pred, gt_ref)[0]))
            torch.set_rng_state(rng_state)              # 恢复,下个 prompt 继续新鲜采样
            rews = np.array(rews); adv = (rews - rews.mean()) / (rews.std() + 1e-4)
            step_rew.append(rews.mean()); step_adv.append(np.abs(adv).mean())
            # --- policy loss（逐 rollout 累积梯度）+ KL-to-SFT ---
            temps = stage_T[stage]; norm = args.group * args.prompts_per_step
            for (gen, _), a in zip(rolls, adv):
                if abs(a) < 1e-6:
                    continue
                lps = rollout_logp(model, inputs, gen, stage, value_ids, temps)   # 逐 token,带梯度
                loss = -(a * lps.sum()) / norm
                if args.kl_coef > 0:                              # KL(π_θ‖SFT) k3 估计,逐 token,梯度只经 lps
                    model.set_adapter("ref")
                    with torch.no_grad():
                        lps_ref = rollout_logp(model, inputs, gen, stage, value_ids, temps)
                    model.set_adapter("default")
                    r = lps_ref - lps
                    kl = (r.exp() - r - 1).sum()
                    loss = loss + args.kl_coef * kl / norm
                    step_kl.append(float(kl.detach()))
                loss.backward()
                n_roll += 1
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % (1 if args.smoke else 10) == 0 or step == 1:
            klmsg = f"KL均 {np.mean(step_kl):.3f} " if step_kl else ""
            print(f"  step {step:>4} reward均 {np.mean(step_rew):+.3f} |adv|均 {np.mean(step_adv):.3f} "
                  f"{klmsg}grad_norm {float(gn):.2f} rollouts {n_roll} ({(time.time()-t0)/step:.1f}s/step)", flush=True)
        if step % args.eval_every == 0:
            print(f"[eval step {step}] {greedy_eval()}", flush=True)
        if not args.smoke and step % args.save_every == 0:
            model.save_pretrained(str(args.out / f"step_{step}"))
    if not args.smoke:
        model.save_pretrained(str(args.out))
    print(f"[grpo] done {(time.time()-t0)/60:.1f}min → {args.out}", flush=True)


if __name__ == "__main__":
    main()
