#!/usr/bin/env python3
"""v2_grpo_chain.py — 全链 GRPO（存档「线路1」本体；per-stage 之后的正式形态）
================================================================================
与 per-stage 版（v2_grpo.py）的唯一区别 = **链条是模型自己的**：
· 整条 rollout 跑在**模型自渲的中间图**上（stage k 的输入 = 用模型自己 1..k-1 的参数渲出的图），
  不再把前序钉成 GT → 闭 exposure bias、训练/部署一致、**跨 stage 的参数耦合被保留**。
· reward = **未改动的** CompositeLoss（到该 version 终图）。目标只能是「像不像目标图」，
  **不给任何单参数开小灶**（2026-07-28 纠偏：单参数 reward 定制 = Goodhart，见 TRAINING_PLAN ★07-28）。
· 信用分配 = **return-to-go**：G_t = Σ_{k≥t}(D_{k}−D_{k+1}) = D_t − D_4。
  即每个 stage 为**它之后的全部结果**负责（策略梯度的正确形式；用当步 r_t 是有偏的
  「近视」变体，会让埋雷的早期 stage 逃脱惩罚 → 级联灾难无人挨罚）。**不改变目标**。
· 优势：逐 stage 在组内 z-norm（同一 stage 横向比 N 条链），保住「组内相对选择」。
· 探索温度表 / KL-to-SFT 沿用 run2 定型配置（只改采样分布与正则，不动目标）。

目标（v*，`--vstar`）：一图多解 —— 同组常有 3~4 个专家 version（87% 的单元有 ≥2 个同 bw 候选），
而 prompt 只给 category+bw、模型无从知道要哪一版；只认本 version 会**把「做出另一种合法好看法」判成错**。
故 **终端锁 v***：跑完链后按终图选 v* = argmin_v D(终图, v 终图)，再用该 v* 回算整条 D_t。
⇒ reward 是良定义的 r(x) = −min_v D(x, v)「靠近任一合法专家看法」；v* 在单条 rollout 内固定，
  故 return-to-go 语义不变。候选一律用**同一张输入图**套各 version 的 GT 参数渲出（与各自裁剪无关，同口径）。

用法（lf 环境）：
  python 5_training/v2_grpo_chain.py --smoke
  python 5_training/v2_grpo_chain.py --steps 150 --out 5_training/output/grpo_chain
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "5_training"))
from v2_codec import SLOTS, VALUE_TOKENS_V, VALUE_TOKENS_EV
from v2_grpo import stage_temps, load_policy, sample_rollout, rollout_logp   # 复用受限解码/KL 等纪律
from v2_build_dataset import SYSTEM_PROMPT_V2, STAGE_DESC, TASTE_USER
from build_sft_dataset import extract_stages, STAGE_NAMES, _imread_rgb, _resize_max_side

DATA = REPO / "sft_data_v2"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ============================================================ 数据单元
def load_units(split, limit=0):
    """taste 集按 (group, version) 唯一化 → 每个 = 一条完整链的 prompt 单元。"""
    units = {}
    for s in json.loads((DATA / f"task_v2_taste_{split}.json").read_text()):
        m = s["_meta"]
        units.setdefault((m["group"], m["version"]),
                         {"group": m["group"], "version": m["version"],
                          "category": m["category"], "bw": bool(m["bw"])})
    out = list(units.values())
    return out[:limit] if limit else out


def _final_gt(group, version):
    """该 version 四个 stage 的 GT 合成终参数。"""
    gt_zh = {}
    for zh, _en in extract_stages(json.loads(
            (REPO / "fit_results" / group / f"version_{version}" / "params.json").read_text())):
        gt_zh.update(zh)
    return gt_zh


def group_siblings(split):
    """group → 同组所有 (version, bw)，供 v* 候选。"""
    sib = {}
    for u in load_units(split):
        sib.setdefault(u["group"], []).append((u["version"], u["bw"]))
    return sib


def prep_unit(u, big_side, small_side, render, siblings=None):
    """取原图(模型输入用大图/reward 用小图)、GT 终图、bw。
    siblings 给定时额外备 v* 候选：**同一张输入图**分别套各 version 的 GT 参数渲出的终图
    （同底图 ⇒ 与各 version 自己的裁剪无关，比较是同口径的）。仅取同 bw 的 version。"""
    vdir = REPO / "fit_results" / u["group"] / f"version_{u['version']}"
    in_png = vdir / "original_cropped.png"
    if not in_png.exists():
        in_png = REPO / "fit_results" / u["group"] / "original_captured_cropped.png"
    raw = _imread_rgb(str(in_png))
    gt_zh = _final_gt(u["group"], u["version"])
    small = _resize_max_side(raw, small_side)
    p = {**u, "big": _resize_max_side(raw, big_side), "small": small,
         "gt_img": render(small, gt_zh)}
    if siblings is not None:
        cands = [(vn, render(small, _final_gt(u["group"], vn)))
                 for vn, bw in siblings.get(u["group"], []) if bw == u["bw"]]
        p["cands"] = cands or [(u["version"], p["gt_img"])]     # 至少含自己
    return p


def to_pil(arr, jpeg_q):
    """自渲中间图 → PIL；jpeg_q>0 时走一次 JPEG 往返，对齐 SFT 的 s 图分布(q92)。"""
    im = Image.fromarray(arr)
    if jpeg_q > 0:
        import io
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=jpeg_q); buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return im


def build_stage_inputs(proc, img_pil, cat, bw, stage, dev):
    """用**内存图**(模型自渲的中间图)拼该 stage 的输入——与 SFT 的 prompt 模板逐字一致。"""
    user_txt = TASTE_USER.format(cat=cat, bw="yes" if bw else "no",
                                 desc=STAGE_DESC[stage]).replace("<image>\n", "", 1)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_V2}]},
            {"role": "user", "content": [{"type": "image", "image": img_pil},
                                         {"type": "text", "text": user_txt}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                    return_dict=True, return_tensors="pt").to(dev)


# ============================================================ 全链 rollout
def chain_rollout(model, proc, p, stage_T, key_ids, value_ids, render, jpeg_q, greedy=False):
    """跑完 4 个 stage，每步喂**模型自己渲的**中间图。
    返回 steps=[(inputs, gen, stage)...] 与 ctxs=[累积参数 after each stage]。
    ⚠ 不在此计算距离——loss 里的 manual_seed 会污染采样 RNG（历史坑），一律推迟到链跑完后统一算。
    ⚠ 用 no_grad 而非 inference_mode：inference tensor 之后无法进 autograd。"""
    ctx, steps, ctxs = {}, [], []
    with torch.no_grad():
        for k, stage in enumerate(STAGE_NAMES):
            img = p["big"] if k == 0 else render(p["big"], ctx)      # ← 自渲中间图(k=0 即原图)
            inputs = build_stage_inputs(proc, to_pil(img, jpeg_q), p["category"],
                                        p["bw"], stage, model.device)
            gen, zh = sample_rollout(model, inputs, stage, key_ids, value_ids,
                                     stage_T[stage], greedy)
            ctx = {**ctx, **zh}
            steps.append((inputs, gen, stage)); ctxs.append(dict(ctx))
    return steps, ctxs


# ============================================================ 训练
def main():
    from differentiable_renderer.opencv_renderer import render
    from differentiable_renderer.losses import CompositeLoss

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=MODEL_ID)
    ap.add_argument("--adapter", type=Path, default=REPO / "5_training/output/sft_v2")
    ap.add_argument("--out", type=Path, default=REPO / "5_training/output/grpo_chain")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts_per_step", type=int, default=1, help="每步的链单元数(每个已产 4 stage 梯度)")
    ap.add_argument("--group", type=int, default=8, help="每单元的完整链 rollout 数 N")
    ap.add_argument("--lr", type=float, default=4e-6)
    ap.add_argument("--explore_scale", type=float, default=1.0, help="逐参数探索温度缩放(0=全T1 消融)")
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--big_side", type=int, default=1024, help="模型输入图边长(对齐 SFT 的 s 图)")
    ap.add_argument("--small_side", type=int, default=512, help="reward 渲染边长(对齐 per-stage 基线)")
    ap.add_argument("--jpeg_q", type=int, default=92, help="中间图 JPEG 往返质量(对齐 s 图分布);0=关")
    ap.add_argument("--bw_loss", action="store_true", default=True, help="bw 目标用 BW 权重打分(与诊断口径一致)")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=15)
    ap.add_argument("--n_eval", type=int, default=16)
    ap.add_argument("--resume", action="store_true", help="从 out/ 下最新 step_N 续跑(共享机器易被 OOM kill)")
    ap.add_argument("--vstar", action="store_true", help="终端锁最近的同组专家版本当靶(处理一图多解)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.prompts_per_step, args.group, args.n_eval = 2, 1, 3, 3

    start_step, resume_ck = 0, None
    if args.resume and args.out.exists():                  # 找最新 step_N
        cks = [(int(d.name.split("_")[1]), d) for d in args.out.glob("step_*") if d.is_dir()]
        if cks:
            start_step, resume_ck = max(cks)
            print(f"[chain] 续跑：从 {resume_ck} (step {start_step}) 起；KL 锚仍为 {args.adapter}", flush=True)
    # ⚠ 策略续 checkpoint，但 ref(KL 锚)必须仍是 SFT，否则锚会跟着漂
    model, proc = load_policy(args.base, resume_ck or args.adapter,
                              with_ref=args.kl_coef > 0, ref_adapter=args.adapter)
    model.processor_tok = proc.tokenizer
    dev = model.device
    tid = proc.tokenizer.convert_tokens_to_ids
    key_ids = {s.token: tid(s.token) for s in SLOTS}
    value_ids = {"v": torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
                 "ev": torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
    stage_T = {st: stage_temps(st, args.explore_scale) for st in STAGE_NAMES}   # WB 不特殊照顾

    lossfn = CompositeLoss(weights={"param_reg": 0.0}).to(dev)                  # ← 未改动的目标
    lossfn_bw = CompositeLoss(weights={**CompositeLoss.BW_LOSS_WEIGHTS, "param_reg": 0.0}).to(dev)
    to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

    def dist(a, b, bw):
        torch.manual_seed(0)                       # 组内 ms_swd 投影一致(调用方负责保护采样 RNG)
        with torch.no_grad():
            return float(((lossfn_bw if (bw and args.bw_loss) else lossfn)(to_t(a), to_t(b)))[0])

    def chain_D(p, ctxs):
        """D_0..D_4：不动 + 走完每个 stage 后，到目标终图的距离。
        --vstar：目标不再钉死本 version，而是 **终端锁最近的同组专家版本**——
        先跑完链、按终图选 v* = argmin_v D(终图, 该 version 终图)，再用该 v* 回算整条 D_t。
        ⇒ reward 函数是良定义的 r(x) = −min_v D(x, v)「靠近任一合法专家看法」；
          v* 在单条 rollout 内固定，故 return-to-go 语义不变。修的是「一图多解却只认一个靶」的不公平。"""
        imgs = [render(p["small"], c) for c in [{}] + ctxs]
        tgt = p["gt_img"]
        if args.vstar and len(p.get("cands", [])) > 1:
            tgt = min(p["cands"], key=lambda vc: dist(imgs[-1], vc[1], p["bw"]))[1]   # 终端锁 v*
        return [dist(im, tgt, p["bw"]) for im in imgs]

    units = load_units("train")
    sib_tr = group_siblings("train") if args.vstar else None
    rng = np.random.default_rng(42)
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    if resume_ck and (resume_ck / "opt_state.pt").exists():
        opt.load_state_dict(torch.load(resume_ck / "opt_state.pt", map_location=dev))
        print(f"[chain] 已恢复优化器状态(Adam 动量)", flush=True)
    print(f"[chain] {len(units)} 链单元；可训 {sum(q.numel() for q in trainable)/1e6:.1f}M；"
          f"N={args.group} lr={args.lr} steps={args.steps} kl={args.kl_coef} vstar={args.vstar}", flush=True)

    # ---- 固定 val 评估集：greedy 全链 → 直接就是门①口径(1 − D_4/D_0) ----
    ev_units = load_units("val")
    sib_va = group_siblings("val") if args.vstar else None
    ev_rng = np.random.default_rng(0)
    ev_prep = [prep_unit(ev_units[i], args.big_side, args.small_side, render, sib_va)
               for i in ev_rng.choice(len(ev_units), min(args.n_eval, len(ev_units)), replace=False)]

    def greedy_eval():
        imps = []
        for p in ev_prep:
            _, ctxs = chain_rollout(model, proc, p, stage_T, key_ids, value_ids,
                                    render, args.jpeg_q, greedy=True)
            D = chain_D(p, ctxs)
            imps.append(1 - D[-1] / max(D[0], 1e-6))
        torch.cuda.empty_cache()
        return (f"全链改善 中位 {np.median(imps):+.3f} 均 {np.mean(imps):+.3f} "
                f">0比例 {np.mean([x > 0 for x in imps]):.0%}")

    print(f"[eval step 0] {greedy_eval()}", flush=True)
    t0 = time.time()
    norm = args.group * args.prompts_per_step * len(STAGE_NAMES)
    for step in range(start_step + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        step_imp, step_std, step_kl, n_upd = [], [], [], 0
        for u in [units[i] for i in rng.integers(0, len(units), args.prompts_per_step)]:
            p = prep_unit(u, args.big_side, args.small_side, render, sib_tr)
            # --- N 条完整链（采样，无梯度） ---
            rolls = [chain_rollout(model, proc, p, stage_T, key_ids, value_ids,
                                   render, args.jpeg_q) for _ in range(args.group)]
            torch.cuda.empty_cache()
            # --- telescoping reward（RNG 保护：dist 里的 seed 不能污染采样） ---
            rng_state = torch.get_rng_state()
            Dm = np.array([chain_D(p, ctxs) for _, ctxs in rolls])     # [N, 5]
            torch.set_rng_state(rng_state)
            # 信用分配 = **return-to-go**：G_t = Σ_{k≥t} r_k = D_t − D_4（[N,4]）。
            # ⚠ 不能用当步立即回报 r_t：那不是策略梯度(有偏)，且等于假装各 stage 独立——
            #   light 把链带进沟里的后果没人承担 ⇒ 级联灾难无人挨罚(实测重尾正是如此)。
            #   G_t 让每个 stage 为**它之后的全部结果**负责，与「参数耦合」一致。
            R = Dm[:, :-1] - Dm[:, -1:]                                # G_t，t=0..3
            A = (R - R.mean(0)) / (R.std(0) + 1e-4)                    # 逐 stage 组内 z-norm
            step_imp.append(np.median(1 - Dm[:, -1] / np.maximum(Dm[:, 0], 1e-6)))
            step_std.append(float(R.std(0).mean()))        # 组内多样性:太小=没得选,GRPO 学不动
            # --- policy loss（逐 rollout×stage 累积梯度） ---
            for i, (steps, _) in enumerate(rolls):
                for t, (inputs, gen, stage) in enumerate(steps):
                    a = float(A[i, t])
                    if abs(a) < 1e-6:
                        continue
                    temps = stage_T[stage]
                    lps = rollout_logp(model, inputs, gen, stage, value_ids, temps)
                    loss = -(a * lps.sum()) / norm
                    if args.kl_coef > 0:                    # KL(π‖SFT)，k3，逐 token
                        model.set_adapter("ref")
                        with torch.no_grad():
                            lps_ref = rollout_logp(model, inputs, gen, stage, value_ids, temps)
                        model.set_adapter("default")
                        r = lps_ref - lps
                        kl = (r.exp() - r - 1).sum()
                        loss = loss + args.kl_coef * kl / norm
                        step_kl.append(float(kl.detach()))
                    loss.backward()
                    n_upd += 1
            del rolls
            torch.cuda.empty_cache()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % (1 if args.smoke else 5) == 0 or step == 1:
            klmsg = f"KL均 {np.mean(step_kl):.3f} " if step_kl else ""
            print(f"  step {step:>4} 链改善中位 {np.mean(step_imp):+.3f} 组内std {np.mean(step_std):.2f} {klmsg}"
                  f"grad_norm {float(gn):.2f} 更新 {n_upd} ({(time.time()-t0)/step:.1f}s/step)", flush=True)
        if step % args.eval_every == 0:
            print(f"[eval step {step}] {greedy_eval()}", flush=True)
        if not args.smoke and step % args.save_every == 0:
            ck = args.out / f"step_{step}"
            model.save_pretrained(str(ck))
            torch.save(opt.state_dict(), ck / "opt_state.pt")     # 续跑要 Adam 动量
    if not args.smoke:
        model.save_pretrained(str(args.out))
    print(f"[chain] done {(time.time()-t0)/60:.1f}min → {args.out}", flush=True)


if __name__ == "__main__":
    main()
