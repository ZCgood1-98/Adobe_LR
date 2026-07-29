#!/usr/bin/env python3
"""
diagnose_stage_error.py — 线5 · SFT 后的分 stage 图像空间诊断
=============================================================

回答一个问题：**模型到底掌握了哪个 stage 的参数组，哪个没掌握？**
（token loss 因参数多解冗余有噪声下限，不可比；图像空间距离才是公平尺子）

对 val（默认）每条样本：
  1. 模型看 stage 输入图（教师强制上下文，同训练分布）→ 直出该 stage 参数 token（v2 bin / v1 JSON）；
  2. 同一原图上渲两张终态：render(前序GT ∪ 预测参数) vs render(前序GT ∪ GT参数)；
  3. 距离 = differentiable_renderer.losses.CompositeLoss（与拟合/未来 GRPO reward
     同一把尺子；黑白样本用 BW_LOSS_WEIGHTS；param_reg=0，纯图像项）；
  4. 对照基线 = render(前序GT ∪ 全零) —— 「这一步啥都不动」的距离。
     模型有效性看 L_pred 相对 L_noop 的改善比例，排除「stage 本身接近无操作」的假好成绩。

两阶段可断点续跑：
  gen   （GPU，lf 环境）：批量推理 → <out>/predictions.jsonl（幂等追加）
  score （CPU池+GPU loss）：渲染+算距离 → <out>/scores.jsonl + report.json + 终端报表

用法（仓库根目录，lf 环境；默认 v2：codec=bins / adapter=sft_v2 / task=task_v2_taste）：
  python 5_training/diagnose_stage_error.py all                      # gen + score（v2）
  python 5_training/diagnose_stage_error.py gen  --limit 40          # 快速试跑
  python 5_training/diagnose_stage_error.py score --workers 12
  python 5_training/diagnose_stage_error.py gen  --no_adapter        # 裸基座对照
  python 5_training/diagnose_stage_error.py all --codec json --task task_a --adapter 5_training/output/sft_v1  # 跑 v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))

from build_sft_dataset import (STAGE_NAMES, MIXER_CHANNELS, CG_ZONES,  # noqa: E402
                               extract_stages, _imread_rgb, _resize_max_side)

# ============================================================
# 英文预测 JSON → 渲染器中文参数（extract_stages 的逆映射）
# ============================================================

SCALAR_EN2ZH = {
    "light":  {"exposure": "曝光度", "temperature": "色温", "tint": "色调"},
    "tone":   {"contrast": "对比度", "highlights": "高光", "shadows": "阴影",
               "whites": "白色色阶", "blacks": "黑色色阶", "dehaze": "去朦胧"},
    "color":  {"vibrance": "自然饱和度", "saturation": "饱和度"},
    "detail": {"texture": "纹理", "clarity": "清晰度", "sharpen": "锐化",
               "noise_reduction": "噪点消除", "vignette": "晕影"},
}
# 缺键默认值（=「不动」）；宽松夹取范围防极端垃圾值弄崩渲染统计
DEFAULTS = {"temperature": 5500.0}
CLAMP = {"exposure": (-10, 10), "temperature": (1500, 20000)}
CLAMP_DEFAULT = (-200, 200)


def _num(v, key, flags):
    try:
        x = float(v)
    except (TypeError, ValueError):
        flags.add(f"bad:{key}")
        return DEFAULTS.get(key, 0.0)
    lo, hi = CLAMP.get(key, CLAMP_DEFAULT)
    if not (lo <= x <= hi):
        flags.add(f"clamp:{key}")
        x = min(max(x, lo), hi)
    return x


def pred_to_zh(stage: str, pred: dict, flags: set) -> dict:
    """预测的英文 JSON → 渲染器中文参数子集。缺键按 0（不动）补，异常值夹取。"""
    zh = {}
    for en, cn in SCALAR_EN2ZH[stage].items():
        if en not in pred:
            flags.add(f"miss:{en}")
        zh[cn] = _num(pred.get(en, DEFAULTS.get(en, 0.0)), en, flags)
    if stage == "tone":
        tc = pred.get("tone_curve")
        if not isinstance(tc, list):
            flags.add("miss:tone_curve"); tc = []
        tc = [_num(v, "tone_curve", flags) for v in tc[:8]]
        zh["tone_curve_dy"] = tc + [0.0] * (8 - len(tc))
    elif stage == "color":
        mixer = pred.get("color_mixer") or {}
        if not isinstance(mixer, dict):
            flags.add("bad:color_mixer"); mixer = {}
        mix = []
        for ch in MIXER_CHANNELS:
            e = mixer.get(ch.lower()) or {}
            if not isinstance(e, dict):
                flags.add("bad:color_mixer"); e = {}
            mix.append({"channel": ch,
                        "H": _num(e.get("h", 0), "mixer", flags),
                        "S": _num(e.get("s", 0), "mixer", flags),
                        "L": _num(e.get("l", 0), "mixer", flags)})
        zh["color_mixer"] = mix
        grading = pred.get("color_grading") or {}
        if not isinstance(grading, dict):
            flags.add("bad:color_grading"); grading = {}
        cg = {}
        for z in CG_ZONES:
            d = grading.get(z) or {}
            if not isinstance(d, dict):
                flags.add("bad:color_grading"); d = {}
            cg[z] = {"R": _num(d.get("r", 0), "grading", flags),
                     "G": _num(d.get("g", 0), "grading", flags),
                     "B": _num(d.get("b", 0), "grading", flags)}
        zh["颜色分级"] = cg
    return zh


def parse_pred_json(raw: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象（训练目标就是裸 JSON，此处只做兜底）。"""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ============================================================
# 每参数 MAE（英文键空间，分辨 stage 内谁没学会）
# ============================================================

def per_param_mae(stage: str, pred: dict, gt: dict, acc: dict):
    def add(k, d):
        a = acc.setdefault(k, [0.0, 0])
        a[0] += abs(d); a[1] += 1
    for en in SCALAR_EN2ZH[stage]:
        try:
            add(en, float(pred.get(en, DEFAULTS.get(en, 0.0))) - gt[en])
        except (TypeError, ValueError):
            pass
    if stage == "tone" and isinstance(pred.get("tone_curve"), list):
        p = pred["tone_curve"][:8] + [0.0] * max(0, 8 - len(pred.get("tone_curve", [])))
        for a, b in zip(p, gt["tone_curve"]):
            try: add("tone_curve", float(a) - b)
            except (TypeError, ValueError): pass
    if stage == "color":
        pm, gm = pred.get("color_mixer") or {}, gt["color_mixer"]
        for ch, gtd in gm.items():
            pd = pm.get(ch) if isinstance(pm, dict) else None
            for a in "hsl":
                try: add("color_mixer", float((pd or {}).get(a, 0)) - gtd[a])
                except (TypeError, ValueError): pass
        pg, gg = pred.get("color_grading") or {}, gt["color_grading"]
        for z, gtd in gg.items():
            pd = pg.get(z) if isinstance(pg, dict) else None
            for a in "rgb":
                try: add("color_grading", float((pd or {}).get(a, 0)) - gtd[a])
                except (TypeError, ValueError): pass


# ============================================================
# Phase gen：模型推理（GPU）
# ============================================================

def _bins_generate(model, processor, inputs, stage, key_ids, value_ids):
    """v2 受限解码：逐槽强制键 token、只在该槽桶组内贪心选值 bin，构造 <k><v><k><v>… schema。
    （训练时键 token 的 loss 被 mask、模型学不会自己吐键；设计就是推理强制注入键、只让模型采 bin。）"""
    import torch
    from v2_codec import STAGE_SLOTS
    dev = model.device
    out = model(**inputs, use_cache=True)                 # 先吃完 prompt+图，建 KV cache
    past = out.past_key_values
    gen = []
    for slot in STAGE_SLOTS[stage]:
        kid = key_ids[slot.token]                         # 强制键
        out = model(input_ids=torch.tensor([[kid]], device=dev),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        aid = value_ids[slot.group]                       # 该桶组值 token id（v=63 / ev=127）
        vlog = out.logits[0, -1]
        vid = int(aid[int(vlog[aid].argmax())])           # 只在组内 bins 里贪心
        out = model(input_ids=torch.tensor([[vid]], device=dev),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        gen += [kid, vid]
    return "".join(processor.tokenizer.convert_ids_to_tokens(gen))


def run_gen(args):
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    samples = json.loads((args.data_dir / f"{args.task}_{args.split}.json").read_text())
    if args.limit:
        samples = samples[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["id"] for l in pred_path.read_text().splitlines() if l}
        print(f"[gen] 已有 {len(done)} 条预测，续跑")

    print(f"[gen] 加载 {args.base}" + ("" if args.no_adapter else f" + LoRA {args.adapter}"))
    processor = AutoProcessor.from_pretrained(
        args.base, min_pixels=65536, max_pixels=1048576)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    if args.codec == "bins":
        # v2：复现训练时的词表手术——注入 247 个新 token 到 tokenizer + 初始化 embedding 行（seed=42
        # 确定性，与训练一致）；adapter 的 trainable_tokens delta 随后叠在此初始化上。缺这步=模型不认 bin token。
        from v2_codec import init_new_embeddings
        init_new_embeddings(model, processor.tokenizer)
    if not args.no_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    key_ids, value_ids = None, None
    if args.codec == "bins":                              # 预备受限解码用的 token id
        from v2_codec import SLOTS, VALUE_TOKENS_V, VALUE_TOKENS_EV
        tid = processor.tokenizer.convert_tokens_to_ids
        key_ids = {sl.token: tid(sl.token) for sl in SLOTS}
        value_ids = {"v": torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=model.device),
                     "ev": torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=model.device)}

    t0, n_new = time.time(), 0
    with pred_path.open("a") as fout, torch.inference_mode():
        for i, s in enumerate(samples):
            meta = s["_meta"]
            sid = f'{meta["group"]}/v{meta["version"]}/{meta["stage"]}'
            if sid in done:
                continue
            sys_txt = s["messages"][0]["content"]
            # 训练格式里 user 文本首行是字面 "<image>"；HF 模板由 image 条目注入
            # 图占位，这里必须剥掉，否则图 token 会翻倍
            user_txt = s["messages"][1]["content"].replace("<image>\n", "", 1)
            img_path = str(args.data_dir / s["images"][0])
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_txt}]},
                {"role": "user", "content": [{"type": "image", "image": img_path},
                                             {"type": "text", "text": user_txt}]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            if args.codec == "bins":                 # v2:受限解码,强制键 schema、只采 bin
                raw = _bins_generate(model, processor, inputs, meta["stage"], key_ids, value_ids)
            else:                                    # v1:自由生成 JSON
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                raw = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            fout.write(json.dumps({"id": sid, "stage": meta["stage"],
                                   "bw": meta["bw"], "category": meta["category"],
                                   "raw": raw}, ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1
            if n_new % 20 == 0:
                rate = n_new / (time.time() - t0)
                print(f"  [{i + 1}/{len(samples)}] {rate:.2f} it/s  "
                      f"eta {(len(samples) - i - 1) / max(rate, 1e-6) / 60:.0f} min",
                      flush=True)
    print(f"[gen] 新增 {n_new} 条 → {pred_path}")


# ============================================================
# Phase score：渲染（CPU池）+ 复合距离（GPU）
# ============================================================

def render_triplet(task: dict):
    """worker：渲 (预测终态, GT 终态, 无操作基线) 三张图。"""
    try:
        from differentiable_renderer.opencv_renderer import render
        key, vn, k = task["key"], task["vn"], task["stage_idx"]
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        params = json.loads((vdir / "params.json").read_text())
        stages = extract_stages(params)

        in_png = vdir / "original_cropped.png"
        if not in_png.exists():
            in_png = REPO_ROOT / "fit_results" / key / "original_captured_cropped.png"
        img0 = _resize_max_side(_imread_rgb(str(in_png)), task["max_side"])

        ctx: dict = {}
        for i in range(k):                       # 前序 stage 的 GT（教师强制上下文）
            ctx.update(stages[i][0])
        gt_img = render(img0, {**ctx, **stages[k][0]})
        pred_img = render(img0, {**ctx, **task["pred_zh"]})
        noop_img = render(img0, dict(ctx))       # 该 stage 全不动
        meta = json.loads((vdir / "fit_meta.json").read_text())
        return {"id": task["id"], "ok": True, "bw_mode": bool(meta.get("bw_mode")),
                "pred": pred_img, "gt": gt_img, "noop": noop_img}
    except Exception as e:  # noqa: BLE001
        return {"id": task["id"], "ok": False, "err": repr(e)}


def run_score(args):
    import torch
    from differentiable_renderer.losses import CompositeLoss

    pred_path = args.out / "predictions.jsonl"
    preds = [json.loads(l) for l in pred_path.read_text().splitlines() if l]
    print(f"[score] {len(preds)} 条预测")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # param_reg 是拟合期的参数正则、不是图像距离，置 0；黑白样本换 BW 权重档
    loss_color = CompositeLoss(weights={"param_reg": 0.0}).to(device)
    loss_bw = CompositeLoss(weights={**CompositeLoss.BW_LOSS_WEIGHTS,
                                     "param_reg": 0.0}).to(device)

    def to_t(img):
        return (torch.from_numpy(img.astype(np.float32) / 255.0)
                .permute(2, 0, 1).unsqueeze(0).to(device))

    def dist(a, b, bw):
        with torch.no_grad():
            # 固定种子：ms_swd 的随机投影/像素抽样对每次调用一致，
            # 否则 pred/noop 两次测量各带 ±2-4% 投影噪声，稀释改善比例
            torch.manual_seed(0)
            loss, _ = (loss_bw if bw else loss_color)(to_t(a), to_t(b))
        return float(loss)

    # 组装渲染任务：解析预测 → 英文参数(算 MAE) + 中文渲染参数(渲染)
    from v2_codec import decode_stage, STAGE_SLOTS
    tasks, rows, mae_acc = [], {}, {}
    for p in preds:
        key_vn, stage = p["id"].rsplit("/", 1)
        key, vn = key_vn.rsplit("/v", 1)
        row = {"id": p["id"], "stage": stage, "bw": p["bw"],
               "category": p["category"], "flags": []}
        flags: set = set()
        if args.codec == "bins":
            toks = re.findall(r"<[^>]+>", p["raw"])
            stage_keys = {s.token for s in STAGE_SLOTS[stage]}
            n_found = sum(1 for t in toks if t in stage_keys)
            if n_found == 0:                          # 没吐出该 stage 任何键 token = 真解析失败
                row["parse_fail"] = True; rows[p["id"]] = row; continue
            if n_found < len(stage_keys):
                flags.add(f"miss:{len(stage_keys) - n_found}键")
            pred_en, pred_zh = decode_stage(stage, toks)
        else:
            obj = parse_pred_json(p["raw"])
            if obj is None:
                row["parse_fail"] = True; rows[p["id"]] = row; continue
            pred_zh = pred_to_zh(stage, obj, flags); pred_en = obj
        row["flags"] = sorted(flags)
        rows[p["id"]] = row
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        gt_en = extract_stages(json.loads((vdir / "params.json").read_text())
                               )[STAGE_NAMES.index(stage)][1]
        per_param_mae(stage, pred_en, gt_en, mae_acc.setdefault(stage, {}))
        tasks.append({"id": p["id"], "key": key, "vn": vn,
                      "stage_idx": STAGE_NAMES.index(stage),
                      "pred_zh": pred_zh, "max_side": args.max_side})

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(render_triplet, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            row = rows[r["id"]]
            if not r["ok"]:
                row["render_err"] = r["err"]
            else:
                bw = r["bw_mode"]
                row["L_pred"] = round(dist(r["pred"], r["gt"], bw), 4)
                row["L_noop"] = round(dist(r["noop"], r["gt"], bw), 4)
            if n % 50 == 0 or n == len(tasks):
                rate = n / (time.time() - t0)
                print(f"  [{n}/{len(tasks)}] {rate:.1f} it/s  "
                      f"eta {(len(tasks) - n) / max(rate, 1e-6) / 60:.0f} min",
                      flush=True)

    (args.out / "scores.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows.values()))

    # ---- 汇总报表 ----
    print("\n===== 分 stage 图像空间诊断 =====")
    print(f"{'stage':8} {'n':>4} {'解析失败':>6} {'键异常':>5} "
          f"{'L_pred p50':>10} {'L_noop p50':>10} {'改善中位':>7} {'改善p10':>7}")
    report = {}
    for st in STAGE_NAMES:
        rs = [r for r in rows.values() if r["stage"] == st]
        ok = [r for r in rs if "L_pred" in r]
        pf = sum(1 for r in rs if r.get("parse_fail") or "render_err" in r)
        fl = sum(1 for r in rs if r.get("flags"))
        imp = [1 - r["L_pred"] / max(r["L_noop"], 1e-6) for r in ok]
        ent = {"n": len(rs), "parse_fail": pf, "flagged": fl,
               "L_pred_p50": round(float(np.median([r["L_pred"] for r in ok])), 3) if ok else None,
               "L_noop_p50": round(float(np.median([r["L_noop"] for r in ok])), 3) if ok else None,
               "improve_p50": round(float(np.median(imp)), 3) if ok else None,
               "improve_p10": round(float(np.percentile(imp, 10)), 3) if ok else None,
               "param_mae": {k: round(v[0] / max(v[1], 1), 2)
                             for k, v in sorted(mae_acc.get(st, {}).items())}}
        report[st] = ent
        print(f"{st:8} {ent['n']:>4} {pf:>6} {fl:>5} "
              f"{str(ent['L_pred_p50']):>10} {str(ent['L_noop_p50']):>10} "
              f"{str(ent['improve_p50']):>7} {str(ent['improve_p10']):>7}")
        print(f"         参数MAE: {ent['param_mae']}")
    # 黑白单列（plan：黑白近乎另一个 regime，得单独看）
    bw_ok = [r for r in rows.values() if r.get("bw") and "L_pred" in r]
    if bw_ok:
        imp = [1 - r["L_pred"] / max(r["L_noop"], 1e-6) for r in bw_ok]
        report["bw_subset"] = {"n": len(bw_ok),
                               "improve_p50": round(float(np.median(imp)), 3)}
        print(f"黑白子集  n={len(bw_ok)}  改善中位={report['bw_subset']['improve_p50']}")

    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n[score] 明细 scores.jsonl / 汇总 report.json → {args.out}/")
    print("读法：改善中位 = 1 − L_pred/L_noop，越接近 1 该 stage 学得越好；"
          "≈0 表示不比全零好；哪个 stage 显著落后再考虑拆分它。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["gen", "score", "all"])
    ap.add_argument("--data_dir", type=Path, default=REPO_ROOT / "sft_data_v2")
    ap.add_argument("--out", type=Path, default=None,
                    help="产出目录，默认 <data_dir>/diag（--no_adapter 时 diag_base）")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--task", default="task_v2_taste",
                    help="数据集前缀，文件=<task>_<split>.json（v1 用 task_a）")
    ap.add_argument("--codec", choices=["bins", "json"], default="bins",
                    help="bins=v2 值离散化 token；json=v1 裸 JSON")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", type=Path,
                    default=REPO_ROOT / "5_training/output/sft_v2")
    ap.add_argument("--no_adapter", action="store_true", help="裸基座对照组")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--max_side", type=int, default=1024, help="与数据构建一致")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.out is None:
        args.out = args.data_dir / ("diag_base" if args.no_adapter else "diag")

    if args.phase in ("gen", "all"):
        run_gen(args)
    if args.phase in ("score", "all"):
        run_score(args)


if __name__ == "__main__":
    main()
