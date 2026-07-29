#!/usr/bin/env python3
"""v2_chain_eval.py — 全链评测（主判据）
================================================================================
在完整 val 上跑 **greedy 全链**（模型自渲中间图，与部署一致），报告端到端真实成绩：
    改善 = 1 − D_4/D_0     （D_0=不修到目标的距离，D_4=模型修完到目标的距离）
· 中位/均值/>0 比例 —— 主判据（★2026-07-28：per-stage 报告卡有 GT 上下文托底、会美化模型）
· 灾难率（改善 < −1，即改得比不修还差一倍以上）+ 逐 stage 净贡献 —— 定位重尾与病灶
· 关键参数分布 —— 查晕影/饱和这类「一发毁图」的极值

用法（lf 环境）：
  python 5_training/v2_chain_eval.py --adapter 5_training/output/grpo_chain1 --tag chain1
  python 5_training/v2_chain_eval.py --no_adapter --tag base          # 裸基座对照
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "5_training"))
from v2_codec import SLOTS, VALUE_TOKENS_V, VALUE_TOKENS_EV
from v2_grpo import stage_temps, load_policy, sample_rollout
from v2_grpo_chain import load_units, prep_unit, chain_rollout, group_siblings
from build_sft_dataset import STAGE_NAMES

WATCH = ["晕影", "饱和度", "自然饱和度", "曝光度", "清晰度", "对比度"]


def main():
    from differentiable_renderer.opencv_renderer import render
    from differentiable_renderer.losses import CompositeLoss

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", type=Path, default=REPO / "5_training/output/sft_v2")
    ap.add_argument("--tag", default="model", help="结果标签(写入 out json)")
    ap.add_argument("--out", type=Path, default=REPO / "sft_data_v2/chain_eval.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--big_side", type=int, default=1024)
    ap.add_argument("--small_side", type=int, default=512)
    ap.add_argument("--jpeg_q", type=int, default=92)
    ap.add_argument("--vstar", action="store_true", help="额外报「靠近任一同组专家版本」的口径")
    args = ap.parse_args()

    model, proc = load_policy(args.base, args.adapter, with_ref=False)
    model.processor_tok = proc.tokenizer
    dev = model.device
    tid = proc.tokenizer.convert_tokens_to_ids
    key_ids = {s.token: tid(s.token) for s in SLOTS}
    value_ids = {"v": torch.tensor([tid(t) for t in VALUE_TOKENS_V], device=dev),
                 "ev": torch.tensor([tid(t) for t in VALUE_TOKENS_EV], device=dev)}
    T = {st: stage_temps(st, 0.0) for st in STAGE_NAMES}      # greedy，与温度无关

    lossfn = CompositeLoss(weights={"param_reg": 0.0}).to(dev)
    lossfn_bw = CompositeLoss(weights={**CompositeLoss.BW_LOSS_WEIGHTS, "param_reg": 0.0}).to(dev)
    to_t = lambda im: torch.from_numpy(im.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

    def dist(a, b, bw):
        torch.manual_seed(0)
        with torch.no_grad():
            return float(((lossfn_bw if bw else lossfn)(to_t(a), to_t(b)))[0])

    units = load_units(args.split, args.limit)
    sib = group_siblings(args.split) if args.vstar else None
    imps, imps_v, Dtraj, params, bws = [], [], [], [], []
    t0 = time.time()
    for i, u in enumerate(units):
        p = prep_unit(u, args.big_side, args.small_side, render, sib)
        _, ctxs = chain_rollout(model, proc, p, T, key_ids, value_ids,
                                render, args.jpeg_q, greedy=True)
        D = [dist(render(p["small"], c), p["gt_img"], p["bw"]) for c in [{}] + ctxs]
        imps.append(1 - D[-1] / max(D[0], 1e-6))
        if sib is not None:                       # v* 口径:靶换成最近的同组专家版本(D_0 同靶)
            best = min(p["cands"], key=lambda vc: dist(render(p["small"], ctxs[-1]), vc[1], p["bw"]))[1]
            d0v = dist(render(p["small"], {}), best, p["bw"])
            d4v = dist(render(p["small"], ctxs[-1]), best, p["bw"])
            imps_v.append(1 - d4v / max(d0v, 1e-6))
        Dtraj.append(D); params.append(ctxs[-1]); bws.append(p["bw"])
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(units)}] 改善中位 {np.median(imps):+.3f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/单元)", flush=True)

    imps = np.array(imps); Dtraj = np.array(Dtraj); bws = np.array(bws)
    # 逐 stage 净贡献：该 stage 让距离减少了多少（正=帮忙）
    stage_gain = {st: float(np.median(Dtraj[:, k] - Dtraj[:, k + 1]))
                  for k, st in enumerate(STAGE_NAMES)}
    res = {"tag": args.tag, "adapter": str(args.adapter), "n": len(units),
           "improve_p50": round(float(np.median(imps)), 4),
           "improve_mean": round(float(imps.mean()), 4),
           "frac_pos": round(float((imps > 0).mean()), 4),
           "frac_disaster": round(float((imps < -1).mean()), 4),
           "improve_p10": round(float(np.percentile(imps, 10)), 4),
           "improve_p90": round(float(np.percentile(imps, 90)), 4),
           "bw_p50": round(float(np.median(imps[bws])), 4) if bws.any() else None,
           "stage_gain_p50": {k: round(v, 3) for k, v in stage_gain.items()},
           **({"vstar_improve_p50": round(float(np.median(imps_v)), 4),
               "vstar_frac_pos": round(float((np.array(imps_v) > 0).mean()), 4),
               "vstar_frac_disaster": round(float((np.array(imps_v) < -1).mean()), 4)} if imps_v else {}),
           "param_p50": {k: round(float(np.median([q.get(k, 0) for q in params])), 2) for k in WATCH},
           "param_extreme": {k: round(float(np.mean([abs(q.get(k, 0)) > 80 for q in params])), 3) for k in WATCH}}

    print(f"\n===== 全链评测 [{args.tag}]  n={res['n']} =====")
    print(f"  改善  中位 {res['improve_p50']:+.3f}   均 {res['improve_mean']:+.3f}   "
          f">0 {res['frac_pos']:.0%}   灾难(<−1) {res['frac_disaster']:.0%}")
    print(f"        p10 {res['improve_p10']:+.3f}  p90 {res['improve_p90']:+.3f}  bw中位 {res['bw_p50']}")
    if imps_v:
        print(f"  [v* 口径] 改善中位 {res['vstar_improve_p50']:+.3f}  >0 {res['vstar_frac_pos']:.0%}  "
              f"灾难 {res['vstar_frac_disaster']:.0%}   (靶=最近的同组专家版本)")
    print(f"  逐stage净贡献(正=帮忙): " + "  ".join(f"{k}:{v:+.2f}" for k, v in res["stage_gain_p50"].items()))
    print(f"  参数中位: " + "  ".join(f"{k}:{v:+.1f}" for k, v in res["param_p50"].items()))
    print(f"  极值率(|v|>80): " + "  ".join(f"{k}:{v:.0%}" for k, v in res["param_extreme"].items()))

    allres = json.loads(args.out.read_text()) if args.out.exists() else {}
    allres[args.tag] = res
    args.out.write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
