#!/usr/bin/env python3
"""诊断结果深度分析（纯读 JSON，任意环境可跑，无 GPU/cv2 依赖）。

在 diagnose_stage_error.py 产出 report.json / predictions.jsonl / scores.jsonl 之后运行：
    python 5_training/analyze_diag.py

输出四部分：
  1) 分 stage 成绩单（改善中位 = 1 − L_pred/L_noop，1=像专家，0=等于不动，负=帮倒忙）
  2) 预测 vs GT 的相关性与保守度（模型敢不敢动、动的方向对不对）
  3) light 难例清单（GT 大幅曝光调整时模型的真实反应）
  4) detail 晕影问题（预测分布 vs 训练 GT 分布）
"""
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAG = REPO_ROOT / "sft_data_v2" / "diag"


def load_all():
    report = json.loads((DIAG / "report.json").read_text())
    preds = {p["id"]: json.loads(p["raw"])
             for p in map(json.loads, open(DIAG / "predictions.jsonl"))}
    scores = {r["id"]: r for r in map(json.loads, open(DIAG / "scores.jsonl"))}
    gt = {}
    for s in json.loads((REPO_ROOT / "sft_data_v2" / "task_a_val.json").read_text()):
        m = s["_meta"]
        gt[f"{m['group']}/v{m['version']}/{m['stage']}"] = json.loads(s["messages"][2]["content"])
    return report, preds, scores, gt


def sec(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main():
    report, preds, scores, gt = load_all()

    # ---- 1) 成绩单 ----
    sec("1) 分 stage 成绩单")
    print(f"{'stage':8} {'改善中位':>8} {'改善p10':>8} {'L_pred':>7} {'L_noop':>7}   解读")
    verdict = {"light": "方向对/幅度缩/常不动", "tone": "保守(幅度~4成)",
               "color": "非黑白≈全零", "detail": "满级晕影毁图"}
    for st in ["light", "tone", "color", "detail"]:
        e = report[st]
        print(f"{st:8} {e['improve_p50']:>8} {e['improve_p10']:>8} "
              f"{e['L_pred_p50']:>7} {e['L_noop_p50']:>7}   {verdict[st]}")
    print(f"黑白子集 (n={report['bw_subset']['n']}): 改善中位 {report['bw_subset']['improve_p50']}"
          "  ← 学会了「黑白→拉满去饱和」")

    # ---- 2) 相关性与保守度 ----
    sec("2) 预测 vs GT：方向(相关系数 r)与胆量(幅度比)")

    def stage_vec(st, getter):
        g, p = [], []
        for k, pd in preds.items():
            if k.endswith("/" + st) and k in gt:
                g.append(getter(gt[k])); p.append(getter(pd))
        return np.array(g), np.array(p)

    ge, pe = stage_vec("light", lambda d: float(d.get("exposure", 0) or 0))
    gtp, ptp = stage_vec("light", lambda d: float(d.get("temperature", 5500) or 5500))
    print(f"light.exposure:    r={np.corrcoef(ge, pe)[0,1]:.2f}  "
          f"|pred|<0.05 占 {(np.abs(pe) < 0.05).mean()*100:.0f}%  (GT |中位|={np.median(np.abs(ge)):.2f})")
    print(f"light.temperature: r={np.corrcoef(gtp, ptp)[0,1]:.2f}  "
          f"pred==5500(不动) 占 {(ptp == 5500).mean()*100:.0f}%")
    for st, keys in [("tone", ["highlights", "shadows", "contrast", "whites", "blacks"]),
                     ("color", ["vibrance", "saturation"])]:
        gm, pm = stage_vec(st, lambda d, K=keys: float(np.mean([abs(float(d.get(x, 0) or 0)) for x in K])))
        ratio = np.median(pm) / max(np.median(gm), 1e-6)
        print(f"{st:5} 平均|幅度|:  GT中位 {np.median(gm):.1f} → 预测中位 {np.median(pm):.1f}"
              f"（只敢动 GT 的 {ratio:.0%}）")

    # ---- 3) light 难例 ----
    sec("3) light 难例（|GT 曝光|>0.8 EV）：GT vs 预测")
    hard = [(gt[k]["exposure"], preds[k].get("exposure", 0),
             1 - scores[k]["L_pred"] / max(scores[k]["L_noop"], 1e-6))
            for k in preds if k.endswith("/light") and k in gt
            and abs(gt[k]["exposure"]) > 0.8 and "L_pred" in scores.get(k, {})]
    hard.sort(key=lambda x: -abs(x[0]))
    print(f"{'GT曝光':>7} {'预测':>7} {'图像改善':>8}")
    for g, p, imp in hard[:12]:
        print(f"{g:>7.2f} {p:>7.2f} {imp:>8.3f}")
    imps = np.array([x[2] for x in hard])
    print(f"难例改善中位 {np.median(imps):.3f}，>0 占比 {(imps > 0).mean()*100:.0f}% (n={len(imps)})")

    # ---- 4) detail 晕影 ----
    sec("4) detail 晕影：模型输出 vs 训练 GT 分布")
    pv = np.array([abs(float(preds[k].get("vignette", 0) or 0))
                   for k in preds if k.endswith("/detail")])
    train_v = np.array([abs(json.loads(s["messages"][2]["content"])["vignette"])
                        for s in json.loads((REPO_ROOT / "sft_data_v2" / "task_a_train.json").read_text())
                        if s["_meta"]["stage"] == "detail"])
    print(f"模型预测 |vignette|: 中位 {np.median(pv):.0f}，>90 占 {(pv > 90).mean()*100:.0f}%")
    print(f"训练 GT  |vignette|: 中位 {np.median(train_v):.0f}，>90 占 {(train_v > 90).mean()*100:.0f}%")
    print("→ 训练数据没这么疯，是模型在重尾标签上塌缩到 ±100 —— GRPO 前建议冻结/夹紧 detail")


if __name__ == "__main__":
    main()
