#!/usr/bin/env python3
"""tables_final_model.py — 把 score_final_model.py 的结果整理成说明表，并与 11-19 的旧按图评分对照

用法：python 6_aes_eval/batch_retouch/tables_final_model.py
产物：<bg>/retouch_final/final_model_test/tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

BG = bq.bg_dir()
D = BG / "retouch_final" / "final_model_test"
FT = bq.REPO_DEFAULT / "7_baselines" / "runs" / "FINAL_TEST"
CATS = ["人物", "风景", "都市", "植物", "动物", "食物", "创意抽象", "静物产品", "交通工具", "其他"]
STYLES = [1, 2, 3, 7, 8, 9]
CLS = {"我们": "我们", "我们·风格": "我们·单个风格", "A": "A 专用修图", "B": "B 生图", "C": "C 语言模型渲染", "D": "D 一键修图", "专家": "专家"}
ORDER = ["我们", "我们·风格", "A", "B", "C", "D", "专家"]

r = pd.read_csv(D / "results.csv")
old = json.loads((FT / "summary.json").read_text())
old_ext = {k.split("·", 1)[-1]: v["mean"] for k, v in old["外部基线·同组ΔQ_ref_k1"].items()}
old_ext["最终模型（6 候选平均）"] = old["arms"]["最终模型"]["ΔQ_ref_k1"]["mean"]


def name(a):
    return f"风格 {STYLES[int(a[1:])]}（候选 {a}）" if a.startswith("k") else a


tot = r[r.scope == "全部"].copy()
tot["o"] = tot.cls.map(ORDER.index)
tot = tot.sort_values(["o", "dQ"], ascending=[True, False])
n_main = int(tot.photos.iloc[0])
md = ["# 最终模型重评（封存测试集，最终评测方案 retouch-final）", "",
      f"只重新打分，未重跑模型。主表用所有基线都有输出的 {n_main} 张测试照片，各臂同一批照片、同样划分；每批 64 张、5 种划分，照片等权。", "",
      "## 表 1　总体", "",
      "| 类别 | 方法 | ΔQ（最终方案） | 95% 区间 | ΔQ>0 的批次 | 旧按图评分（11-19） |", "|---|---|---|---|---|---|"]
for x in tot.itertuples():
    o = old_ext.get(x.arm)
    md.append(f"| {CLS[x.cls]} | {name(x.arm)} | {x.dQ:+.3f} | {x.ci_low:+.3f} ～ {x.ci_high:+.3f} | {100 * x.frac_pos:.0f}% | "
              f"{'—' if o is None else f'{o:+.3f}'} |")
a5 = r[r.scope == "全部500张"].iloc[0]
md += ["", f"最终模型在全部 500 张测试照片上：ΔQ {a5.dQ:+.3f}（95% 区间 {a5.ci_low:+.3f} ～ {a5.ci_high:+.3f}）。"]

cmp = tot[tot.arm.isin(old_ext) & ~tot.cls.isin(["我们·风格", "专家"])]
rho = spearmanr(cmp.dQ, [old_ext[a] for a in cmp.arm]).statistic
rank_new = int((cmp.dQ > cmp[cmp.arm == "最终模型（6 候选平均）"].dQ.iloc[0]).sum()) + 1
md += [f"新旧两种评分在这批测试照片上的方法排名一致性（Spearman，{len(cmp)} 个方法）：{rho:.3f}；最终模型在 {len(cmp)} 个方法中排第 {rank_new}。", ""]

ncat = r[(r.arm == "最终模型（6 候选平均）") & r.scope.isin(CATS)].set_index("scope").photos.to_dict()
md += ["## 表 2　按图片类别的 ΔQ（最终方案）", "",
       "| 类别 | 方法 | " + " | ".join(f"{c}（{ncat.get(c, 0)}）" for c in CATS) + " |", "|---|---|" + "---|" * len(CATS)]
for x in tot[tot.cls != "我们·风格"].itertuples():
    cells = []
    for c in CATS:
        y = r[(r.arm == x.arm) & (r.scope == c)]
        cells.append("—" if y.empty else f"{y.iloc[0].dQ:+.2f}" + ("†" if y.iloc[0].small else ""))
    md.append(f"| {CLS[x.cls]} | {name(x.arm)} | " + " | ".join(cells) + " |")
md += ["", "说明：",
       "- ΔQ = 输出批的分数 − 同一批原图的分数；我们的模型对比各条件的原图 q95，基线对比组级原图 q95，专家对比原图 png。",
       "- 我们的模型每张照片随机取 1 个条件；6 个候选分别对应风格库 1/2/3/7/8/9。",
       "- † 该类别测试照片不足 64 张，整类作为一批计算，波动较大。",
       "- 旧按图评分为 11-19 的 ΔQ_ref_k1（同一批 500 张测试照片，基线只作参照），尺度与最终方案不同，只比排名。",
       ""]
(D / "tables.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
