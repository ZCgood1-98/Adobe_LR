#!/usr/bin/env python3
"""tables_full.py — 把 score_full.py 的结果整理成两张说明表（Markdown）

用法：python 6_aes_eval/batch_retouch/tables_full.py <table_xxx.csv> <输出.md> [标题]
"""
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "全数据集批量双库评审"
d = pd.read_csv(src)
CATS = ["人物", "风景", "都市", "植物", "动物", "食物", "创意抽象", "静物产品", "交通工具", "其他"]
CLS = {"对照": "对照", "A": "A 专用修图模型", "B": "B 生图模型", "C": "C 语言模型驱动渲染器", "D": "D 商业软件一键", "专家": "专家"}
ORDER = ["对照", "A", "B", "C", "D", "专家"]
NAME = {"原图_jpg95": "原图另存 jpg95（对原图 png）"}

tot = d[d.category == "全部"].copy()
tot["o"] = tot.cls.map(ORDER.index)
tot = tot.sort_values(["o", "dQ"], ascending=[True, False])
md = [f"# {title}", "",
      "## 表 1　各模型总体", "",
      "| 类别 | 模型 | 组数 | 批次数 | ΔQ | 95% 区间 | ΔQ>0 的批次 | Δ色彩丰富度 |", "|---|---|---|---|---|---|---|---|"]
for r in tot.itertuples():
    md.append(f"| {CLS[r.cls]} | {NAME.get(r.arm, r.arm)} | {r.groups} | {r.batches} | {r.dQ:+.3f} | {r.ci_low:+.3f} ～ {r.ci_high:+.3f} | "
              f"{100 * r.frac_pos:.0f}% | {r.d_colorful:+.1f} |")
ncat = d[(d.arm == "原图_jpg95")].set_index("category").groups.to_dict()
md += ["", "## 表 2　按图片类别的 ΔQ", "",
       "| 类别 | 模型 | " + " | ".join(f"{c}（{ncat.get(c, '—')}）" for c in CATS) + " |", "|---|---|" + "---|" * len(CATS)]
for r in tot.itertuples():
    cells = []
    for c in CATS:
        x = d[(d.arm == r.arm) & (d.cls == r.cls) & (d.category == c)]
        if x.empty:
            cells.append("—")
        else:
            x = x.iloc[0]
            cells.append(f"{x.dQ:+.2f}" + ("†" if x.small else ""))
    md.append(f"| {CLS[r.cls]} | {NAME.get(r.arm, r.arm)} | " + " | ".join(cells) + " |")
md += ["", "说明：",
       "- ΔQ = 输出批的分数 − 同一批原图的分数（每批 64 组），正值表示评审认为输出比原图好；模型输出对比原图 jpg95，专家终图对比原图 png。",
       "- 各模型使用完全相同的公共组与批次；专家每组随机取 1 个版本。",
       "- † 该类别组数不足 64，整类作为一批计算，波动较大。",
       "- 95% 区间为按组重抽 200 次；Δ色彩丰富度为逐组平均，用来看分数是否伴随变鲜艳。",
       ""]
open(dst, "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))
