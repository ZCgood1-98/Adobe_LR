#!/usr/bin/env python3
"""summary_tables.py — 500 图对「原图 vs version_0」：人工标注与模型评审的汇总表
================================================================================

要回答两件事：
  ① 数据集质量：人和模型是否都认为修后图（version_0）比原图好；
  ② 模型的局限：位置偏好（正反两序不一致）、对「颜色更鲜艳」这一简单线索的依赖。

人工标注者（每人完整 500 对；同一人对同一图对投了多次的取最后一票）：
  A  旧第二轮投票 visitor 1742b70a（2026-04，原始 png、旧投票页）
  B  旧第二轮投票 visitor 3582f64a（同上）
  C  本人新投票 votes.jsonl（2026-11，两侧统一 jpg95、中性标签、本地投票页）
  D… 离线包导回的 votes_<名字>.jsonl（2026-11，同一批 jpg95，展示顺序单独打乱）
旧投票里另一位只投了 39 票的标注者不计入。

表 1  人工标注：每人选修后图的比例、合计、图对层面的共识分布
表 2  标注者两两一致性（含置换检验）
表 3  按「修后图比原图更鲜艳 / 更淡」拆分：人和模型选修后图的比例
表 4  模型：选修后图比例、正反两序一致率；只在两序一致的图对上与人工共识比对

人工共识 = 至少 3/4 的标注者意见相同的图对（2:2 的图对不算共识）。
色彩丰富度 = Hasler–Süsstrunk colorfulness，在长边 512 的缩图上计算。

用法（仓库根目录）：
  python 3_llm_eval/human500/summary_tables.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from eval_d1_neutral import chosen_of

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_RECORDS = "Analysis_votes/vote_analysis/vote_records_cleaned.csv"
OLD_ANNOTATORS = {"A": "1742b70a", "B": "3582f64a"}
COND = {"A": "旧投票（2026-04，原始 png）", "B": "旧投票（2026-04，原始 png）",
        "C": "新投票（2026-11，jpg95，本人）"}
ORDERS = ("original_first", "version_0_first")
V0, ORIG = "version_0", "original"
N_PERM = 20000


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_humans(repo: Path, data: Path) -> dict[str, dict[str, str]]:
    """{标注者: {图对 ID: 'version_0' | 'original'}}"""
    ann: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(repo / "7_baselines" / "Analysis_votes.zip") as z:
        rows = list(csv.DictReader(io.StringIO(z.read(OLD_RECORDS).decode("utf-8-sig"))))
    for name, prefix in OLD_ANNOTATORS.items():
        rs = sorted((r for r in rows if r["visitor_id"].startswith(prefix)), key=lambda r: r["timestamp"])
        ann[name] = {r["base_id"]: r["winner_team"] for r in rs}
    ann["C"] = {v["sample_id"]: v["chosen"] for v in load_jsonl(data / "votes.jsonl")}
    for f in sorted(data.glob("votes_*.jsonl")):           # 离线包导回的其他标注者
        name = f.stem[len("votes_"):]
        ann[name] = {v["sample_id"]: v["chosen"] for v in load_jsonl(f)}
        COND.setdefault(name, "离线包（2026-11，jpg95）")
    return ann


def load_model(path: Path) -> dict[tuple[str, str], str]:
    """{(图对, 顺序): 选了哪张}，按原始回复重新解析，每个键取最后一条有效记录。"""
    best = {}
    for r in load_jsonl(path):
        if r.get("raw"):
            c = chosen_of(r)[0]
            if c:
                best[(r["sample_id"], r["order"])] = c
    return best


def colorfulness(p: Path) -> float:
    im = Image.open(p).convert("RGB")
    im.thumbnail((512, 512), Image.LANCZOS)
    x = np.asarray(im, dtype=np.float64)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def colorfulness_delta(data: Path, ids: list[str]) -> dict[str, float]:
    """修后图 − 原图 的色彩丰富度；结果缓存在 image_feats.json。"""
    cache = data / "image_feats.json"
    feats = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if not all(s in feats for s in ids):
        for s in ids:
            if s not in feats:
                feats[s] = {"colorfulness_original": colorfulness(data / "images" / s / "original.jpg"),
                            "colorfulness_version_0": colorfulness(data / "images" / s / "version_0.jpg")}
        cache.write_text(json.dumps(feats, indent=1), encoding="utf-8")
    return {s: feats[s]["colorfulness_version_0"] - feats[s]["colorfulness_original"] for s in ids}


def agreement(a: dict, b: dict, ids: list[str], rng) -> tuple[float, float, float, float]:
    """逐对一致率、互相独立时的期望、Cohen κ、置换检验单侧 p（打乱一方，保持各自选修后图的比例）。"""
    ya = np.array([a[s] == V0 for s in ids]); yb = np.array([b[s] == V0 for s in ids])
    ag = float((ya == yb).mean())
    pa, pb = ya.mean(), yb.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    null = np.array([(rng.permutation(ya) == yb).mean() for _ in range(N_PERM)])
    p = float(((null >= ag).sum() + 1) / (N_PERM + 1))
    return ag, pe, (ag - pe) / (1 - pe), p


def pct(c: float, n: float) -> str:
    return "—" if not n else f"{100 * c / n:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="人工标注与模型评审的汇总表")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    repo = args.repo.resolve()
    data = (args.data or repo / "eval_results" / "human500").resolve()
    rng = np.random.default_rng(args.seed)

    ann = load_humans(repo, data)
    names = list(ann)
    ids = sorted(ann["C"])
    assert len(ids) == 500 and all(set(ids) == set(ann[k]) for k in names), "各标注者应覆盖同样的 500 对"
    n_h = len(names)
    need = n_h // 2 + 1                                       # 共识门槛：严格多数（4 人时为 3）

    n_v0 = {s: sum(ann[k][s] == V0 for k in names) for s in ids}
    consensus = {s: V0 if n_v0[s] >= need else ORIG for s in ids if n_v0[s] >= need or n_h - n_v0[s] >= need}
    cons_v0 = sum(v == V0 for v in consensus.values())
    cons_or = len(consensus) - cons_v0
    tied = 500 - len(consensus)
    total_v0 = sum(n_v0.values())
    dcol = colorfulness_delta(data, ids)
    more = [s for s in ids if dcol[s] > 0]
    less = [s for s in ids if dcol[s] <= 0]

    # ---------------- 表 1 ----------------
    md = ["# 500 图对「原图 vs version_0」：人工标注与模型评审汇总", "",
          "## 表 1　人工标注：修后图是否比原图好", "",
          "| 标注者 | 标注条件 | 选修后图 | 选原图 | 选修后图比例 |", "|---|---|---|---|---|"]
    for k in names:
        c = sum(ann[k][s] == V0 for s in ids)
        md.append(f"| {k} | {COND[k]} | {c} | {500 - c} | {pct(c, 500)} |")
    md += [f"| **{n_h} 人合计** | {500 * n_h} 个判断 | **{total_v0}** | **{500 * n_h - total_v0}** | "
           f"**{pct(total_v0, 500 * n_h)}** |", "",
           f"图对层面（至少 {need}/{n_h} 人意见相同算共识）：共识为修后图更好 {cons_v0} 对（{pct(cons_v0, 500)}），"
           f"共识为原图更好 {cons_or} 对（{pct(cons_or, 500)}），意见对半 {tied} 对（{pct(tied, 500)}）。", ""]

    # ---------------- 表 2 ----------------
    md += ["## 表 2　标注者之间的一致性", "",
           "| 标注者对 | 逐对一致率 | 若互相独立的期望 | Cohen κ | 置换检验 p |", "|---|---|---|---|---|"]
    pairs_stat = []
    for i in range(n_h):
        for j in range(i + 1, n_h):
            x, y = names[i], names[j]
            ag, pe, kp, p = agreement(ann[x], ann[y], ids, rng)
            pairs_stat.append((kp, f"| {x} – {y} | {100 * ag:.1f}% | {100 * pe:.1f}% | {kp:+.2f} | "
                                   f"{'<0.001' if p < 0.001 else f'{p:.3f}'} |"))
    md += [r for _, r in sorted(pairs_stat, reverse=True)]
    md += ["", "置换检验：打乱其中一人的选择（保持他选修后图的比例），看一致率达到实测值的概率；p 小说明两人的一致不是巧合。", ""]

    # ---------------- 模型 ----------------
    models = {}
    for f in sorted((data / "llm_d1").glob("*.jsonl")):
        best = load_model(f)
        if best:
            models[f.stem.split("__")[1]] = best

    def model_score(best, s):
        """一个图对上选修后图的得分：两种顺序各算一次（都选修后图 1，都选原图 0，不一致 0.5）"""
        vals = [best[(s, o)] == V0 for o in ORDERS if (s, o) in best]
        return float(np.mean(vals)) if vals else np.nan

    # ---------------- 表 3 ----------------
    md += ["## 表 3　修后图更鲜艳 vs 更淡：选修后图的比例", "",
           f"按色彩丰富度变化拆分：修后图比原图更鲜艳的 {len(more)} 对，更淡的 {len(less)} 对。"
           "差值越大，说明越是按「哪张颜色更艳」来选。", "",
           "| 评审 | 修后图更鲜艳时 | 修后图更淡时 | 差值 |", "|---|---|---|---|"]
    t3 = {}
    for k in names:
        pm = np.mean([ann[k][s] == V0 for s in more]); pl = np.mean([ann[k][s] == V0 for s in less])
        t3[f"人工 {k}"] = (pm, pl)
        md.append(f"| 人工 {k} | {100 * pm:.1f}% | {100 * pl:.1f}% | {100 * (pm - pl):+.1f} |")
    rows3 = []
    for m, best in models.items():
        pm = np.nanmean([model_score(best, s) for s in more]); pl = np.nanmean([model_score(best, s) for s in less])
        t3[f"模型 {m}"] = (pm, pl)
        rows3.append((pm - pl, f"| 模型 {m} | {100 * pm:.1f}% | {100 * pl:.1f}% | {100 * (pm - pl):+.1f} |"))
    md += [r for _, r in sorted(rows3)]
    md.append("")

    # ---------------- 表 4 ----------------
    md += ["## 表 4　模型评审（每个模型 500 对 × 正反两种顺序）", "",
           "| 模型 | 选修后图（全部判断） | 正反两序一致率 | 一致图对中选修后图 | 与人工共识一致 |",
           "|---|---|---|---|---|"]
    out_models, rows4 = {}, []
    for m, best in models.items():
        both = [s for s in ids if (s, ORDERS[0]) in best and (s, ORDERS[1]) in best]
        cons = {s: best[(s, ORDERS[0])] for s in both if best[(s, ORDERS[0])] == best[(s, ORDERS[1])]}
        n_all = len(best)
        v0_all = sum(c == V0 for c in best.values())
        v0_cons = sum(c == V0 for c in cons.values())
        hc = [s for s in cons if s in consensus]
        ag_c = sum(cons[s] == consensus[s] for s in hc)
        hc_or = [s for s in hc if consensus[s] == ORIG]
        hit_or = sum(cons[s] == ORIG for s in hc_or)
        out_models[m] = {"n_judgments": n_all, "v0_all": v0_all / n_all, "consistent_rate": len(cons) / len(both),
                         "n_consistent": len(cons), "v0_consistent": v0_cons / len(cons),
                         "agree_consensus": ag_c / len(hc), "n_consensus": len(hc),
                         "orig_hit": hit_or, "orig_n": len(hc_or),
                         "v0_more_colorful": t3[f"模型 {m}"][0], "v0_less_colorful": t3[f"模型 {m}"][1]}
        rows4.append((v0_cons / len(cons),
                      f"| {m} | {pct(v0_all, n_all)} | {pct(len(cons), len(both))} | {pct(v0_cons, len(cons))} | "
                      f"{pct(ag_c, len(hc))} |"))
    md += [r for _, r in sorted(rows4, reverse=True)]
    md.append(f"| *对照：一律选修后图* | 100% | — | — | {pct(cons_v0, len(consensus))} |")

    # 留一法：每位人工与其余人的多数票（人数为奇数时无平票）——说明逐对一致率主要由基础比例决定
    loo = {}
    if n_h % 2 == 0:
        for k in names:
            others = [g for g in names if g != k]
            loo[k] = float(np.mean([ann[k][s] == (V0 if sum(ann[g][s] == V0 for g in others) * 2 > len(others) else ORIG)
                                    for s in ids]))
        base = []
        for drop in names:
            grp = [g for g in names if g != drop]
            base.append(np.mean([sum(ann[g][s] == V0 for g in grp) * 2 > len(grp) for s in ids]))
        loo_base = float(np.mean(base))
    md += ["",
           "说明：",
           "- 后三列只在模型正反两序一致的图对上统计；「人工共识」只含至少 "
           f"{need}/{n_h} 人意见相同的 {len(consensus)} 对。",
           "- 正反两序一致率 = 把同一对图的先后顺序对调后，模型仍选同一张图的图对占比。",
           f"- 人工共识为原图更好的 {cons_or} 对里，修后图更淡的占 "
           f"{pct(sum(dcol[s] <= 0 for s in consensus if consensus[s] == ORIG), cons_or)}，模型在这些对上选不选原图主要反映对颜色的依赖（表 3），"
           "因此不单列；数字见 summary_tables.json。"]
    if loo:
        md.append("- 逐对一致率主要由「修后图通常更好」这一基础比例决定：每位人工与其余三人多数票的一致率为 "
                  + "、".join(f"{k} {100 * v:.1f}%" for k, v in sorted(loo.items(), key=lambda x: -x[1]))
                  + f"，都低于一律选修后图（{100 * loo_base:.1f}%）。所以「模型没有超过一律选修后图」不能单独说明模型能力不足。")
    md.append("")

    (data / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")
    (data / "summary_tables.json").write_text(json.dumps({
        "humans": {k: sum(ann[k][s] == V0 for s in ids) for k in names},
        "total_v0": total_v0, "n_judgments": 500 * n_h,
        "consensus": {"threshold": need, "version_0": cons_v0, "original": cons_or, "tied": tied},
        "colorfulness_split": {k: {"more": v[0], "less": v[1]} for k, v in t3.items()},
        "n_more_colorful": len(more), "n_less_colorful": len(less),
        "leave_one_out": loo, "leave_one_out_always_v0": loo_base if loo else None,
        "models": out_models}, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(data / "per_pair_labels.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", *names, "n_version_0", "consensus", "d_colorfulness"])
        for s in ids:
            w.writerow([s, *(ann[k][s] for k in names), n_v0[s], consensus.get(s, "tie"), round(dcol[s], 3)])
    print("\n".join(md))
    print(f"已写出：{data}/summary_tables.md、summary_tables.json、per_pair_labels.csv")


if __name__ == "__main__":
    main()
