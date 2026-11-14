#!/usr/bin/env python3
"""summary_tables.py — 500 图对「原图 vs version_0」：人工标注与模型评审的三张汇总表
================================================================================

要回答两件事：
  ① 数据集质量：人和模型是否都认为修后图（version_0）比原图好；
  ② 模型的局限：位置偏好（正反两序不一致）、以及与人的逐对判断差多少。

三位完整标注者（每人 500 对）：
  A  旧第二轮投票 visitor 1742b70a（2026-04，原始 png、旧投票页）
  B  旧第二轮投票 visitor 3582f64a（同上）
  C  新投票（2026-11，两侧统一 jpg95、中性标签、本地投票页）
旧投票里另一位只投了 39 票的标注者不计入。同一人对同一图对投了多次的，取最后一票。

表 1  人工标注：每人选修后图的比例、三人票型分布、多数票结论
表 2  标注者两两一致性（判断「人工参考」本身有多可靠）
表 3  模型：选修后图比例、位置偏好、正反两序一致率；只在两序一致的图对上与人工比对

模型与人工的比对口径：
  · 人工参考 = 三人多数票；
  · 另报「A、C 两人意见相同」的图对上的结果 —— B 与另两人的一致率等于随机水平（见表 2），
    多数票在 A、C 分歧的图对上实际由 B 决定，所以这个子集是更可靠的参考。

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

from eval_d1_neutral import chosen_of

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_RECORDS = "Analysis_votes/vote_analysis/vote_records_cleaned.csv"
OLD_ANNOTATORS = {"A": "1742b70a", "B": "3582f64a"}
ORDERS = ("original_first", "version_0_first")
V0, ORIG = "version_0", "original"


def load_humans(repo: Path, data: Path) -> dict[str, dict[str, str]]:
    """{标注者: {图对 ID: 'version_0' | 'original'}}，同一图对取最后一票。"""
    ann: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(repo / "7_baselines" / "Analysis_votes.zip") as z:
        rows = list(csv.DictReader(io.StringIO(z.read(OLD_RECORDS).decode("utf-8-sig"))))
    for name, prefix in OLD_ANNOTATORS.items():
        rs = sorted((r for r in rows if r["visitor_id"].startswith(prefix)), key=lambda r: r["timestamp"])
        ann[name] = {r["base_id"]: r["winner_team"] for r in rs}
    ann["C"] = {}
    for line in (data / "votes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            v = json.loads(line)
            ann["C"][v["sample_id"]] = v["chosen"]
    return ann


def load_model(path: Path) -> dict[tuple[str, str], str]:
    """{(图对, 顺序): 选了哪张}，按原始回复重新解析，每个键取最后一条有效记录。"""
    best = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("raw"):
                c = chosen_of(r)[0]
                if c:
                    best[(r["sample_id"], r["order"])] = (c, r["a_is"])
    return best


def kappa(a: dict, b: dict, ids: list[str]) -> tuple[float, float]:
    agree = sum(a[s] == b[s] for s in ids) / len(ids)
    pa = sum(a[s] == V0 for s in ids) / len(ids)
    pb = sum(b[s] == V0 for s in ids) / len(ids)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return agree, (agree - pe) / (1 - pe)


def pct(c: int, n: int) -> str:
    return "—" if not n else f"{100 * c / n:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="人工标注与模型评审的三张汇总表")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()
    repo = args.repo.resolve()
    data = (args.data or repo / "eval_results" / "human500").resolve()

    ann = load_humans(repo, data)
    ids = sorted(ann["C"])
    assert len(ids) == 500 and all(set(ids) == set(ann[k]) for k in ann), "三位标注者应各覆盖同样的 500 对"

    n_v0 = {s: sum(ann[k][s] == V0 for k in "ABC") for s in ids}       # 每对有几人选修后图
    majority = {s: V0 if n_v0[s] >= 2 else ORIG for s in ids}
    ac_same = {s: ann["A"][s] for s in ids if ann["A"][s] == ann["C"][s]}  # A、C 意见相同的图对
    dist = Counter(n_v0.values())
    maj_v0 = sum(v == V0 for v in majority.values())
    ac_v0 = sum(v == V0 for v in ac_same.values())

    md = ["# 500 图对「原图 vs version_0」：人工标注与模型评审汇总", "",
          "## 表 1　人工标注：修后图是否比原图好", "",
          "| 标注者 | 标注条件 | 选修后图 | 选原图 | 选修后图比例 |", "|---|---|---|---|---|"]
    cond = {"A": "旧投票（2026-04）", "B": "旧投票（2026-04）", "C": "新投票（2026-11，本人）"}
    for k in "ABC":
        c = sum(ann[k][s] == V0 for s in ids)
        md.append(f"| {k} | {cond[k]} | {c} | {500 - c} | {pct(c, 500)} |")
    md += [f"| **三人多数票** | 至少 2 人选同一张 | **{maj_v0}** | **{500 - maj_v0}** | **{pct(maj_v0, 500)}** |",
           f"| A、C 意见相同的图对 | 共 {len(ac_same)} 对 | {ac_v0} | {len(ac_same) - ac_v0} | {pct(ac_v0, len(ac_same))} |",
           "",
           "三人票型分布（选修后图 : 选原图）：" +
           "，".join(f"{k}:{3 - k} 有 {dist[k]} 对（{pct(dist[k], 500)}）" for k in (3, 2, 1, 0)) + "。", "",
           "## 表 2　标注者之间的一致性", "",
           "| 标注者对 | 逐对一致率 | 若互相独立的期望一致率 | Cohen κ |", "|---|---|---|---|"]
    for x, y in (("A", "C"), ("A", "B"), ("B", "C")):
        ag, kp = kappa(ann[x], ann[y], ids)
        md.append(f"| {x} – {y} | {100 * ag:.1f}% | {100 * (ag - kp) / (1 - kp) if kp != 1 else 100:.1f}% | {kp:+.2f} |")
    both_v0 = [s for s in ids if ann["A"][s] == V0 and ann["C"][s] == V0]
    b_on = sum(ann["B"][s] == V0 for s in both_v0)
    md += ["",
           f"在 A、C 都选修后图的 {len(both_v0)} 对上，B 选修后图的只有 {b_on} 对（{pct(b_on, len(both_v0))}）。", "",
           "## 表 3　模型评审（每个模型 500 对 × 正反两种顺序）", "",
           "| 模型 | 选修后图（全部判断） | 正反两序一致率 | 一致图对中选修后图 | 与三人多数票一致 | 与 A、C 共识一致 |",
           "|---|---|---|---|---|---|"]

    out_models = {}
    rows_md = []
    for f in sorted((data / "llm_d1").glob("*.jsonl")):
        best = load_model(f)
        if not best:
            continue
        m = f.stem.split("__")[1]
        both = [s for s in ids if (s, ORDERS[0]) in best and (s, ORDERS[1]) in best]
        cons = {s: best[(s, ORDERS[0])][0] for s in both if best[(s, ORDERS[0])][0] == best[(s, ORDERS[1])][0]}
        n_all = len(best)
        v0_all = sum(c == V0 for c, _ in best.values())
        first = sum(c == a_is for c, a_is in best.values())
        v0_cons = sum(c == V0 for c in cons.values())
        ag_maj = sum(cons[s] == majority[s] for s in cons)
        ac_ids = [s for s in cons if s in ac_same]
        ag_ac = sum(cons[s] == ac_same[s] for s in ac_ids)
        ac_orig = [s for s in ac_ids if ac_same[s] == ORIG]
        hit_orig = sum(cons[s] == ORIG for s in ac_orig)
        out_models[m] = {"n_judgments": n_all, "v0_all": v0_all / n_all, "pick_first": first / n_all,
                         "consistent_rate": len(cons) / len(both), "n_consistent": len(cons),
                         "v0_consistent": v0_cons / len(cons), "agree_majority": ag_maj / len(cons),
                         "agree_ac": ag_ac / len(ac_ids), "n_ac": len(ac_ids),
                         "orig_hit": hit_orig, "orig_n": len(ac_orig)}
        rows_md.append((v0_cons / len(cons),
                        f"| {m} | {pct(v0_all, n_all)} | {pct(len(cons), len(both))} | {pct(v0_cons, len(cons))} | "
                        f"{pct(ag_maj, len(cons))} | {pct(ag_ac, len(ac_ids))} |"))
    md += [r for _, r in sorted(rows_md, reverse=True)]
    md.append(f"| *对照：一律选修后图* | 100% | — | — | {pct(maj_v0, 500)} | {pct(ac_v0, len(ac_same))} |")
    ag_ac_h, _ = kappa(ann["A"], ann["C"], ids)
    md += ["",
           "对照：",
           "- 后三列只在模型正反两序一致的图对上统计（各模型 321–402 对）；选第一张的比例、可比对数等见 summary_tables.json。",
           f"- 人与人之间：A 与 C 的逐对一致率是 {100 * ag_ac_h:.1f}%。",
           "- 「正反两序一致率」= 把同一对图的先后顺序对调后，模型仍选同一张图的图对占比；人工投票每对只看一次，没有对应数字。",
           ""]

    (data / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")
    (data / "summary_tables.json").write_text(json.dumps({
        "humans": {k: sum(ann[k][s] == V0 for s in ids) for k in "ABC"},
        "majority_v0": maj_v0, "vote_pattern": {f"{k}:{3 - k}": dist[k] for k in (3, 2, 1, 0)},
        "ac_same": {"n": len(ac_same), "v0": ac_v0}, "models": out_models}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # 逐对明细，方便以后挑图看
    with open(data / "per_pair_labels.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "A", "B", "C", "n_version_0", "majority"])
        for s in ids:
            w.writerow([s, ann["A"][s], ann["B"][s], ann["C"][s], n_v0[s], majority[s]])
    print("\n".join(md))
    print(f"已写出：{data}/summary_tables.md、summary_tables.json、per_pair_labels.csv")


if __name__ == "__main__":
    main()
