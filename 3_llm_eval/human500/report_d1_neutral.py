#!/usr/bin/env python3
"""report_d1_neutral.py — 汇总 500 图对 D1 重测（中性标签、正反两序）的结果与花费
================================================================================

读 eval_d1_neutral.py 写出的 llm_d1/<模型>.jsonl，每个 (图对, 顺序) 取最后一条成功记录。

每个模型报告：
  · 选修后图（version_0）的比例：全部判断 / 原图在前 / 修后图在前，及按图对自助法 95% 区间
  · 选第一张（图 A）的比例 —— 位置偏好，无偏应为 50%
  · 正反两序一致率：两次都选了同一张图（而不是同一个位置）的图对占比；
    一致的图对里选修后图的比例 = 剔除位置偏好后的「真实偏好」
  · 四个子维度各自选修后图的比例
  · 构图一致 / 不一致（专家另外裁剪过）两类图对分开的结果
  · 与人工投票的逐对一致率：旧投票（压缩包第二轮，仅取非平局的图对）；
    新投票（votes.jsonl，有多少算多少）
  · 花费：调用次数（含重试）、费用、平均输入 / 输出 token

用法（仓库根目录）：
  python 3_llm_eval/human500/report_d1_neutral.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBDIMS = ("light", "color", "detail", "impact")
ORDERS = ("original_first", "version_0_first")
OLD_VOTES_IN_ZIP = "Analysis_votes/vote_analysis/original_vs_version_0_pair_majority_result.csv"


def load_model(path: Path) -> tuple[dict, dict]:
    """返回 ({(图对, 顺序): 最后一条成功记录}, 花费统计)。花费按**所有**记录累计（失败、重跑的也花了钱）。"""
    best, spend = {}, {"records": 0, "calls": 0, "cost": 0.0, "pt": [], "ct": [], "returned": defaultdict(int),
                       "failed": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        spend["records"] += 1
        spend["calls"] += r.get("n_calls", 0)
        spend["cost"] += r.get("cost") or 0.0
        u = r.get("usage") or {}
        if u.get("prompt_tokens"):
            spend["pt"].append(u["prompt_tokens"]); spend["ct"].append(u.get("completion_tokens") or 0)
        if r.get("returned_model"):
            spend["returned"][r["returned_model"]] += 1
        if r.get("ok"):
            best[(r["sample_id"], r["order"])] = r
        else:
            spend["failed"] += 1
    return best, spend


def boot(vals_by_pair: dict[str, list[float]], n_boot: int, rng) -> tuple[float, float] | None:
    """按图对重采样：每个图对贡献 (命中数, 判断数)。"""
    c = np.array([sum(v) for v in vals_by_pair.values()], dtype=float)
    t = np.array([len(v) for v in vals_by_pair.values()], dtype=float)
    if len(t) < 2:
        return None
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    s = c[idx].sum(1) / t[idx].sum(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def rate(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def score(best: dict, human_old: dict, human_new: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    recs = list(best.values())
    v0 = lambda r: r["chosen"] == "version_0"  # noqa: E731
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[r["sample_id"]].append(float(v0(r)))

    # 正反两序都有结果的图对
    both = {s: (best[(s, ORDERS[0])], best[(s, ORDERS[1])]) for s in {k[0] for k in best}
            if (s, ORDERS[0]) in best and (s, ORDERS[1]) in best}
    consistent = {s: a["chosen"] for s, (a, b) in both.items() if a["chosen"] == b["chosen"]}
    # 不一致 = 两次选了同一个位置：都选 A（第一张）还是都选 B
    flip_first = sum(1 for s, (a, b) in both.items() if a["chosen"] != b["chosen"] and a["chosen"] == a["a_is"])

    def agree(human: dict) -> tuple[float | None, int]:
        """只在模型两序一致、且人工有明确结论的图对上比。"""
        common = [s for s in consistent if s in human]
        return rate(consistent[s] == human[s] for s in common), len(common)

    out = {
        "n_ok": len(recs), "n_pairs_both": len(both),
        "v0_all": rate(map(v0, recs)), "v0_all_ci": boot(by_pair, n_boot, rng),
        "v0_orig_first": rate(v0(r) for r in recs if r["order"] == ORDERS[0]),
        "v0_v0_first": rate(v0(r) for r in recs if r["order"] == ORDERS[1]),
        "pick_first": rate(r["chosen"] == r["a_is"] for r in recs),
        "consistent_rate": rate(s in consistent for s in both),
        "consistent_n": len(consistent),
        "v0_consistent": rate(c == "version_0" for c in consistent.values()),
        "orig_consistent": rate(c == "original" for c in consistent.values()),
        "flip_n": len(both) - len(consistent),
        "flip_pick_first": flip_first,
        "v0_aspect_same": rate(v0(r) for r in recs if not r["aspect_mismatch"]),
        "v0_aspect_diff": rate(v0(r) for r in recs if r["aspect_mismatch"]),
    }
    for k in SUBDIMS:
        out[f"sub_{k}"] = rate(r["chosen_sub"][k] == "version_0" for r in recs if r["chosen_sub"].get(k))
    out["agree_old"], out["agree_old_n"] = agree(human_old)
    out["agree_new"], out["agree_new_n"] = agree(human_new)
    return out


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 500 图对 D1 重测结果与花费")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    repo = args.repo.resolve()
    data = (args.data or repo / "eval_results" / "human500").resolve()

    # 旧人工投票（第二轮，按图对多数票；平局的图对不参与比较）
    human_old, old_stat = {}, {"version_0": 0, "original": 0, "tie": 0}
    with zipfile.ZipFile(repo / "7_baselines" / "Analysis_votes.zip") as z:
        for row in csv.DictReader(io.StringIO(z.read(OLD_VOTES_IN_ZIP).decode("utf-8-sig"))):
            res = row["final_result"]
            if res == "version_0_better":
                human_old[row["base_id"]] = "version_0"; old_stat["version_0"] += 1
            elif res == "original_better":
                human_old[row["base_id"]] = "original"; old_stat["original"] += 1
            else:
                old_stat["tie"] += 1
    # 新人工投票（每对取最后一票）
    human_new = {}
    vp = data / "votes.jsonl"
    if vp.exists():
        for line in vp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                human_new[v["sample_id"]] = v["chosen"]

    S, spend = {}, {}
    for f in sorted((data / "llm_d1").glob("*.jsonl")):
        best, sp = load_model(f)
        if best:
            m = f.stem.replace("__", "/")
            S[m] = score(best, human_old, human_new, args.boot, args.seed)
            spend[m] = sp
    order = sorted(S, key=lambda m: -(S[m]["v0_consistent"] or 0))

    def row(name, fn):
        return f"| {name} | " + " | ".join(fn(m) for m in order) + " |"

    short = lambda m: m.split("/")[1]  # noqa: E731
    md = ["# 500 图对 D1 重测（中性标签、强制二选一、正反两序）", "",
          f"每个模型 500 对 × 2 种顺序；方括号为按图对自助法 95% 区间（{args.boot} 次）。"
          "列按「两序一致的图对里选修后图的比例」从高到低排。", "",
          "| 字段 | " + " | ".join(short(m) for m in order) + " |", "|---|" + "---|" * len(order),
          row("厂商 / 完整 ID", lambda m: f"`{m}`"),
          row("有效判断数（满分 1000）", lambda m: str(S[m]["n_ok"])),
          row("**选修后图，全部判断**", lambda m: f3(S[m]["v0_all"])),
          row("95% 区间", lambda m: "—" if not S[m]["v0_all_ci"] else f"{S[m]['v0_all_ci'][0]:.3f}–{S[m]['v0_all_ci'][1]:.3f}"),
          row("原图在前时选修后图", lambda m: f3(S[m]["v0_orig_first"])),
          row("修后图在前时选修后图", lambda m: f3(S[m]["v0_v0_first"])),
          row("选第一张的比例（无偏 50%）", lambda m: pct(S[m]["pick_first"])),
          row("**正反两序一致率**", lambda m: pct(S[m]["consistent_rate"])),
          row("两序一致的图对数", lambda m: str(S[m]["consistent_n"])),
          row("**一致图对里选修后图**", lambda m: f3(S[m]["v0_consistent"])),
          row("两序不一致的图对数（其中两次都选第一张）", lambda m: f"{S[m]['flip_n']}（{S[m]['flip_pick_first']}）"),
          row("子维度：光影", lambda m: f3(S[m]["sub_light"])),
          row("子维度：色彩", lambda m: f3(S[m]["sub_color"])),
          row("子维度：细节", lambda m: f3(S[m]["sub_detail"])),
          row("子维度：视觉冲击", lambda m: f3(S[m]["sub_impact"])),
          row("构图一致的图对：选修后图", lambda m: f3(S[m]["v0_aspect_same"])),
          row("构图不一致的图对：选修后图", lambda m: f3(S[m]["v0_aspect_diff"])),
          row("与旧人工投票一致率（可比图对数）", lambda m: f"{f3(S[m]['agree_old'])}（{S[m]['agree_old_n']}）"),
          row("与新人工投票一致率（可比图对数）", lambda m: f"{f3(S[m]['agree_new'])}（{S[m]['agree_new_n']}）"),
          "",
          f"旧人工投票（第二轮，每对 2–3 票）：修后图胜 {old_stat['version_0']} 对、原图胜 {old_stat['original']} 对、"
          f"平局 {old_stat['tie']} 对；只在非平局的图对上与模型比较。新人工投票目前 {len(human_new)} 对。",
          "", "## 花费", "",
          "| 模型 | 记录数 | 其中失败 | 实际调用次数（含重试） | 费用（美元） | 平均输入 token | 平均输出 token | 接口返回的模型名 |",
          "|---|---|---|---|---|---|---|---|"]
    tot_cost = tot_calls = 0
    for m in order:
        sp = spend[m]
        tot_cost += sp["cost"]; tot_calls += sp["calls"]
        md.append(f"| `{m}` | {sp['records']} | {sp['failed']} | {sp['calls']} | {sp['cost']:.2f} | "
                  f"{np.mean(sp['pt']) if sp['pt'] else 0:.0f} | {np.mean(sp['ct']) if sp['ct'] else 0:.0f} | "
                  f"{', '.join(f'{k}×{v}' for k, v in sp['returned'].items())} |")
    md.append(f"| **合计** | | | {tot_calls} | **{tot_cost:.2f}** | | | |")
    md.append("")

    (data / "d1_neutral_report.md").write_text("\n".join(md), encoding="utf-8")
    (data / "d1_neutral_summary.json").write_text(json.dumps(
        {"scores": S, "spend": {m: {**{k: v for k, v in sp.items() if k not in ("pt", "ct", "returned")},
                                    "returned": dict(sp["returned"]),
                                    "avg_prompt_tokens": float(np.mean(sp["pt"])) if sp["pt"] else None,
                                    "avg_completion_tokens": float(np.mean(sp["ct"])) if sp["ct"] else None}
                                for m, sp in spend.items()},
         "human_old": old_stat, "human_new_n": len(human_new)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md))
    print(f"已写出：{data}/d1_neutral_report.md、d1_neutral_summary.json")


if __name__ == "__main__":
    main()
