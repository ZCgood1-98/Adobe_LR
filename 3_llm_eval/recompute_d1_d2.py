#!/usr/bin/env python3
"""recompute_d1_d2.py — 从已保存的原始回复离线重算 D1 / D2，输出一张不加权的总表
================================================================================

为什么要重算（`evaluate_llms.py` 自带汇总里的三处问题，全部可离线修正，不花 API）：

  ① D2 丢弃 E 选项：`parse_mcq_answer` 只认 A–D，模型答「E 明显减弱」时回退去正文里
     找字母，多数被误判成 A ⇒ 负向题几乎全错（gemini 正向 0.906 / 负向 0.020）。
  ② D1 不认加粗写法：`PREFER: **B**` 被记成未解析（gpt-5.4 子维度 19%、grok-4 35%），
     个别还会被回退正则错配到别的子维度上。
  ③ 色温标准答案错：563 条里 253 条是相对值（如 +12），代码一律减 5500 ⇒ 永远判「变冷」。
     本脚本对相对值改用其自身正负号；绝对开尔文值相对原图的方向无从得知，不计分。

本脚本只读 `raw_results.json`，不调用任何 API，也不改动原始结果。

口径：
  · 一律按「有有效回复的题」计准确率，无回复/解析不了的单独报比例；
  · 不设权重、不算综合分；
  · 95% 置信区间 = 按「组」重采样的自助法（同组的多道题不独立）。

⚠ 离线修不了的设计问题（读表时要记住）：
  · D1 的 prompt 明文告诉模型哪张是「修图后照片」⇒ 不是盲测；
  · D2 选项是「增强/减弱」，对色温（暖/冷）和色调（洋红/绿）语义含糊；
  · D2 的标准答案是滑块正负号，而终图是多参数叠加的净效果。

用法（仓库根目录）：
  python 3_llm_eval/recompute_d1_d2.py
  python 3_llm_eval/recompute_d1_d2.py --raw eval_results/raw_results.json --out_dir eval_results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── 模型信息 ──────────────────────────────────────────────────────────────────
# OpenRouter ID 取自 evaluate_llms.py 的 MODELS（raw_results.json 本身只存了别名，没存 ID）。
# 其余字段是 2026-11-12 从 OpenRouter 公开接口 /api/v1/models 抓的快照；
# 价格、上下文长度是抓取当天的值，不一定等于 2026-07 评测时的值。
MODEL_INFO: dict[str, dict] = {
    "gpt-5.4": {
        "full_name": "GPT-5.4", "vendor": "OpenAI", "openrouter_id": "openai/gpt-5.4",
        "released": "2026-03-06", "context": 1_050_000,
        "input_modalities": "文本/图像/文件", "price_in": 2.5, "price_out": 15.0,
    },
    "gemini-nano-banana": {
        "full_name": "Gemini 3.1 Flash Image Preview (Nano Banana 2)", "vendor": "Google",
        "openrouter_id": "google/gemini-3.1-flash-image-preview",
        "released": "2026-02-27", "context": 65_536,
        "input_modalities": "文本/图像", "price_in": 0.5, "price_out": 3.0,
    },
    "qwen-3-vl": {
        # ⚠ 这个 ID 在 2026-11-12 的 OpenRouter 列表里查不到（现有 8B / 30B-A3B / 235B-A22B），
        #   所以没有元信息；评测时实际调到的是哪个型号无法从结果文件核实。
        "full_name": "Qwen3-VL 32B Instruct", "vendor": "阿里 Qwen",
        "openrouter_id": "qwen/qwen3-vl-32b-instruct",
        "released": None, "context": None,
        "input_modalities": None, "price_in": None, "price_out": None,
    },
    "llama-4": {
        "full_name": "Llama 4 Maverick", "vendor": "Meta",
        "openrouter_id": "meta-llama/llama-4-maverick",
        "released": "2025-04-06", "context": 524_288,
        "input_modalities": "文本/图像", "price_in": 0.35, "price_out": 1.0,
    },
    "grok-4": {
        "full_name": "Grok 4.20", "vendor": "xAI", "openrouter_id": "x-ai/grok-4.20",
        "released": "2026-04-01", "context": 2_000_000,
        "input_modalities": "文本/图像/文件", "price_in": 1.25, "price_out": 2.5,
    },
    "nova-2": {
        "full_name": "Nova 2 Lite", "vendor": "Amazon", "openrouter_id": "amazon/nova-2-lite-v1",
        "released": "2025-12-03", "context": 1_000_000,
        "input_modalities": "文本/图像/视频/文件", "price_in": 0.3, "price_out": 2.5,
    },
    "claude-sonnet": {
        # ⚠ 这一行在 MODELS 里是注释掉的，ID 按注释内容填；且只跑了 291 组。
        "full_name": "Claude Sonnet 4.6", "vendor": "Anthropic",
        "openrouter_id": "anthropic/claude-sonnet-4.6",
        "released": "2026-02-18", "context": 1_000_000,
        "input_modalities": "文本/图像/文件", "price_in": 3.0, "price_out": 15.0,
    },
}

# ── 解析器 ────────────────────────────────────────────────────────────────────
# 统一思路：标签与冒号、冒号与字母之间允许最多 6 个非字母数字字符（** < [ 空格等），
# 字母后必须是词边界（避免把 "Both" 的 B 当成答案）；有多处匹配时取最后一处（结论在末尾）。
_RE_OVERALL = re.compile(r"(?<![A-Za-z_])PREFER(?!_)\W{0,6}:\W{0,6}([ABC])\b")
_RE_SUB = {
    k: re.compile(rf"PREFER_{k.upper()}\W{{0,6}}:\W{{0,6}}([ABC])\b")
    for k in ("light", "color", "detail", "impact")
}
_RE_ANSWER = re.compile(r"ANSWER\W{0,6}:\W{0,6}([A-E])\b")
# 没写 ANSWER 行、直接抄了选项原文的情况：「E. Significantly weakened/...」
_RE_OPTION_LINE = re.compile(r"(?m)^\W{0,6}([A-E])\.\s+(?:Significantly|Slightly|Virtually)\b")

SUBDIMS = ("light", "color", "detail", "impact")


def _last(rx: re.Pattern, text: str) -> str | None:
    found = rx.findall(text)
    return found[-1] if found else None


def parse_d1(raw: str | None) -> dict:
    """D1 一次成对比较 → {overall, light, color, detail, impact}，取值 A/B/C/None。"""
    if not raw:
        return {k: None for k in ("overall",) + SUBDIMS}
    out = {"overall": _last(_RE_OVERALL, raw)}
    for k in SUBDIMS:
        out[k] = _last(_RE_SUB[k], raw)
    return out


def parse_d2(raw: str | None) -> str | None:
    """D2 单选题 → A/B/C/D/E/None。"""
    if not raw:
        return None
    ans = _last(_RE_ANSWER, raw)
    if ans:
        return ans
    # 回退：正文里只出现了**一个**选项原文行才采信（把 5 个选项全抄一遍的不算）
    opts = set(_RE_OPTION_LINE.findall(raw))
    return opts.pop() if len(opts) == 1 else None


# ── D2 标准答案 ───────────────────────────────────────────────────────────────
NEUTRAL_KELVIN = 5500  # 与 evaluate_llms.py 一致；存下来的 param_value = 原值 − 5500
D2_PARAMS = ["高光", "阴影", "白色色阶", "黑色色阶", "自然饱和度", "饱和度", "去朦胧", "色调", "色温"]
D2_AMBIGUOUS = {"色调", "色温"}  # 「增强/减弱」对这两根轴语义含糊


def d2_ground_truth(q: dict) -> str | None:
    """返回 'pos' / 'neg'；标准答案不可判定时返回 None（该题不计分）。"""
    if q["label"] != "色温":
        return "pos" if q["expected"] == ["A", "B"] else "neg"
    orig = q["param_value"] + NEUTRAL_KELVIN  # 还原成 details.json 里的原值
    if abs(orig) <= 100:          # 相对值（JPEG 原片）：正 = 更暖
        if orig == 0:
            return None
        return "pos" if orig > 0 else "neg"
    return None                    # 绝对开尔文值：相对原图是暖是冷无从得知


# ── 统计 ──────────────────────────────────────────────────────────────────────
def _rate(c: float, n: float) -> float | None:
    return c / n if n else None


def _boot_ci(correct: np.ndarray, total: np.ndarray, n_boot: int, rng) -> tuple[float, float] | None:
    """按组重采样的比率自助法：每组贡献 (答对数, 题数)，统计量 = Σ答对 / Σ题数。"""
    keep = total > 0
    correct, total = correct[keep].astype(float), total[keep].astype(float)
    n = len(total)
    if n < 2:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    stat = correct[idx].sum(1) / total[idx].sum(1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def score_model(samples: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sids = sorted(samples)

    # ---------- D1 ----------
    c = {  # 计数器：[命中, 分母]
        "overall": [0, 0], "d1a": [0, 0], "d1b": [0, 0],
        "edited_first": [0, 0], "edited_second": [0, 0],
        **{k: [0, 0] for k in SUBDIMS},
    }
    n_tests = n_noreply = n_unparsed = n_tie = n_pick_first = n_pick_ab = 0
    sub_asked = sub_unparsed = 0
    five_field_acc: list[float] = []            # 原榜单口径：每次比较 5 个判断的平均
    g_correct = np.zeros(len(sids)); g_total = np.zeros(len(sids))   # 逐组，用于自助法

    for gi, sid in enumerate(sids):
        d1 = samples[sid].get("dim1") or {}
        tests = ([("a", d1.get("d1a"))] if d1.get("d1a") else []) + [("b", t) for t in d1.get("d1b", [])]
        for tag, t in tests:
            n_tests += 1
            raw = t.get("raw")
            if not raw:
                n_noreply += 1
                continue
            p = parse_d1(raw)
            edited = t["edited_option"]            # 修后图是 A（第一张）还是 B（第二张）
            fields = []
            for k in SUBDIMS:
                sub_asked += 1
                if p[k] is None:
                    sub_unparsed += 1
                    continue
                ok = p[k] == edited
                c[k][0] += ok; c[k][1] += 1
                fields.append(ok)
            if p["overall"] is None:
                n_unparsed += 1
            else:
                ok = p["overall"] == edited
                fields.append(ok)
                for key in ("overall", "d1a" if tag == "a" else "d1b",
                            "edited_first" if edited == "A" else "edited_second"):
                    c[key][0] += ok; c[key][1] += 1
                g_correct[gi] += ok; g_total[gi] += 1
                n_tie += p["overall"] == "C"
                if p["overall"] in "AB":
                    n_pick_ab += 1
                    n_pick_first += p["overall"] == "A"
            if fields:
                five_field_acc.append(sum(fields) / len(fields))

    d1 = {
        "n_tests": n_tests,
        "n_valid": c["overall"][1],
        "overall": _rate(*c["overall"]),
        "overall_ci": _boot_ci(g_correct, g_total, n_boot, rng),
        "five_field_mean": float(np.mean(five_field_acc)) if five_field_acc else None,
        "d1a": _rate(*c["d1a"]), "d1a_n": c["d1a"][1],
        "d1b": _rate(*c["d1b"]), "d1b_n": c["d1b"][1],
        **{k: _rate(*c[k]) for k in SUBDIMS},
        "tie_rate": _rate(n_tie, c["overall"][1]),
        "pick_first_rate": _rate(n_pick_first, n_pick_ab),
        "acc_edited_first": _rate(*c["edited_first"]),
        "acc_edited_second": _rate(*c["edited_second"]),
        "no_reply_rate": _rate(n_noreply, n_tests),
        "unparsed_rate": _rate(n_unparsed, n_tests),
        "subdim_unparsed_rate": _rate(sub_unparsed, sub_asked),
    }

    # ---------- D2 ----------
    per = {p: [0, 0] for p in D2_PARAMS}
    dirs = {"pos": [0, 0], "neg": [0, 0]}
    n_asked = n_no_gt = n_gt = n_unparsed2 = n_c = n_say_up = n_pos7 = 0
    g_correct = np.zeros(len(sids)); g_total = np.zeros(len(sids))
    g_correct7 = np.zeros(len(sids)); g_total7 = np.zeros(len(sids))

    for gi, sid in enumerate(sids):
        for q in (samples[sid].get("dim2") or {}).get("per_param", []):
            n_asked += 1
            gt = d2_ground_truth(q)
            if gt is None:
                n_no_gt += 1
                continue
            n_gt += 1
            ans = parse_d2(q.get("raw"))
            if ans is None:
                n_unparsed2 += 1
                continue
            ok = ans in (("A", "B") if gt == "pos" else ("D", "E"))
            per[q["label"]][0] += ok; per[q["label"]][1] += 1
            dirs[gt][0] += ok; dirs[gt][1] += 1
            g_correct[gi] += ok; g_total[gi] += 1
            if q["label"] not in D2_AMBIGUOUS:
                g_correct7[gi] += ok; g_total7[gi] += 1
                n_pos7 += gt == "pos"
            n_c += ans == "C"
            n_say_up += ans in "AB"

    n_valid2 = dirs["pos"][1] + dirs["neg"][1]
    d2 = {
        "n_asked": n_asked, "n_no_gt": n_no_gt, "n_valid": n_valid2,
        "overall": _rate(g_correct.sum(), g_total.sum()),
        "overall_ci": _boot_ci(g_correct, g_total, n_boot, rng),
        "core7": _rate(g_correct7.sum(), g_total7.sum()),
        "core7_ci": _boot_ci(g_correct7, g_total7, n_boot, rng),
        "core7_n": int(g_total7.sum()),
        "pos": _rate(*dirs["pos"]), "pos_n": dirs["pos"][1],
        "neg": _rate(*dirs["neg"]), "neg_n": dirs["neg"][1],
        "say_up_rate": _rate(n_say_up, n_valid2),
        "true_up_rate": _rate(dirs["pos"][1], n_valid2),
        "true_up_rate_core7": _rate(n_pos7, g_total7.sum()),
        "no_change_rate": _rate(n_c, n_valid2),
        "unparsed_rate": _rate(n_unparsed2, n_gt),
        "per_param": {p: {"acc": _rate(*per[p]), "n": per[p][1]} for p in D2_PARAMS},
    }
    return {"n_groups": len(sids), "d1": d1, "d2": d2}


# ── 输出 ──────────────────────────────────────────────────────────────────────
def _f(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _ci(x, ci) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}" if ci is None else f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def build_rows(order: list[str], S: dict) -> list[tuple[str, list[str]]]:
    """总表：每行一个字段，每列一个模型。"""
    I = MODEL_INFO

    def info(key, fmt=str):
        return [("—" if I.get(m, {}).get(key) is None else fmt(I[m][key])) for m in order]

    def price(m):
        i = I.get(m, {})
        return "—" if i.get("price_in") is None else f"${i['price_in']:g} / ${i['price_out']:g}"

    rows: list[tuple[str, list[str]]] = [
        ("**模型信息**", [""] * len(order)),
        ("完整型号", info("full_name")),
        ("厂商", info("vendor")),
        ("OpenRouter ID", [f"`{I[m]['openrouter_id']}`" if m in I else "—" for m in order]),
        ("上架日期", info("released")),
        ("上下文长度（token）", info("context", lambda v: f"{v:,}")),
        ("输入模态", info("input_modalities")),
        ("价格（每百万 token，输入 / 输出）", [price(m) for m in order]),
        ("评测组数", [str(S[m]["n_groups"]) for m in order]),
        ("**D1 审美偏好（原图 vs 修后图）**", [""] * len(order)),
        ("成对比较数（有效 / 全部）", [f"{S[m]['d1']['n_valid']} / {S[m]['d1']['n_tests']}" for m in order]),
        ("总体选修后图 [95% CI]", [_ci(S[m]["d1"]["overall"], S[m]["d1"]["overall_ci"]) for m in order]),
        ("五项判断平均（旧榜单口径）", [_f(S[m]["d1"]["five_field_mean"]) for m in order]),
        ("version_0（D1a）", [_f(S[m]["d1"]["d1a"]) for m in order]),
        ("version_1~3（D1b）", [_f(S[m]["d1"]["d1b"]) for m in order]),
        ("子维度：光影", [_f(S[m]["d1"]["light"]) for m in order]),
        ("子维度：色彩", [_f(S[m]["d1"]["color"]) for m in order]),
        ("子维度：细节", [_f(S[m]["d1"]["detail"]) for m in order]),
        ("子维度：视觉冲击", [_f(S[m]["d1"]["impact"]) for m in order]),
        ("判平局比例", [_pct(S[m]["d1"]["tie_rate"]) for m in order]),
        ("选第一张的比例（无偏应约 50%）", [_pct(S[m]["d1"]["pick_first_rate"]) for m in order]),
        ("修后图放第一张时的准确率", [_f(S[m]["d1"]["acc_edited_first"]) for m in order]),
        ("修后图放第二张时的准确率", [_f(S[m]["d1"]["acc_edited_second"]) for m in order]),
        ("无回复 / 总体未解析", [f"{_pct(S[m]['d1']['no_reply_rate'])} / {_pct(S[m]['d1']['unparsed_rate'])}" for m in order]),
        ("子维度未解析", [_pct(S[m]["d1"]["subdim_unparsed_rate"]) for m in order]),
        ("**D2 参数方向感知（增强还是减弱）**", [""] * len(order)),
        ("题数（有效 / 有标准答案 / 全部）", [f"{S[m]['d2']['n_valid']} / {S[m]['d2']['n_asked'] - S[m]['d2']['n_no_gt']} / {S[m]['d2']['n_asked']}" for m in order]),
        ("方向准确率，全部 9 个参数 [95% CI]", [_ci(S[m]["d2"]["overall"], S[m]["d2"]["overall_ci"]) for m in order]),
        ("方向准确率，去掉色温和色调 [95% CI]", [_ci(S[m]["d2"]["core7"], S[m]["d2"]["core7_ci"]) for m in order]),
        ("对照：一律答「增强」能得的分（9 个参数 / 去掉色温色调）", [f"{_f(S[m]['d2']['true_up_rate'])} / {_f(S[m]['d2']['true_up_rate_core7'])}" for m in order]),
        ("正向题（应答增强）", [_f(S[m]["d2"]["pos"]) for m in order]),
        ("负向题（应答减弱）", [_f(S[m]["d2"]["neg"]) for m in order]),
        ("答「增强」的比例（真实正向占比）", [f"{_pct(S[m]['d2']['say_up_rate'])}（{_pct(S[m]['d2']['true_up_rate'])}）" for m in order]),
        ("答「几乎没变」的比例", [_pct(S[m]["d2"]["no_change_rate"]) for m in order]),
        ("未解析比例", [_pct(S[m]["d2"]["unparsed_rate"]) for m in order]),
    ]
    for p in D2_PARAMS:
        note = "（仅相对值，已改标准答案）" if p == "色温" else ""
        rows.append((f"逐参数：{p}{note}", [_f(S[m]["d2"]["per_param"][p]["acc"]) for m in order]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="离线重算 D1 / D2，输出不加权总表")
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "eval_results" / "raw_results.json")
    ap.add_argument("--out_dir", type=Path, default=REPO_ROOT / "eval_results")
    ap.add_argument("--boot", type=int, default=2000, help="自助法重采样次数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    R = json.loads(args.raw.read_text(encoding="utf-8"))
    S = {m: score_model(res, args.boot, args.seed) for m, res in R.items() if res}

    # 列顺序：按 D1 总体从高到低（只为排版，不代表综合排名）
    order = sorted(S, key=lambda m: -(S[m]["d1"]["overall"] or 0))
    rows = build_rows(order, S)

    # Markdown：字段为行、模型为列（7 个模型 × 40 多个字段，这样才看得下）
    md = ["# D1 / D2 重算总表（不加权）", "",
          f"数据：`{args.raw.name}`，离线重新解析原始回复；准确率按有有效回复的题计；"
          f"方括号内为按组自助法 95% 置信区间（{args.boot} 次）。列按 D1 总体排序，仅为排版。", "",
          "| 字段 | " + " | ".join(order) + " |",
          "|---|" + "---|" * len(order)]
    md += [f"| {name} | " + " | ".join(vals) + " |" for name, vals in rows]
    md += ["",
           "读表须知：",
           "- D1 的 prompt 明文标出了哪张是修图后照片，不是盲测。",
           "- D2 选项是「增强 / 减弱」，对色温（暖 / 冷）和色调（洋红 / 绿）语义含糊，这两行接近随机不代表模型看不出。",
           "- D2 要和「一律答增强」的对照行比：正向题约占 59%，不看图也能拿到这个分。",
           "- D2 标准答案是滑块正负号，终图是多参数叠加的净效果，两者不总一致。",
           "- 色温只保留相对值的题并改用其自身正负号；绝对开尔文值的题无法判定方向，未计分。",
           "- 模型元信息是 2026-11-12 的 OpenRouter 快照；`qwen/qwen3-vl-32b-instruct` 当天已查不到；"
           "claude-sonnet 的 ID 取自脚本里被注释掉的一行，且只跑了 291 组。", ""]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "d1_d2_table.md").write_text("\n".join(md), encoding="utf-8")

    # CSV：每行一个模型，方便后续用 pandas 处理
    with open(args.out_dir / "d1_d2_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型"] + [re.sub(r"\*\*", "", name) for name, _ in rows if not name.startswith("**")])
        for j, m in enumerate(order):
            w.writerow([m] + [vals[j].replace("`", "") for name, vals in rows if not name.startswith("**")])

    (args.out_dir / "d1_d2_summary.json").write_text(
        json.dumps({m: {"info": MODEL_INFO.get(m), **S[m]} for m in order}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    print("\n".join(md))
    print(f"已写出：{args.out_dir}/d1_d2_table.md、d1_d2_table.csv、d1_d2_summary.json")


if __name__ == "__main__":
    main()
