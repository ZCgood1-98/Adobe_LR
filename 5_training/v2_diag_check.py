#!/usr/bin/env python3
"""v2_diag_check.py — d2 诊断产出整体验收(只读)
==================================================
读 diag_gen_report.json / diag_d2_checkpoint.jsonl / task_v2_diag_train.json,
一页打印:A 完整性 · B 过滤统计 · C 散文结构 · D bw语义 · E 训练格式/图片存在 · 汇总红旗。

--emit-review 额外产出 diag_d2_review.json(全部被拒 + 抽样通过),供 v2_diag_review.py 可视化翻阅:
  python 5_training/v2_diag_check.py
  python 5_training/v2_diag_check.py --emit-review
  python 5_training/v2_diag_review.py --review sft_data_v2/diag_d2_review.json
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DD = REPO / "sft_data_v2"


def load_jsonl(p: Path) -> list:
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "—"


def bar(title):
    print("=" * 66); print(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-review", action="store_true", help="产出 diag_d2_review.json 供可视化翻阅")
    ap.add_argument("--n_pass_sample", type=int, default=60, help="review 里抽样多少条通过样本")
    args = ap.parse_args()

    report = json.loads((DD / "diag_gen_report.json").read_text()) if (DD / "diag_gen_report.json").exists() else {}
    ck = load_jsonl(DD / "diag_d2_checkpoint.jsonl")
    task = json.loads((DD / "task_v2_diag_train.json").read_text())
    warn = []

    # ---------- A 完整性 ----------
    bar("A. 完整性 / 一致性")
    print(f"  report: total={report.get('total')} ok={report.get('ok')} missing={report.get('missing')} "
          f"errors={report.get('errors_this_run')} passed={report.get('passed')} "
          f"cost=${report.get('cost_usd')} {report.get('elapsed_min')}min")
    vids = [r["vid"] for r in ck]
    dup = [v for v, c in Counter(vids).items() if c > 1]
    ck_pass = [r for r in ck if r.get("passed")]
    rej = [r for r in ck if not r.get("passed")]
    print(f"  checkpoint 行数={len(ck)}  重复vid={len(dup)}  passed={len(ck_pass)}  被拒={len(rej)}")
    print(f"  task 样本={len(task)}  " + ("✓ =passed" if len(ck_pass) == len(task) else "⚠ ≠passed"))
    ck_pass_vids = {r["vid"] for r in ck_pass}
    task_vids = [f"{s['_meta']['group']}/v{s['_meta']['version']}" for s in task]
    miss_in_ck = [v for v in task_vids if v not in ck_pass_vids]
    dup_task = [v for v, c in Counter(task_vids).items() if c > 1]
    print(f"  task vid 不在 checkpoint-passed 的={len(miss_in_ck)}  task 重复vid={len(dup_task)}")
    if report.get("missing"): warn.append(f"report.missing={report['missing']}(有 version 没跑完)")
    if report.get("errors_this_run"): warn.append(f"最后一轮仍有 {report['errors_this_run']} 个失败")
    if dup: warn.append(f"checkpoint {len(dup)} 个重复 vid")
    if len(ck_pass) != len(task): warn.append("passed 数 ≠ task 样本数")
    if miss_in_ck: warn.append(f"{len(miss_in_ck)} 个 task vid 不在 checkpoint-passed")
    if dup_task: warn.append(f"task 有 {len(dup_task)} 个重复 vid")

    # ---------- B 过滤统计 ----------
    bar("B. 过滤统计")
    print(f"  通过 {len(ck_pass)}/{len(ck)} ({pct(len(ck_pass), len(ck))})   被拒 {len(rej)}")
    fatal_axes = Counter(v["axis"] for r in rej for v in r.get("violations", []) if v.get("fatal"))
    print(f"  被拒的 fatal 轴(显著轴/bw): {dict(fatal_axes) or '无'}")
    tol_axes = Counter(v["axis"] for r in ck_pass for v in r.get("violations", [])
                       if v.get("severity") == "hard" and not v.get("fatal"))
    n_tol = sum(1 for r in ck_pass if any(v.get("severity") == "hard" and not v.get("fatal")
                                          for v in r.get("violations", [])))
    print(f"  通过里含'容忍反向'的样本 {n_tol}；容忍轴分布 {dict(tol_axes) or '无'}")
    print("  分类别通过率:")
    for c in sorted({r["category"] for r in ck}):
        sub = [r for r in ck if r["category"] == c]; sp = [r for r in sub if r["passed"]]
        print(f"    {c:12s} {len(sp):>4}/{len(sub):<4} ({pct(len(sp), len(sub))})")
    for bw in (True, False):
        sub = [r for r in ck if r["bw"] == bw]; sp = [r for r in sub if r["passed"]]
        print(f"  bw={str(bw):5s}: {len(sp)}/{len(sub)} ({pct(len(sp), len(sub))})")

    # ---------- C 散文结构(通过样本=训练标签) ----------
    bar("C. 散文结构 (通过样本=训练标签)")
    heads = ("subject & mood", "current look", "target look")
    head_hit = {h: 0 for h in heads}
    n_all3 = 0; leak_plan = []; leak_json = []; wc = []; short = []; nonascii = []
    for r in ck_pass:
        p = r["prose"]; pl = p.lower()
        hits = [h in pl for h in heads]
        for h, hit in zip(heads, hits):
            head_hit[h] += hit
        if all(hits): n_all3 += 1
        if "plan:" in pl or "main issues:" in pl: leak_plan.append(r["vid"])
        if re.search(r'\{\s*"(exposure|wb|bw|saturation)"', p): leak_json.append(r["vid"])
        w = len(p.split()); wc.append(w)
        if w < 20: short.append(r["vid"])
        if sum(1 for ch in p if ord(ch) > 127) > 10: nonascii.append(r["vid"])
    wc.sort()
    q = lambda a: wc[min(int(len(wc) * a), len(wc) - 1)] if wc else 0
    print(f"  三段齐全: {n_all3}/{len(ck_pass)} ({pct(n_all3, len(ck_pass))})")
    print("  各段命中: " + "  ".join(f"{h}={head_hit[h]}" for h in heads))
    print(f"  词数: min={wc[0] if wc else 0} p5={q(.05)} 中位={q(.5)} p95={q(.95)} max={wc[-1] if wc else 0} (目标60-100)")
    print(f"  <40词={sum(1 for w in wc if w < 40)}  >180词={sum(1 for w in wc if w > 180)}  过短<20词={len(short)}")
    print(f"  残留 Plan/Main issues={len(leak_plan)}  JSON泄漏进散文={len(leak_json)}  疑似非英文={len(nonascii)}")
    old = sum(1 for s in task if "four parts" in s["messages"][0]["content"]
              or "retouching plan" in s["messages"][0]["content"].lower())
    print(f"  task 用户提示词用旧四段版的={old} (应为0)")
    if n_all3 < len(ck_pass) * 0.97: warn.append(f"仅 {pct(n_all3, len(ck_pass))} 散文三段齐全")
    if leak_plan: warn.append(f"{len(leak_plan)} 条散文残留 Plan/Main issues")
    if leak_json: warn.append(f"{len(leak_json)} 条散文泄漏方向 JSON")
    if short: warn.append(f"{len(short)} 条散文过短(<20词,疑似截断)")
    if nonascii: warn.append(f"{len(nonascii)} 条疑似非英文")
    if old: warn.append(f"{old} 条用了旧四段 prompt")

    # ---------- D bw 语义 ----------
    bar("D. bw 语义抽查 (Target look 是否与 bw 目标一致)")
    # bw=True 用宽松词:提到任一即算表达了黑白/去色意图
    bw_terms = ("black", "white", "monochrom", "grayscale", "greyscale", "b&w", "desaturat")
    # bw=False 用严格式:只认无歧义的灰度断言,避开 monochrome/monochromatic/desaturated(单色相调色的合法用词)
    strong_bw = re.compile(r"black[- ]and[- ]white|grayscale|greyscale|\bb&w\b", re.I)
    def target_seg(p):
        m = re.search(r"target look\s*:?(.*)$", p, re.I | re.S)
        return (m.group(1) if m else p).lower()
    bw_missing = []; color_saysbw = []
    for r in ck_pass:
        tl = target_seg(r["prose"])
        if r["bw"] and not any(t in tl for t in bw_terms): bw_missing.append(r["vid"])
        # selective-color(去饱和到近黑白但保留局部色相)是合法的彩色编辑,不算误称
        if (not r["bw"]) and strong_bw.search(tl) and "selective" not in tl: color_saysbw.append(r["vid"])
    n_bw = sum(1 for r in ck_pass if r["bw"])
    print(f"  bw=True 共 {n_bw} 条；其 Target look 没提黑白的 {len(bw_missing)} ({pct(len(bw_missing), n_bw)})")
    print(f"  bw=False 却在 Target look 说黑白的 {len(color_saysbw)}")
    if n_bw and len(bw_missing) > n_bw * 0.15: warn.append(f"{len(bw_missing)} 条黑白样本 Target look 未提黑白(占比偏高,建议翻阅)")
    if color_saysbw: warn.append(f"{len(color_saysbw)} 条彩色样本 Target look 误称黑白")

    # ---------- E 训练格式 / 图片存在 ----------
    bar("E. 训练格式 / 图片存在性 (task 文件)")
    bad_schema = 0; missing_img = []; imgs = []
    for s in task:
        m = s.get("messages", [])
        good = (len(m) == 2 and m[0].get("role") == "user" and "<image>" in m[0].get("content", "")
                and m[1].get("role") == "assistant" and m[1].get("content", "").strip()
                and isinstance(s.get("images"), list) and len(s["images"]) == 1)
        if not good: bad_schema += 1
        ip = s["images"][0] if s.get("images") else None
        if ip:
            imgs.append(ip)
            if not (DD / ip).exists(): missing_img.append(ip)
    dup_img = [i for i, c in Counter(imgs).items() if c > 1]
    print(f"  schema 异常={bad_schema}  图片路径缺失={len(missing_img)}  重复图片路径={len(dup_img)}")
    if missing_img: print(f"    缺失示例: {missing_img[:3]}")
    if bad_schema: warn.append(f"{bad_schema} 条 task schema 异常")
    if missing_img: warn.append(f"{len(missing_img)} 条 task 图片文件不存在(训练会炸)")
    if dup_img: warn.append(f"{len(dup_img)} 个重复图片路径")

    # ---------- 可选:产出可视化复审文件 ----------
    if args.emit_review:
        random.seed(42)
        samp = random.sample(ck_pass, min(args.n_pass_sample, len(ck_pass)))
        review = [{"vid": r["vid"], "category": r.get("category"), "bw": r["bw"],
                   "passed": r["passed"], "violations": r.get("violations", []),
                   "prose": r["prose"], "json": r.get("json", {})} for r in rej + samp]
        (DD / "diag_d2_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1))
        print(f"\n[--emit-review] → sft_data_v2/diag_d2_review.json"
              f"（{len(review)} 条 = 全部被拒 {len(rej)} + 抽样通过 {len(samp)}）")
        print("  翻阅: python 5_training/v2_diag_review.py --review sft_data_v2/diag_d2_review.json")

    # ---------- 汇总 ----------
    print("\n" + "=" * 66)
    if warn:
        print(f"⚠ 发现 {len(warn)} 类需关注:")
        for w in warn:
            print("  -", w)
    else:
        print("✅ 全部检查通过,未发现问题")


if __name__ == "__main__":
    main()
