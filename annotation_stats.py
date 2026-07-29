#!/usr/bin/env python3
"""fit_annotations.json 标注结果统计。

输出三部分：
  1. 总体概况：组数（train/test）、version 总数、整组舍弃、保留/舍弃、黑白数量
  2. 分类统计（剔除「整组舍弃」的组）：每个类别的组数、train/test 分布、version 保留/舍弃
  3. 异常检查：未整组舍弃、还有保留 version、却没选内容分类的组（正常应为 0）

用法：
    python annotation_stats.py                     # 默认读脚本同级 fit_annotations.json
    python annotation_stats.py --json 路径.json
"""
import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def decision_of(v):
    """兼容新旧两种 version 记录格式，归一化为 'keep' / 'discard'。"""
    if isinstance(v, dict):
        return "discard" if v.get("decision") == "discard" else "keep"
    return "discard" if v == "discard" else "keep"


def bw_of(v):
    """该 version 是否勾了黑白（旧字符串格式无逐 version 黑白，按 False 处理）。"""
    return bool(v.get("bw")) if isinstance(v, dict) else False


def disp_width(s):
    """终端显示宽度（CJK 全角算 2 列），用于含中文的表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s, width):
    """按显示宽度左对齐补空格。"""
    return s + " " * max(0, width - disp_width(s))


def main():
    ap = argparse.ArgumentParser(description="fit_annotations.json 标注结果统计")
    ap.add_argument("--json", type=Path, default=REPO_ROOT / "fit_annotations.json",
                    help="标注结果 JSON 路径")
    args = ap.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))

    # ===== 1. 总体概况 =====
    n_groups = len(data)
    n_done = sum(1 for r in data.values() if r.get("done"))
    split_groups = defaultdict(int)
    n_versions = 0
    n_keep = n_disc = n_bw_keep = 0
    n_disc_groups = 0        # 整组舍弃的组数
    disc_group_versions = 0  # 这些组名下的 version 数
    for key, rec in data.items():
        split_groups[key.split("/", 1)[0]] += 1
        vs = rec.get("versions") or {}
        n_versions += len(vs)
        if rec.get("discarded"):
            # 整组舍弃：导出时整组剔除，其 version 不计入保留/舍弃细分
            n_disc_groups += 1
            disc_group_versions += len(vs)
            continue
        for v in vs.values():
            if decision_of(v) == "keep":
                n_keep += 1
                n_bw_keep += bw_of(v)
            else:
                n_disc += 1

    print("=" * 66)
    print(f"标注文件：{args.json}")
    print("=" * 66)
    split_str = " / ".join(f"{s} {n}" for s, n in sorted(split_groups.items(), reverse=True))
    print(f"总组数       ：{n_groups}（{split_str}），已完成标注 {n_done}")
    print(f"version 总数 ：{n_versions}")
    print(f"整组舍弃     ：{n_disc_groups} 组（名下 {disc_group_versions} 个 version，导出时整组剔除）")
    print(f"有效组       ：{n_groups - n_disc_groups} 组，version 保留 {n_keep} / 舍弃 {n_disc}")
    print(f"保留中黑白   ：{n_bw_keep} 个 version")

    # ===== 2. 分类统计（不含整组舍弃的组）=====
    stats = defaultdict(lambda: defaultdict(int))  # 类别 -> 各项计数
    for key, rec in data.items():
        if rec.get("discarded"):
            continue
        cat = rec.get("category") or "（未分类）"
        st = stats[cat]
        st["groups"] += 1
        st[key.split("/", 1)[0]] += 1  # train / test 组数
        for v in (rec.get("versions") or {}).values():
            st["versions"] += 1
            if decision_of(v) == "keep":
                st["keep"] += 1
                st["bw"] += bw_of(v)
            else:
                st["disc"] += 1

    print()
    print("分类统计（已剔除整组舍弃的组）")
    header = (pad("类别", 12) + f"{'组数':>6}{'train':>8}{'test':>7}"
              f"{'version':>10}{'保留':>7}{'舍弃':>7}{'黑白(保留中)':>13}")
    print(header)
    print("-" * disp_width(header))
    rows = sorted(stats.items(), key=lambda kv: -kv[1]["groups"])
    for cat, st in rows:
        print(pad(cat, 12) + f"{st['groups']:>7}{st['train']:>9}{st['test']:>8}"
              f"{st['versions']:>11}{st['keep']:>8}{st['disc']:>8}{st['bw']:>12}")
    print("-" * disp_width(header))
    tot = defaultdict(int)
    for st in stats.values():
        for k, v in st.items():
            tot[k] += v
    print(pad("合计", 12) + f"{tot['groups']:>7}{tot['train']:>9}{tot['test']:>8}"
          f"{tot['versions']:>11}{tot['keep']:>8}{tot['disc']:>8}{tot['bw']:>12}")

    # ===== 3. 异常检查：没分类但还有保留 version 的组 =====
    need_fix, harmless = [], 0
    for key, rec in data.items():
        if rec.get("discarded") or rec.get("category"):
            continue
        kept = sum(1 for v in (rec.get("versions") or {}).values() if decision_of(v) == "keep")
        if kept > 0:
            need_fix.append((key, kept))
        else:
            harmless += 1  # 未分类但所有 version 都舍弃了，导出时无影响
    print()
    if need_fix:
        print(f"⚠ 异常：{len(need_fix)} 组未选内容分类、但仍有保留 version（需回标注工具补标）：")
        for key, kept in need_fix[:20]:
            print(f"    {key}（保留 {kept} 个 version）")
        if len(need_fix) > 20:
            print(f"    …… 其余 {len(need_fix) - 20} 组略")
    else:
        print("✓ 异常检查通过：所有含保留 version 的有效组都已选内容分类")
    if harmless:
        print(f"（另有 {harmless} 组未分类、但其 version 已全部舍弃，导出时无影响）")


if __name__ == "__main__":
    main()
