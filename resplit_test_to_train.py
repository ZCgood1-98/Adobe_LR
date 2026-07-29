#!/usr/bin/env python3
"""按类别比例把 test 集缩减到目标组数，多余的组移入 train。

规则：
  - 只做 test -> train 的单向移动，不往 test 新增任何图片；
  - 两棵树同步移动，保持源数据与拟合结果的 split 一致：
        fit_results/test/<ID>       -> fit_results/train/<ID>
        try_new_dataset1/test/<ID>  -> try_new_dataset1/train/<ID>
  - 留在 test 的组按 --match 指定的参照比例分层抽样（largest remainder 取整 + 固定随机种子）：
        --match all   参照「整个数据集（train+test 全部有效组）」的类别占比（默认）
        --match test  参照「原 test 可用组自身」的类别占比
    若某类别名额超过 test 池的可用上限，则按上限封顶、多余名额按比例转给其余类别；
  - 「可用」= 未整组舍弃、有内容分类、≥1 个保留 version、组内有原图、源数据集有对应目录。
    不可用的 test 组一律移入 train —— 它们导出时本来就会被过滤，留在 test 只会虚占名额；
  - fit_annotations.json 的键同步改写 "test/<ID>" -> "train/<ID>"（先备份、原子写回）；
  - 全部动作写入 resplit_log_*.json，可按日志把目录移回、键改回以完全恢复。

用法：
    python resplit_test_to_train.py               # 预览（默认 dry-run，不动任何文件）
    python resplit_test_to_train.py --execute     # 实际执行
    可选：--test-target 500  --seed 42  --match all|test
"""
import argparse
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ORIG_NAME = "original_captured_cropped.png"


def annotator_running():
    """粗查标注 Flask 服务是否在运行——它会在下次保存时整体覆写标注 JSON，必须先停。"""
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "annotate_fit_results.py" in cmd and "python" in cmd:
            return cmd.strip()
    return None


def kept_count(rec):
    """一组里「保留」的 version 数（兼容旧字符串格式）。"""
    n = 0
    for v in (rec.get("versions") or {}).values():
        d = v.get("decision") if isinstance(v, dict) else v
        n += d != "discard"
    return n


def allocate(weights, total, caps):
    """largest remainder 法按 weights 的比例分 total 个名额。

    caps 是各类别的可用上限（test 池大小）：某类别名额超上限时按上限封顶、
    从候选中移除，多余名额下一轮按剩余类别的比例继续分，直到分完。
    """
    final = {}
    active = {c: w for c, w in weights.items() if w > 0 and caps.get(c, 0) > 0}
    remaining = total
    while remaining > 0 and active:
        grand = sum(active.values())
        quotas = {c: active[c] * remaining / grand for c in active}
        alloc = {c: int(q) for c, q in quotas.items()}
        rest = remaining - sum(alloc.values())
        for c in sorted(quotas, key=lambda c: quotas[c] - alloc[c], reverse=True)[:rest]:
            alloc[c] += 1
        over = [c for c in active if alloc[c] > caps[c]]
        if not over:
            for c, n in alloc.items():
                final[c] = final.get(c, 0) + n
            break
        for c in over:  # 封顶后移出候选，名额转给其余类别
            final[c] = caps[c]
            remaining -= caps[c]
            del active[c]
    assert sum(final.values()) == total, "名额分配不完整（可用池不足）"
    return final


def main():
    ap = argparse.ArgumentParser(description="按类别比例把 test 集缩减到目标组数，多余组移入 train")
    ap.add_argument("--fit-root", type=Path, default=REPO_ROOT / "fit_results", help="拟合结果根目录")
    ap.add_argument("--src-root", type=Path, default=REPO_ROOT / "try_new_dataset1", help="源数据集根目录")
    ap.add_argument("--anno", type=Path, default=REPO_ROOT / "fit_annotations.json", help="标注 JSON")
    ap.add_argument("--test-target", type=int, default=500, help="缩减后 test 组数")
    ap.add_argument("--match", choices=["all", "test"], default="all",
                    help="类别比例参照：all=整个数据集有效组占比（默认），test=原 test 自身占比")
    ap.add_argument("--seed", type=int, default=42, help="分层抽样随机种子")
    ap.add_argument("--execute", action="store_true", help="实际执行（默认只预览）")
    args = ap.parse_args()

    if args.execute:
        cmd = annotator_running()
        if cmd:
            raise SystemExit(f"检测到标注服务正在运行，请先停掉它再执行：\n  {cmd}")

    anno = json.load(open(args.anno, encoding="utf-8"))

    # ---- 磁盘与标注对账（只看 test 侧）----
    disk_test = sorted(
        p.name for p in (args.fit_root / "test").iterdir()
        if p.is_dir() and any(v.is_dir() and v.name.startswith("version_") for v in p.iterdir())
    )
    anno_test = sorted(k.split("/", 1)[1] for k in anno if k.startswith("test/"))
    if disk_test != anno_test:
        only_disk = set(disk_test) - set(anno_test)
        only_anno = set(anno_test) - set(disk_test)
        raise SystemExit(f"test 磁盘与标注不一致，先解决再重划分。仅磁盘有: {sorted(only_disk)[:5]}…"
                         f" 仅标注有: {sorted(only_anno)[:5]}…")

    # ---- 把每个 test 组分成「可用（进抽样池）」和「必移（不可用）」----
    pool = defaultdict(list)   # 类别 -> [id]
    forced = []                # [(id, 原因)]
    for gid in disk_test:
        rec = anno[f"test/{gid}"]
        if rec.get("discarded"):
            forced.append((gid, "整组舍弃")); continue
        if not rec.get("category"):
            forced.append((gid, "无内容分类")); continue
        if kept_count(rec) == 0:
            forced.append((gid, "无保留version")); continue
        if not (args.fit_root / "test" / gid / ORIG_NAME).is_file():
            forced.append((gid, "缺原图")); continue
        if not (args.src_root / "test" / gid).is_dir():
            forced.append((gid, "缺源目录")); continue
        pool[rec["category"]].append(gid)

    n_pool = sum(len(v) for v in pool.values())
    if args.test_target > n_pool:
        raise SystemExit(f"可用 test 组只有 {n_pool}，小于目标 {args.test_target}")

    # ---- 分层抽样：按参照比例分配名额，组内固定种子随机选 ----
    counts = {c: len(v) for c, v in pool.items()}  # test 池各类别可用上限
    if args.match == "all":
        # 参照整个数据集（train+test）全部有效组的类别占比
        weights = dict(Counter(r["category"] for r in anno.values()
                               if not r.get("discarded") and r.get("category")))
    else:
        weights = counts
    targets = allocate(weights, args.test_target, counts)
    rnd = random.Random(args.seed)
    stay, sampled_out = {}, []
    for cat in sorted(pool):
        ids = sorted(pool[cat])
        chosen = set(rnd.sample(ids, targets.get(cat, 0)))
        stay[cat] = sorted(chosen)
        sampled_out += [(g, "分层抽样未选中") for g in ids if g not in chosen]

    movers = forced + sampled_out  # 全部要移入 train 的组

    # ---- 打印计划 ----
    ref_name = "整个数据集有效组" if args.match == "all" else "原 test 可用组"
    grand_w = sum(weights.values())
    print(f"test 现有 {len(disk_test)} 组：可用 {n_pool} + 不可用 {len(forced)}")
    print(f"目标：test 保留 {args.test_target} 组，移入 train {len(movers)} 组"
          f"（seed={args.seed}，比例参照：{ref_name}）\n")
    print(f"{'类别':<6}{'参照占比':>9}{'test池':>8}{'名额':>6}{'新test占比':>11}")
    for cat in sorted(weights, key=weights.get, reverse=True):
        n = targets.get(cat, 0)
        capped = "（池不足封顶）" if n == counts.get(cat, 0) and n < round(weights[cat]/grand_w*args.test_target) else ""
        print(f"{cat:<6}{weights[cat]/grand_w:>8.1%}{counts.get(cat,0):>8}{n:>6}"
              f"{n/args.test_target:>10.1%}{capped}")
    reasons = Counter(r for _, r in movers)
    print(f"\n移动原因: {dict(reasons)}")

    if not args.execute:
        print("\n[dry-run] 未做任何改动。确认无误后加 --execute 执行。")
        return

    # ---- 执行前检查：目标位置不能已存在 ----
    for gid, _ in movers:
        for root in (args.fit_root, args.src_root):
            if (root / "train" / gid).exists():
                raise SystemExit(f"目标已存在，中止：{root/'train'/gid}")

    # ---- 移动目录（两棵树），随后改标注键，最后写日志 ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    moved_records = []
    for gid, reason in movers:
        (args.fit_root / "test" / gid).rename(args.fit_root / "train" / gid)
        src_moved = (args.src_root / "test" / gid).is_dir()
        if src_moved:
            (args.src_root / "test" / gid).rename(args.src_root / "train" / gid)
        moved_records.append({"id": gid, "reason": reason, "src_moved": src_moved})

    moved_ids = {gid for gid, _ in movers}
    backup = args.anno.with_suffix(args.anno.suffix + ".bak_before_resplit")
    shutil.copy2(args.anno, backup)
    new_anno = {}
    for k, v in anno.items():  # 保持原键序，只改被移动组的前缀
        split, gid = k.split("/", 1)
        if split == "test" and gid in moved_ids:
            k = f"train/{gid}"
        new_anno[k] = v
    tmp = args.anno.with_suffix(args.anno.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_anno, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.anno)

    log_path = REPO_ROOT / f"resplit_log_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "time": ts, "seed": args.seed, "test_target": args.test_target,
            "match": args.match, "ref_weights": weights,
            "fit_root": str(args.fit_root), "src_root": str(args.src_root),
            "pool_counts": counts, "targets": targets, "stay": stay,
            "moved": moved_records, "anno_backup": str(backup),
            "restore_hint": "恢复 = 把 moved 里每个 id 的两棵树目录从 train 移回 test，"
                            "并把标注键 train/<id> 改回 test/<id>（或直接用 anno_backup 覆盖）",
        }, f, ensure_ascii=False, indent=2)

    # ---- 收尾核验 ----
    n_test_fit = sum(1 for p in (args.fit_root / "test").iterdir() if p.is_dir())
    n_test_src = sum(1 for p in (args.src_root / "test").iterdir() if p.is_dir())
    print(f"\n完成：移动 {len(moved_records)} 组。"
          f"fit_results/test 现有 {n_test_fit} 组，try_new_dataset1/test 现有 {n_test_src} 组")
    print(f"标注备份：{backup}\n恢复日志：{log_path}")


if __name__ == "__main__":
    main()
