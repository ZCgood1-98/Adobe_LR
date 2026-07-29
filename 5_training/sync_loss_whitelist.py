#!/usr/bin/env python3
"""复审后同步 loss 白名单。

规则：loss_whitelist_candidates.txt（重拟合后 loss>8、几何已干净的 version）里，
复审后标注仍为 keep 的 → 并入 loss_whitelist.txt；已被舍弃的不进（就算进了也会先被
标注过滤挡住，白名单只对 keep 的 version 有意义）。既有白名单条目原样保留（只增不减，
需移除请直接编辑文件）。幂等可重跑。

用法（复审完成后，仓库根目录）：python 5_training/sync_loss_whitelist.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WL = REPO_ROOT / "sft_data_v2" / "loss_whitelist.txt"
CAND = REPO_ROOT / "sft_data_v2" / "loss_whitelist_candidates.txt"


def read_vids(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def main():
    ann = json.loads((REPO_ROOT / "fit_annotations.json").read_text())
    existing = read_vids(WL)
    added, dropped = [], []
    for vid in read_vids(CAND):
        key, vn = vid.rsplit("/v", 1)
        a = ann.get(key) or {}
        vinfo = (a.get("versions") or {}).get(vn)
        dec = vinfo.get("decision") if isinstance(vinfo, dict) else vinfo
        keep = bool(a.get("done")) and not a.get("discarded") and dec == "keep"
        (added if keep else dropped).append(vid)

    merged = sorted(set(existing) | set(added))
    WL.write_text("# loss>8 但人工审美背书（复审 keep）+ 渲染一致性干净的 version\n"
                  "# 由 5_training/sync_loss_whitelist.py 维护，可重跑\n"
                  + "\n".join(merged) + ("\n" if merged else ""))
    print(f"候选 {len(added) + len(dropped)} 个：标注为 keep 并入 {len(added)}，"
          f"舍弃/未标注不进 {len(dropped)}")
    print(f"白名单现共 {len(merged)} 条（此前 {len(existing)}）→ {WL}")


if __name__ == "__main__":
    main()
