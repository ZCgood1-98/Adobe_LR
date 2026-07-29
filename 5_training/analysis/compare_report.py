"""并排比两个 diag 的 report.json:每 stage improve_p50 + 每参数 MAE(带 delta)。
用法: python compare_report.py <基线diag目录> <新diag目录>"""
import json, sys
from pathlib import Path

base = json.load(open(Path(sys.argv[1]) / "report.json"))
new = json.load(open(Path(sys.argv[2]) / "report.json"))
STAGES = ["light", "tone", "color", "detail"]
# 升 T 的靶子 / 被保护的,标注出来
BOOST = {"contrast", "texture", "clarity", "exposure", "tint", "whites", "temperature"}
PROTECT = {"saturation", "vibrance"}

print(f"{'stage/参数':22} {'基线':>9} {'新':>9} {'Δ':>9}   判读")
print("-" * 66)
for st in STAGES:
    b, n = base[st], new[st]
    d = n["improve_p50"] - b["improve_p50"]
    flag = "↑好" if d > 0.005 else ("↓退" if d < -0.005 else "≈")
    print(f"{'▶ '+st+' improve_p50':22} {b['improve_p50']:>+9.3f} {n['improve_p50']:>+9.3f} {d:>+9.3f}   {flag}")
    keys = sorted(set(b.get("param_mae", {})) | set(n.get("param_mae", {})))
    for k in keys:
        bm, nm = b["param_mae"].get(k), n["param_mae"].get(k)
        if bm is None or nm is None:
            continue
        dd = nm - bm
        tag = "⭐升T" if k in BOOST else ("✋护" if k in PROTECT else "")
        # MAE 越小越好:降=好
        v = "↓好" if dd < -0.01 * max(bm, 1e-9) else ("↑差" if dd > 0.01 * max(bm, 1e-9) else "≈")
        print(f"    {k:18} {bm:>9.2f} {nm:>9.2f} {dd:>+9.2f}   {tag} {v}")
print("\nbw:", base.get("bw_subset", {}).get("improve_p50"), "→", new.get("bw_subset", {}).get("improve_p50"))
