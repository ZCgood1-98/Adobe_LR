#!/usr/bin/env python3
"""evaluate_final.py — 用最终方案（retouch-final）重跑验证，作为存档（不用于选择方案）
================================================================================

与 retouch-v2 同一批验证数据与计算方式：V1 全部专家版本（含按组 bootstrap）、V3 极端风格（含场景 bootstrap）、
V4 MSC 双向、V5 合成饱和度；并列出 retouch-v2 五来源 FB 的数字作对照，按 v2 判据标注是否达标（仅供参考）。
输出：<bg>/retouch_final/{v1all_batches.csv, fb_v3.csv, fb_v4.csv, fb_v5.csv, validation.json}

用法（aes 环境）：python 6_aes_eval/batch_retouch/evaluate_final.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402
import evaluate_v2 as e2  # noqa: E402
from evaluate_v1 import rho  # noqa: E402
from final_scheme import final_scorer  # noqa: E402

BG = bq.bg_dir()
OUT = BG / "retouch_final"
PROTO = Path(__file__).resolve().parent / "retouch_final_protocol.json"


def main():
    OUT.mkdir(exist_ok=True)
    tgt = OUT / "protocol.json"
    if not tgt.exists():
        tgt.write_bytes(PROTO.read_bytes())
    h = hashlib.sha256(tgt.read_bytes()).hexdigest()
    assert h == hashlib.sha256(PROTO.read_bytes()).hexdigest(), "结果目录里的协议与代码目录不一致"
    sc = final_scorer()
    v1, v1b = e2.eval_v1({"final": sc}, out=OUT)
    v3, v4, v5 = e2.fb_v3(sc, out=OUT), e2.fb_v4(sc, out=OUT), e2.fb_v5(sc, out=OUT)
    s = {}
    for sub in ["vivid", "muted", "bw"]:
        d = v1[v1.subgroup == sub]
        bb = v1b[v1b.subgroup == sub].rho.dropna() if sub != "bw" else pd.Series(dtype=float)
        s[f"V1_{sub}"] = {"batches": int(len(d)), "same_mean_dQ": float(d.same.mean()), "same_frac_pos": float((d.same > 0).mean()),
                          "mixed_spearman_mean": float(np.nanmean([rho(dd.mixed, dd.label) for _, dd in d.groupby("seed")])),
                          "mixed_ci": [float(bb.quantile(.025)), float(bb.quantile(.975))] if len(bb) else None}
    s.update({"V3": v3, "V4": v4, "V5": v5})
    v2 = json.loads((BG / "retouch_v2" / "summary.json").read_text())
    ref = v2["CLIP-R1S-contrast"]["V4"]["mixed_spearman_mean"]
    k1 = all(s[f"V1_{g}"]["same_mean_dQ"] > 0 and s[f"V1_{g}"]["same_frac_pos"] >= 0.8 and
             s[f"V1_{g}"]["mixed_spearman_mean"] > 0 and s[f"V1_{g}"]["mixed_ci"][0] > 0 for g in ["vivid", "muted"])
    k2 = all(v["spearman"] > 0 for v in v3.values())
    k3 = v4["mixed_spearman_mean"] >= 0.5 and v4["mixed_spearman_mean"] >= ref - 0.05
    res = {"protocol_sha256": h, "record_only": True, "final": s,
           "v2_criteria_for_reference": {"K1": bool(k1), "K2": bool(k2), "K3": bool(k3), "K3_threshold": ref - 0.05},
           "v2_five_source_FB_for_comparison": {k: v2["FB-R1S-contrast"][k] for k in ["V1_vivid", "V1_muted", "V1_bw", "V3", "V4", "V5"]}}
    (OUT / "validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, allow_nan=True))
    print(json.dumps(res["v2_criteria_for_reference"], ensure_ascii=False))
    for sub in ["vivid", "muted", "bw"]:
        v = s[f"V1_{sub}"]
        print(f"V1 {sub}: 同向 {v['same_mean_dQ']:+.3f}（正 {v['same_frac_pos']:.2f}）随机方向 {v['mixed_spearman_mean']:+.3f} {v['mixed_ci']}")
    print("V3:", {a: round(v["spearman"], 2) for a, v in v3.items()}, " V4:", round(v4["mixed_spearman_mean"], 3),
          " V5 峰值:", v5["peak_factor"])


if __name__ == "__main__":
    main()
