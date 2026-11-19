#!/usr/bin/env python3
"""verify_batchq.py — 确认 batchq 与已冻结结果逐数一致，以及新图片编码路径与缓存一致

  ① 首轮 batch_results.csv：两种库、全部批次的 ΔQ
  ② content_v2 human_results.csv：filtered_global 全部批次的 ΔQ
  ③ 重新编码若干张已缓存图片（REPID、本项目原图、专家终图、MSC），比较 16 维坐标与 768 维特征

用法（aes 环境）：python 6_aes_eval/batch_retouch/verify_batchq.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

BG = bq.bg_dir()
Z = bq.load_coords()
out = {}

# ① 首轮
pairs = pd.read_parquet(BG / "calibration_pairs.parquet")
batches = pd.read_parquet(BG / "batches.parquet")
ref = pd.read_csv(BG / "batch_results.csv")
for arm in ["source_balanced", "source_and_color_balanced"]:
    sc = bq.round1_scorer(Z, arm=arm)
    diffs = []
    for (n, seed, b), g in batches.groupby(["n_requested", "seed", "batch"]):
        r = pairs.iloc[g.pair_row]
        dq = sc.score(Z[r.idx_edited])["Q_contrast"] - sc.score(Z[r.idx_original])["Q_contrast"]
        want = ref[(ref.arm == arm) & (ref.n_requested == n) & (ref.seed == seed) & (ref.batch == b)].delta_Q.item()
        diffs.append(abs(dq - want))
    out[f"round1_{arm}"] = {"batches": len(diffs), "max_abs_diff": float(max(diffs))}
    print(f"① 首轮 {arm}: {len(diffs)} 批，ΔQ 最大差 {max(diffs):.2e}", flush=True)

# ② content_v2 filtered_global
H2 = BG / "content_v2"
p2 = pd.read_parquet(H2 / "pairs.parquet")
b2 = pd.read_parquet(H2 / "human_batches.parquet")
r2 = pd.read_csv(H2 / "human_results.csv")
sc2 = bq.v2_global_scorer(Z)
diffs = []
for (n, seed, b), g in b2.groupby(["n", "seed", "batch"]):
    r = p2.iloc[g.row]
    dq = sc2.score(Z[r.idx_edited])["Q_contrast"] - sc2.score(Z[r.idx_original])["Q_contrast"]
    want = r2[(r2.arm == "filtered_global") & (r2.n == n) & (r2.seed == seed) & (r2.batch == b)].delta_Q.item()
    diffs.append(abs(dq - want))
out["v2_filtered_global"] = {"batches": len(diffs), "max_abs_diff": float(max(diffs))}
print(f"② content_v2 filtered_global: {len(diffs)} 批，ΔQ 最大差 {max(diffs):.2e}", flush=True)

# ③ 编码路径
u = pd.read_parquet(BG / "audit_universe.parquet", columns=["key", "feature_idx", "source", "path"])
rng = np.random.default_rng(0)
pick = []
for src in ["REPID", "PROJECT_project_inputs", "PROJECT_train_gt_real", "MSC", "AVA"]:
    g = u[u.source.eq(src)]
    g = g[g.path.map(lambda p: Path(p).exists())]
    pick.append(g.iloc[rng.choice(len(g), 6, replace=False)])
pick = pd.concat(pick)
cc = bq.ClipCoords()
F = cc.features(pick.path.tolist(), batch=16)
coords_new = cc.project(F)
allf = np.load(BG / "all_features.npy", mmap_mode="r")
res = {}
for src, g in pick.assign(i=range(len(pick))).groupby("source"):
    idx = g.feature_idx.to_numpy()
    cos = (F[g.i] * allf[idx].astype(np.float32)).sum(1) / np.linalg.norm(allf[idx].astype(np.float32), axis=1)
    d = np.abs(coords_new[g.i] - Z[idx]).max()
    res[src] = {"min_cosine_768": float(cos.min()), "max_abs_diff_coords": float(d)}
    print(f"③ {src}: 768 维特征最小余弦 {cos.min():.5f}，16 维坐标最大差 {d:.4f}", flush=True)
out["encoding_check"] = res
print(json.dumps(out, ensure_ascii=False, indent=1))
