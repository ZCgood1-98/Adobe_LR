#!/usr/bin/env python3
"""build_sets_v1.py — retouch-v1 的验证集：V1 专家修图对原图（重新提特征）、V2 REPID 偏好对、V3 极端风格
================================================================================

按 retouch_v1_protocol.json 构建，结果写入 <bg>/retouch_v1/：
  v1_pairs.parquet + v1_coords.npy   训练分区每组 1 对（条件输入图 vs 真实专家终图），坐标为当前编码路径重新提取
  v2_pairs.parquet                   REPID train 明确对，每个场景每类 1 对（指向 coords.npy 的行号）
  v3_levels.parquet + v3_coords.npy  极端风格 4 轴 × 19 场景 × 9 档（缓存特征经同一 PCA 投影）

用法（aes 环境）：python 6_aes_eval/batch_retouch/build_sets_v1.py
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
OUT = BG / "retouch_v1"
PROTO = Path(__file__).resolve().parent / "retouch_v1_protocol.json"
SEED = 20261119


def check_protocol():
    a = hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    b = hashlib.sha256(PROTO.read_bytes()).hexdigest()
    assert a == b, "结果目录里的协议与代码目录不一致"
    return a


def colorfulness(rgb_u8: np.ndarray) -> float:
    x = rgb_u8.astype(np.float64)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def build_v1(cc: bq.ClipCoords):
    rows = [json.loads(line) for line in (REPO / "sft_data_v2/set_sft/conditions.jsonl").read_text().splitlines()]
    cand = []
    for c in rows:
        if c["split"] != "train":
            continue
        uuid = c["source_group"].split("/")[1]
        inp = REPO / c["input_path"]
        if not inp.exists():
            continue
        for v in c["versions"]:
            ex = next((p for sp in ("train", "test")
                       if (p := REPO / "scraped_data_v4" / sp / uuid / f"version_{v['version']}" / "main_image.png").exists()), None)
            if ex is not None:
                cand.append({"family": c["source_group"], "condition_id": c["condition_id"], "version": str(v["version"]),
                             "bw": bool(c["bw"]), "category": c["category"], "input_path": str(inp), "expert_path": str(ex)})
    cand = pd.DataFrame(cand).sort_values(["family", "condition_id", "version"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    # 按组名排序依次抽（显式循环，保留分组列；随机数调用顺序与最初的 groupby.apply 版本相同）
    pick = pd.DataFrame([g.iloc[rng.integers(len(g))] for _, g in cand.groupby("family", sort=True)]).reset_index(drop=True)
    print(f"V1：训练分区可用 (条件, 版本) {len(cand)} 个，覆盖 {cand.family.nunique()} 组；每组抽 1 对 → {len(pick)} 对", flush=True)
    old = OUT / "v1_pairs.parquet"
    if old.exists() and (OUT / "v1_coords.npy").exists():
        prev = pd.read_parquet(old)
        if (len(prev) == len(pick) and (prev.input_path.values == pick.input_path.values).all()
                and (prev.expert_path.values == pick.expert_path.values).all()):
            keep = [c for c in prev.columns if c not in pick.columns]
            pick = pd.concat([pick, prev[keep]], axis=1)
            pick.to_parquet(old, index=False)
            print("  已有特征与本次抽样逐条一致，直接复用（补上 family 列）", flush=True)
            return pick

    paths = sorted(set(pick.input_path) | set(pick.expert_path))
    from PIL import Image, ImageOps
    import torch

    def load(p):
        with Image.open(p) as im:
            im = cc.resize_long(ImageOps.exif_transpose(im).convert("RGB"))
            return cc.enc.preprocess(im), colorfulness(np.asarray(im))

    feats, cfs = [], []
    with ThreadPoolExecutor(8) as ex:
        for i in range(0, len(paths), 64):
            got = list(ex.map(load, paths[i:i + 64]))
            feats.append(cc.enc.encode(torch.stack([g[0] for g in got])).cpu().numpy())
            cfs += [g[1] for g in got]
            if (i // 64) % 20 == 0:
                print(f"  V1 提取 {min(i + 64, len(paths))}/{len(paths)}", flush=True)
    coords = cc.project(np.concatenate(feats))
    pos = {p: i for i, p in enumerate(paths)}
    pick["idx_in"] = pick.input_path.map(pos)
    pick["idx_ex"] = pick.expert_path.map(pos)
    pick["cf_in"] = pick.idx_in.map(lambda i: cfs[i])
    pick["cf_ex"] = pick.idx_ex.map(lambda i: cfs[i])
    pick["d_cf"] = pick.cf_ex - pick.cf_in
    pick["subgroup"] = np.where(pick.bw, "bw", np.where(pick.d_cf > 0, "vivid", "muted"))
    np.save(OUT / "v1_coords.npy", coords)
    pick.to_parquet(OUT / "v1_pairs.parquet", index=False)
    print("V1 子组：", pick.subgroup.value_counts().to_dict(), flush=True)
    return pick


def build_v2():
    p = pd.read_parquet(REPO / "5_training/analysis/results/dear/pairs.parquet")
    p = p[(p.split == "train") & ((p.mos - 0.5).abs() > 0.15)].copy()
    u = pd.read_parquet(BG / "audit_universe.parquet", columns=["key", "feature_idx"])
    idx = dict(zip(u.key, u.feature_idx))
    p["idx_left"] = ("REPID:" + p.name + "|" + p.left).map(idx)
    p["idx_right"] = ("REPID:" + p.name + "|" + p.right).map(idx)
    p = p.dropna(subset=["idx_left", "idx_right"])
    p["family"] = p.name
    p["pair_kind"] = np.where(p.kind.eq("render_vs_render"), "expert_vs_expert", "vs_neutral")
    rng = np.random.default_rng(SEED)
    p = p.sort_values(["pair_kind", "family", "left", "right"])
    out = pd.DataFrame([g.iloc[rng.integers(len(g))] for _, g in p.groupby(["pair_kind", "family"], sort=True)]).reset_index(drop=True)
    # 胜者 / 败者在 coords.npy 中的行号
    out["idx_win"] = np.where(out.mos > 0.5, out.idx_left, out.idx_right).astype(int)
    out["idx_lose"] = np.where(out.mos > 0.5, out.idx_right, out.idx_left).astype(int)
    out.to_parquet(OUT / "v2_pairs.parquet", index=False)
    print("V2：REPID train 明确对，每场景每类 1 对：", out.pair_kind.value_counts().to_dict(), flush=True)


def build_v3(cc: bq.ClipCoords):
    z = np.load(Path("/home/chen/Work/datasets/aes_cache/feats/clip_vitl14/repid_extreme.npz"))
    lst = pd.read_parquet(REPO / "6_aes_eval/runs/lists/repid_extreme.parquet")
    feat = dict(zip(z["ids"].astype(str), z["feat"]))
    lst = lst[lst.id.isin(feat)].reset_index(drop=True)
    coords = cc.project(np.stack([feat[i] for i in lst.id]).astype(np.float32))
    np.save(OUT / "v3_coords.npy", coords)
    lst.to_parquet(OUT / "v3_levels.parquet", index=False)
    print("V3：极端风格", lst.groupby("axis").size().to_dict(), flush=True)


def main():
    h = check_protocol()
    cc = bq.ClipCoords(REPO)
    build_v1(cc)
    build_v2()
    build_v3(cc)
    (OUT / "build_sets.json").write_text(json.dumps({"protocol_sha256": h}, indent=1))


if __name__ == "__main__":
    main()
