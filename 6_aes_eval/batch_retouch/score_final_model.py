#!/usr/bin/env python3
"""score_final_model.py — 用最终评测方案（retouch-final）重评我们的最终模型在封存测试集上的输出
================================================================================

输入：7_baselines/runs/FINAL_TEST/（2026-11-19 最终流水线的测试输出：1,046 个条件 × 6 个候选 + 各条件原图 q95）。
      只重新打分，不重新运行模型；原 summary.json（旧按图 Q_ref_k1）不改动。
口径：
  · 照片等权：每种划分里每张照片随机取 1 个条件（RNG 固定）；每批 64 张不同照片；5 种划分
  · 最终模型：6 个候选（风格库 1/2/3/7/8/9，对应 k0–k5）各按一批算 ΔQ = Q(候选批) − Q(同一批条件原图 q95)，再对 6 个候选取平均；另报每个风格
  · 参照：同一批测试照片、同样划分上的 A/B/C/D 各基线（对组级原图 q95）与专家（每张照片取 1 个版本，对原图 png）
  · 主表只用所有基线都有输出的照片（完全配对）；最终模型在全部 500 张上的结果另报
  · 分类别：类内照片 ≥64 张按批，否则整类一批（标注）
  · 95% 区间：按照片重抽 200 次
产物：<bg>/retouch_final/final_model_test/{summary.json, results.csv, tables.md}

用法（aes 环境）：python 6_aes_eval/batch_retouch/score_final_model.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
FT = REPO / "7_baselines" / "runs" / "FINAL_TEST"
OUT = BG / "retouch_final" / "final_model_test"
SEED, N, SEEDS, BOOT = 20261119, 64, [0, 1, 2, 3, 4], 200
STYLES = [1, 2, 3, 7, 8, 9]
CAT_ZH = {"portrait": "人物", "landscape": "风景", "urban": "都市", "plant": "植物", "animal": "动物", "food": "食物",
          "creative": "创意抽象", "still_life": "静物产品", "vehicle": "交通工具", "other": "其他"}
CATS = ["人物", "风景", "都市", "植物", "动物", "食物", "创意抽象", "静物产品", "交通工具", "其他"]
G = {}


def extract(paths, out_npz):
    """AesCLIP 投影前 768 维、L2 归一化特征（与 extract_full.py --fb 相同的预处理）。"""
    if out_npz.exists():
        return
    import torch
    from PIL import Image, ImageOps
    sys.path.insert(0, str(REPO / "6_aes_eval"))
    from common import resize_long
    from encoders import build
    enc = build("aesclip_pre")

    def load(p):
        with Image.open(p) as im:
            return enc.preprocess(resize_long(ImageOps.exif_transpose(im).convert("RGB")))

    feats, t0 = [], time.time()
    with ThreadPoolExecutor(10) as ex:
        for i in range(0, len(paths), 64):
            xs = list(ex.map(load, paths[i:i + 64]))
            f = enc.encode(torch.stack(xs)).cpu().numpy()
            feats.append(f / np.linalg.norm(f, axis=1, keepdims=True))
    np.savez(out_npz, paths=np.array(paths), coords=np.concatenate(feats).astype(np.float16))
    print(f"  提取 {len(paths)} 张，{time.time() - t0:.0f}s", flush=True)


def batches_for(photos, seed):
    """一种划分：照片打乱后每 64 张一批；不足 64 张时整组一批。"""
    if len(photos) < N:
        return [np.arange(len(photos))]
    ix = np.random.default_rng(SEED + seed).permutation(len(photos))
    return [ix[lo:lo + N] for lo in range(0, len(photos) - N + 1, N)]


def pick_cond(photos, seed):
    rng = np.random.default_rng(SEED + 100 + seed)
    conds = G["conds"]
    return np.array([conds[p][rng.integers(len(conds[p]))] for p in photos])


def arm_keys(arm, photos, cids):
    """返回（基准键, 输出键）列表；最终模型的 arm 为 'k0'…'k5'。"""
    if arm.startswith("k"):
        k = int(arm[1:])
        return [G["orig"][c] for c in cids], [G["cand"][(c, k)] for c in cids]
    if arm == "专家":
        return [G["piv"].loc[p, "原图"] for p in photos], [G["ex"][p] for p in photos]
    return [G["piv"].loc[p, "原图_jpg95"] for p in photos], [G["piv"].loc[p, arm] for p in photos]


def run_set(arm, photos, boot=True):
    """在一组照片上：5 种划分的批均值 ΔQ、ΔQ>0 的批比例、按照片重抽的 95% 区间。"""
    sc = G["score"]
    vals = []
    for s in SEEDS:
        cids = pick_cond(photos, s)
        for ix in batches_for(photos, s):
            kb, ka = arm_keys(arm, photos[ix], cids[ix])
            vals.append(sc(ka) - sc(kb))
    lo = hi = np.nan
    if boot:
        rng = np.random.default_rng(SEED + 2000)
        bs = []
        for _ in range(BOOT):
            ph = photos[rng.integers(0, len(photos), len(photos))]
            cids = pick_cond(ph, 0)
            d = []
            for ix in batches_for(ph, 0):
                kb, ka = arm_keys(arm, ph[ix], cids[ix])
                d.append(sc(ka) - sc(kb))
            bs.append(np.mean(d))
        lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    return {"dQ": float(np.mean(vals)), "batches": len(vals), "frac_pos": float(np.mean(np.array(vals) > 0)),
            "ci_low": lo, "ci_high": hi, "small": len(photos) < N, "photos": len(photos)}


def run_model_mean(photos, boot=True):
    """最终模型 6 个候选的平均：同一批照片、同一条件，6 个候选的 ΔQ 先在批内平均。"""
    sc = G["score"]

    def batch_mean(ph, cids):
        b = [G["orig"][c] for c in cids]
        qb = sc(b)
        return np.mean([sc([G["cand"][(c, k)] for c in cids]) - qb for k in range(6)])

    vals = []
    for s in SEEDS:
        cids = pick_cond(photos, s)
        for ix in batches_for(photos, s):
            vals.append(batch_mean(photos[ix], cids[ix]))
    lo = hi = np.nan
    if boot:
        rng = np.random.default_rng(SEED + 2000)
        bs = []
        for _ in range(BOOT):
            ph = photos[rng.integers(0, len(photos), len(photos))]
            cids = pick_cond(ph, 0)
            bs.append(np.mean([batch_mean(ph[ix], cids[ix]) for ix in batches_for(ph, 0)]))
        lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    return {"dQ": float(np.mean(vals)), "batches": len(vals), "frac_pos": float(np.mean(np.array(vals) > 0)),
            "ci_low": lo, "ci_high": hi, "small": len(photos) < N, "photos": len(photos)}


def job(task):
    name, scope, photos = task
    r = run_model_mean(photos) if name == "最终模型（6 候选平均）" else run_set(name, photos)
    print(f"  {scope} {name} 完成", flush=True)
    return {"arm": name, "scope": scope, **r}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lst = pd.DataFrame(json.loads((FT / "list.json").read_text()))
    extract(sorted(lst.path.unique()), OUT / "final_test_fb_raw.npz")

    from final_scheme import final_scorer
    ks = final_scorer(extra=[BG / "retouch_full" / "full_fb_raw.npz", OUT / "final_test_fb_raw.npz"])
    G["score"] = lambda keys: ks.score(list(keys))["contrast"]

    o = lst[lst.arm == "原图_jpg95"]
    G["orig"] = dict(zip(o.cid, o.path))
    c = lst[lst.arm == "最终模型"]
    G["cand"] = {(r.cid, int(r.k)): r.path for r in c.itertuples()}
    G["conds"] = {g: sorted(d.cid) for g, d in o.groupby("group")}
    cat_of = {r.group: CAT_ZH[r.category] for r in o.itertuples()}

    m = pd.read_parquet(BG / "retouch_v1" / "full_manifest.parquet")
    m = m[m.exists & (m.split == "test")]
    G["piv"] = m[m.cls != "专家"].pivot_table(index="family", columns="arm", values="path", aggfunc="first")
    rng = np.random.default_rng(SEED)
    exp_rows = m[m.cls == "专家"].sort_values(["family", "version"])
    G["ex"] = {f: d.path.iloc[rng.integers(len(d))] for f, d in exp_rows.groupby("family", sort=True)}
    ref_arms = [a for a in m[~m.cls.isin(["对照", "专家"])].arm.unique()]
    common = np.array(sorted(G["piv"].dropna(subset=ref_arms + ["原图_jpg95", "原图"]).index.intersection(list(G["conds"]))))
    common = np.array([p for p in common if p in G["ex"]])
    allp = np.array(sorted(G["conds"]))
    print(f"测试照片：最终模型 {len(allp)} 张；所有基线都有输出（主表）{len(common)} 张", flush=True)

    arms = ["最终模型（6 候选平均）"] + [f"k{k}" for k in range(6)] + ref_arms + ["专家"]
    tasks = [(a, "全部", common) for a in arms] + [("最终模型（6 候选平均）", "全部500张", allp)]
    for cat in CATS:
        ph = np.array([p for p in common if cat_of[p] == cat])
        if len(ph):
            tasks += [(a, cat, ph) for a in ["最终模型（6 候选平均）"] + ref_arms + ["专家"]]
    with mp.get_context("fork").Pool(16) as pool:
        rows = pool.map(job, tasks)
    df = pd.DataFrame(rows)
    cls_of = dict(zip(m.arm, m.cls))
    df["cls"] = df.arm.map(lambda a: "我们" if a.startswith("最终") else ("我们·风格" if a.startswith("k") else cls_of.get(a, "专家")))
    df.to_csv(OUT / "results.csv", index=False)

    # 旧按图 Q_ref_k1（2026-11-19）作对照
    old = json.loads((FT / "summary.json").read_text())
    (OUT / "summary.json").write_text(json.dumps({
        "scheme": "retouch-final（批量双库 · FB 表征 · 四来源参考库）",
        "photos_main": int(len(common)), "photos_all": int(len(allp)),
        "old_per_image_Q_ref_k1_final_model": old["arms"]["最终模型"]["ΔQ_ref_k1"],
        "rows": rows}, ensure_ascii=False, indent=1, allow_nan=True))
    print(f"已写出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
