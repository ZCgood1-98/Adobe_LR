#!/usr/bin/env python3
"""score_full.py — 用选定的批量双库方案给全数据集打分（逐臂、逐类别），按臂多进程并行
================================================================================

口径（与 retouch 协议一致的批量设计）：
  · 每个臂与同格式基准配对：模型输出（q95 jpg）对「原图_jpg95」；专家终图（png）对「原图」png；
    「原图_jpg95」对「原图」作为压缩本身的对照
  · 模型臂统一用「所有模型臂都有输出」的公共组，保证各模型在完全相同的批次上比较
  · 专家：每组随机取 1 个版本（RNG 固定），在全部组上与原图比较
  · 总体：每批 64 组、5 种划分，ΔQ = Q(输出批) − Q(基准批)，报全部完整批的均值与 ΔQ>0 的批次比例
  · 分类别：类内满 64 组按同样方式分批；不足 64 组的类整类作为一批，标注 small=True
  · 95% 区间：按组重抽 200 次（类内重抽），按 seed 0 规则重新分批
  · 附带 Δ色彩丰富度（7_baselines/runs/ABC_aes/groups.parquet 的逐组色彩丰富度；专家为组内各版本均值）

用法（aes 环境）：
  python 6_aes_eval/batch_retouch/score_full.py --method fb_final        # 最终方案（需先运行 extract_full.py --fb）
  python 6_aes_eval/batch_retouch/score_full.py --method fb              # retouch-v2 的五来源候选
  python 6_aes_eval/batch_retouch/score_full.py --method clip --bank R1S # 需先运行 extract_full.py
产物：<bg>/retouch_full/table_<方案>.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
V1, OUTD = BG / "retouch_v1", BG / "retouch_full"
SEED, N, SEEDS, BOOT = 20261119, 64, [0, 1, 2, 3, 4], 200
CATS = ["人物", "风景", "都市", "植物", "动物", "食物", "创意抽象", "静物产品", "交通工具", "其他"]
G = {}          # fork 后由子进程共享：打分函数、数据


def partitions(n, seed):
    ix = np.random.default_rng(SEED + seed).permutation(n)
    return [ix[lo:lo + N] for lo in range(0, n - N + 1, N)]


def batch_dq(kb, ka, seeds=SEEDS):
    score = G["score"]
    n = len(kb)
    if n < N:
        return [score(ka) - score(kb)], True
    return [score(ka[ix]) - score(kb[ix]) for s in seeds for ix in partitions(n, s)], False


def boot_ci(kb, ka):
    rng = np.random.default_rng(SEED + 2000)
    vals = []
    for _ in range(BOOT):
        s = rng.integers(0, len(kb), len(kb))
        d, _ = batch_dq(kb[s], ka[s], seeds=[0])
        vals.append(np.mean(d))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def run_job(job):
    cls, arm, fams, base_col, arm_col, cf_label, cf_base = job
    piv, ex, cat, cf = G["piv"], G["ex"], G["cat"], G["cf"]

    def keys(fs, col):
        return (ex.loc[fs].to_numpy() if col == "专家" else piv.loc[fs, col].to_numpy()).astype(str)

    rows = []
    for c in ["全部"] + CATS:
        fs = fams if c == "全部" else np.array([f for f in fams if cat[f] == c])
        if len(fs) == 0:
            continue
        kb, ka = keys(fs, base_col), keys(fs, arm_col)
        d, small = batch_dq(kb, ka)
        lo, hi = boot_ci(kb, ka)
        dcf = np.array([cf.get((f, cf_label), np.nan) - cf.get((f, cf_base), np.nan) for f in fs])
        rows.append({"cls": cls, "arm": arm, "category": c, "groups": len(fs), "batches": len(d), "small": small,
                     "dQ": float(np.mean(d)), "ci_low": lo, "ci_high": hi, "frac_pos": float(np.mean(np.array(d) > 0)),
                     "d_colorful": float(np.nanmean(dcf)) if np.isfinite(dcf).any() else np.nan})
    print(f"  {cls} {arm} 完成", flush=True)
    return rows


def make_scorer(method, bank):
    if method == "clip":
        z = np.load(OUTD / "full_coords.npz")
        lookup = dict(zip(z["paths"].astype(str), z["coords"]))
        Z = bq.load_coords(REPO)
        sc = (bq.round1_scorer(Z, REPO, {"R1S": "source_balanced", "R1C": "source_and_color_balanced"}[bank])
              if bank in ("R1S", "R1C") else bq.v2_global_scorer(Z, REPO))
        return lambda keys: sc.score(np.stack([lookup[k] for k in keys]))["Q_contrast"]
    if method == "fb_final":                # 最终方案：四来源参考库（不含 AVA），PCA 也只用这四个来源拟合
        from final_scheme import final_scorer
        ks = final_scorer(extra=OUTD / "full_fb_raw.npz")
    else:
        from evaluate_v2 import fb_scorer       # retouch-v2 候选：五来源
        ks = fb_scorer(extra=OUTD / "full_fb_raw.npz")
    return lambda keys: ks.score(list(keys))["contrast"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["clip", "fb", "fb_final"], required=True)
    ap.add_argument("--bank", default="R1S", choices=["R1S", "R1C", "V2G"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    tag = {"clip": f"clip_{args.bank}", "fb": "fb_R1S", "fb_final": "fb_final"}[args.method]
    m = pd.read_parquet(V1 / "full_manifest.parquet")
    m = m[m.exists]
    G["score"] = make_scorer(args.method, args.bank)

    g = pd.read_parquet(REPO / "7_baselines/runs/ABC_aes/groups.parquet")
    G["cf"] = {(r.group, r.arm): r.colorfulness for r in g.itertuples()}
    piv = m[m.cls != "专家"].pivot_table(index="family", columns="arm", values="path", aggfunc="first")
    G["piv"] = piv
    G["cat"] = m.drop_duplicates("family").set_index("family").category
    model_arms = m[~m.cls.isin(["对照", "专家"])][["cls", "arm"]].drop_duplicates().itertuples(index=False)
    model_arms = [(c, a) for c, a in model_arms]
    common = piv.dropna(subset=[a for _, a in model_arms] + ["原图_jpg95"]).index.to_numpy()
    rng = np.random.default_rng(SEED)
    exp_rows = m[m.cls == "专家"].sort_values(["family", "version"])
    G["ex"] = pd.Series({f: d.path.iloc[rng.integers(len(d))] for f, d in exp_rows.groupby("family", sort=True)})
    print(f"公共组 {len(common)}（共 {len(piv)}）；专家覆盖 {len(G['ex'])} 组", flush=True)

    label = {"A": lambda a: a, "B": lambda a: f"B·{a}", "C": lambda a: f"C·{a}", "D": lambda a: f"D·{a}"}
    jobs = [("对照", "原图_jpg95", common, "原图", "原图_jpg95", "原图_jpg95", "原图")]
    jobs += [(c, a, common, "原图_jpg95", a, label[c](a), "原图_jpg95") for c, a in model_arms]
    jobs += [("专家", "专家", G["ex"].index.to_numpy(), "原图", "专家", "专家", "原图")]
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for part in pool.map(run_job, jobs) for r in part]
    df = pd.DataFrame(rows)
    OUTD.mkdir(exist_ok=True)
    df.to_csv(OUTD / f"table_{tag}.csv", index=False)
    print(f"已写出 {OUTD / f'table_{tag}.csv'}", flush=True)


if __name__ == "__main__":
    main()
