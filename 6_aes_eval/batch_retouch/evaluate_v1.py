#!/usr/bin/env python3
"""evaluate_v1.py — 按 retouch_v1_protocol.json 评测 6 个候选（3 种参考库 × contrast / ideal），给出判定
================================================================================

输入：build_sets_v1.py 的产物 + 首轮/content_v2 冻结产物（全部只读）
输出（<bg>/retouch_v1/）：
  v1_expert.csv / v2_repid.csv / v3_extreme.csv / v4_msc.csv / v5_saturation.csv   逐批或逐档明细
  summary.json   各候选在各判据上的数字
  judge.json     K1/K2/K3 是否通过、最终选择

用法（aes 环境）：python 6_aes_eval/batch_retouch/evaluate_v1.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
OUT = BG / "retouch_v1"
PROTO = Path(__file__).resolve().parent / "retouch_v1_protocol.json"
SEED = 20261119
N = 64
SEEDS = [0, 1, 2, 3, 4]
MODES = ["contrast", "ideal"]
FACTORS = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16]


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def delta(sc, z_before, z_after):
    a, b = sc.score(z_before), sc.score(z_after)
    return {m: b[f"Q_{m}"] - a[f"Q_{m}"] for m in ("contrast", "ideal", "low")}


def partitions(n, seed):
    ix = np.random.default_rng(SEED + seed).permutation(n)
    return [ix[lo:lo + N] for lo in range(0, n - N + 1, N)]          # 只要完整批次


def oriented(n):
    rev = np.zeros(n, bool)
    rev[np.random.default_rng(SEED).permutation(n)[: n // 2]] = True
    return rev


# ---------------------------------------------------------------- V1 专家修图
def eval_v1(scorers):
    pairs = pd.read_parquet(OUT / "v1_pairs.parquet")
    C = np.load(OUT / "v1_coords.npy")
    rows, boot = [], []
    for sub in ["vivid", "muted", "bw"]:
        g = pairs[pairs.subgroup.eq(sub)].reset_index(drop=True)
        zin, zex = C[g.idx_in.to_numpy()], C[g.idx_ex.to_numpy()]
        rev = oriented(len(g))
        zb = np.where(rev[:, None], zex, zin)
        za = np.where(rev[:, None], zin, zex)
        lab = np.where(rev, -1.0, 1.0)
        for name, sc in scorers.items():
            for seed in SEEDS:
                for b, ix in enumerate(partitions(len(g), seed)):
                    same = delta(sc, zin[ix], zex[ix])
                    mixd = delta(sc, zb[ix], za[ix])
                    rows.append({"bank": name, "subgroup": sub, "seed": seed, "batch": b, "label": float(lab[ix].mean()),
                                 **{f"same_{m}": v for m, v in same.items()}, **{f"mixed_{m}": v for m, v in mixd.items()}})
            if sub == "bw":
                continue
            # 家族 bootstrap：重抽家族 → 按 seed 0 的排列方式重组批次 → 随机方向 Spearman
            rng = np.random.default_rng(SEED + 1000)
            for t in range(200):
                s = rng.integers(0, len(g), len(g))
                ix_all = np.random.default_rng(SEED).permutation(len(s))
                dq = {m: [] for m in MODES}
                labs = []
                for lo in range(0, len(s) - N + 1, N):
                    ii = s[ix_all[lo:lo + N]]
                    d = delta(sc, zb[ii], za[ii])
                    for m in MODES:
                        dq[m].append(d[m])
                    labs.append(lab[ii].mean())
                boot.append({"bank": name, "subgroup": sub, "rep": t, **{m: rho(dq[m], labs) for m in MODES}})
        print(f"  V1 {sub} 完成", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v1_expert.csv", index=False)
    return df, pd.DataFrame(boot)


# ---------------------------------------------------------------- V2 REPID 同图偏好（大众口味）
def eval_v2(scorers, Z):
    p = pd.read_parquet(OUT / "v2_pairs.parquet")
    u = pd.read_parquet(BG / "audit_universe.parquet", columns=["feature_idx", "colorful"]).set_index("feature_idx").colorful
    rows = []
    for kind, g in p.groupby("pair_kind"):
        g = g.reset_index(drop=True)
        rev = oriented(len(g))
        ib = np.where(rev, g.idx_win, g.idx_lose).astype(int)
        ia = np.where(rev, g.idx_lose, g.idx_win).astype(int)
        lab = np.where(rev, -1.0, 1.0)
        cf_b, cf_a = u.loc[ib].to_numpy(), u.loc[ia].to_numpy()
        for name, sc in scorers.items():
            for seed in SEEDS:
                for b, ix in enumerate(partitions(len(g), seed)):
                    d = delta(sc, Z[ib[ix]], Z[ia[ix]])
                    rows.append({"bank": name, "kind": kind, "seed": seed, "batch": b, "label": float(lab[ix].mean()),
                                 "d_colorful": float(cf_a[ix].mean() - cf_b[ix].mean()), **d})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v2_repid.csv", index=False)
    return df


# ---------------------------------------------------------------- V3 极端风格（大众投票）
def human_curve(votes, scenes=None):
    w = {}
    for r in votes.itertuples():
        k = 1 if scenes is None else scenes.get(r.name, 0)
        if k == 0:
            continue
        w.setdefault(r.left, []).extend([r.mos] * k)
        w.setdefault(r.right, []).extend([1 - r.mos] * k)
    return {lv: float(np.mean(v)) for lv, v in w.items()}


def eval_v3(scorers):
    lv = pd.read_parquet(OUT / "v3_levels.parquet")
    C = np.load(OUT / "v3_coords.npy")
    rows, summ = [], []
    for axis, g in lv.groupby("axis"):
        votes = pd.read_csv(Path("/home/chen/Work/datasets/REPID/extreme_styles") / f"extreme_style_{axis}.csv")
        base = "con+0" if axis == "contrast" else "original"
        levels = sorted(g.level.unique())
        scenes = sorted(g.scene.unique())
        pos = {(r.level, r.scene): i for i, r in zip(g.index, g.itertuples())}
        hc = human_curve(votes)

        def curve(sc, sc_list):
            """每档相对原图档的 ΔQ（三种组合一次算出）：{档: {mode: ΔQ}}"""
            qb = sc.score(C[[pos[(base, s)] for s in sc_list]])
            out = {}
            for l in levels:
                q = sc.score(C[[pos[(l, s)] for s in sc_list]])
                out[l] = {m: q[f"Q_{m}"] - qb[f"Q_{m}"] for m in ("contrast", "ideal", "low")}
            return out

        for name, sc in scorers.items():
            mc = curve(sc, scenes)
            for l in levels:
                for mode in ("contrast", "ideal", "low"):
                    rows.append({"bank": name, "mode": mode, "axis": axis, "level": l, "delta_Q": mc[l][mode], "human": hc[l]})
            rng = np.random.default_rng(SEED + 7)
            bs = {m: [] for m in MODES}
            for _ in range(1000):
                pick = rng.choice(scenes, len(scenes), replace=True)
                cnt = pd.Series(pick).value_counts().to_dict()
                m2 = curve(sc, list(pick))
                h2 = human_curve(votes, cnt)
                for m in MODES:
                    bs[m].append(rho([m2[l][m] for l in levels], [h2[l] for l in levels]))
            for m in MODES:
                b = np.array([x for x in bs[m] if not np.isnan(x)])
                summ.append({"bank": name, "mode": m, "axis": axis,
                             "spearman": rho([mc[l][m] for l in levels], [hc[l] for l in levels]),
                             "ci_low": float(np.quantile(b, .025)), "ci_high": float(np.quantile(b, .975)),
                             "metric_best_level": max(levels, key=lambda l: mc[l][m]), "human_best_level": max(hc, key=hc.get)})
        print(f"  V3 {axis} 完成", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "v3_extreme.csv", index=False)
    return pd.DataFrame(summ)


# ---------------------------------------------------------------- V4 MSC 退化
def eval_v4(scorers, Z):
    pairs = pd.read_parquet(BG / "calibration_pairs.parquet")
    bi = pd.read_parquet(BG / "bidirectional_pairs.parquet")
    assert (bi.family_id.values == pairs.family_id.values).all()
    batches = pd.read_parquet(BG / "batches.parquet")
    rows = []
    for name, sc in scorers.items():
        for (n, seed, b), g in batches.groupby(["n_requested", "seed", "batch"]):
            if n != N or g.partial.any():
                continue
            r, rb = pairs.iloc[g.pair_row], bi.iloc[g.pair_row]
            same = delta(sc, Z[r.idx_original], Z[r.idx_edited])
            mixd = delta(sc, Z[rb.before_idx], Z[rb.after_idx])
            rows.append({"bank": name, "seed": int(seed), "batch": int(b), "delta_mos": float(r.delta_mos.mean()),
                         "oriented_delta_mos": float(rb.oriented_delta_mos.mean()),
                         **{f"same_{m}": v for m, v in same.items()}, **{f"mixed_{m}": v for m, v in mixd.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v4_msc.csv", index=False)
    return df


# ---------------------------------------------------------------- V5 合成饱和度
def eval_v5(scorers):
    arr = {}
    for f in FACTORS:
        z = np.load(BG / "oversaturation" / ("factor_" + str(f).replace(".", "p") + ".npz"))
        arr[f] = (z["family_id"].astype(str), z["coords"].astype(np.float64))
    fam = arr[1][0]
    for f in FACTORS:
        assert (arr[f][0] == fam).all()
    perm = np.random.default_rng(SEED).permutation(len(fam))
    rows = []
    for name, sc in scorers.items():
        for b in range(len(fam) // N):
            ix = perm[b * N:(b + 1) * N]
            base = sc.score(arr[1][1][ix])
            for f in FACTORS:
                s = sc.score(arr[f][1][ix])
                rows.append({"bank": name, "batch": b, "factor": f, **{m: s[f"Q_{m}"] - base[f"Q_{m}"] for m in ("contrast", "ideal", "low")}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v5_saturation.csv", index=False)
    return df


def main():
    h = hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    assert h == hashlib.sha256(PROTO.read_bytes()).hexdigest()
    Z = bq.load_coords(REPO)
    scorers = {"R1S": bq.round1_scorer(Z, REPO, "source_balanced"),
               "R1C": bq.round1_scorer(Z, REPO, "source_and_color_balanced"),
               "V2G": bq.v2_global_scorer(Z, REPO)}
    print("开始评测", flush=True)
    v1, v1b = eval_v1(scorers)
    v2 = eval_v2(scorers, Z)
    print("  V2 完成", flush=True)
    v3 = eval_v3(scorers)
    v4 = eval_v4(scorers, Z)
    print("  V4 完成", flush=True)
    v5 = eval_v5(scorers)
    print("  V5 完成", flush=True)

    summary, judge = {}, {}
    for bank in scorers:
        for mode in MODES:
            cand = f"{bank}-{mode}"
            s = {}
            # V1
            for sub in ["vivid", "muted", "bw"]:
                d = v1[(v1.bank == bank) & (v1.subgroup == sub)]
                mixed = [rho(dd[f"mixed_{mode}"], dd.label) for _, dd in d.groupby("seed")]
                bb = v1b[(v1b.bank == bank) & (v1b.subgroup == sub)][mode].dropna() if sub != "bw" else pd.Series(dtype=float)
                s[f"V1_{sub}"] = {"batches": int(len(d)), "same_mean_dQ": float(d[f"same_{mode}"].mean()),
                                  "same_frac_pos": float((d[f"same_{mode}"] > 0).mean()),
                                  "mixed_spearman_mean": float(np.nanmean(mixed)),
                                  "mixed_ci": [float(bb.quantile(.025)), float(bb.quantile(.975))] if len(bb) else None,
                                  "decomp_same_ideal": float(d.same_ideal.mean()), "decomp_same_low": float(d.same_low.mean())}
            # V2
            for kind in ["expert_vs_expert", "vs_neutral"]:
                d = v2[(v2.bank == bank) & (v2.kind == kind)]
                s[f"V2_{kind}"] = {"spearman_mean": float(np.nanmean([rho(dd[mode], dd.label) for _, dd in d.groupby("seed")])),
                                   "colorful_baseline": float(np.nanmean([rho(dd.d_colorful, dd.label) for _, dd in d.groupby("seed")]))}
            # V3
            d3 = v3[(v3.bank == bank) & (v3["mode"] == mode)]
            s["V3"] = {r.axis: {"spearman": r.spearman, "ci": [r.ci_low, r.ci_high], "metric_best": r.metric_best_level,
                                "human_best": r.human_best_level} for r in d3.itertuples()}
            # V4
            d4 = v4[v4.bank == bank]
            s["V4"] = {"same_spearman_mean": float(np.nanmean([rho(dd[f"same_{mode}"], dd.delta_mos) for _, dd in d4.groupby("seed")])),
                       "mixed_spearman_mean": float(np.nanmean([rho(dd[f"mixed_{mode}"], dd.oriented_delta_mos) for _, dd in d4.groupby("seed")]))}
            # V5
            d5 = v5[v5.bank == bank].groupby("factor")[mode].mean()
            s["V5"] = {"mean_dQ": {str(k): float(v) for k, v in d5.items()}, "peak_factor": float(d5.idxmax()),
                       "dQ_8x_below_original": bool(d5[8] < 0), "dQ_16x_below_original": bool(d5[16] < 0)}
            summary[cand] = s

    for cand, s in summary.items():
        bank, mode = cand.split("-")
        k1 = all(s[f"V1_{g}"]["same_mean_dQ"] > 0 and s[f"V1_{g}"]["same_frac_pos"] >= 0.8 and
                 s[f"V1_{g}"]["mixed_spearman_mean"] > 0 and s[f"V1_{g}"]["mixed_ci"][0] > 0 for g in ["vivid", "muted"])
        k2 = all(v["spearman"] > 0 for v in s["V3"].values())
        ref = summary[f"{bank}-contrast"]["V4"]["mixed_spearman_mean"]
        k3 = s["V4"]["mixed_spearman_mean"] >= 0.5 and s["V4"]["mixed_spearman_mean"] >= ref - 0.05
        judge[cand] = {"K1": bool(k1), "K2": bool(k2), "K3": bool(k3), "pass": bool(k1 and k2 and k3)}
    passing = [c for c, j in judge.items() if j["pass"]]
    choice = None
    if passing:
        best = max(summary[c]["V1_muted"]["mixed_spearman_mean"] for c in passing)
        near = [c for c in passing if best - summary[c]["V1_muted"]["mixed_spearman_mean"] < 0.02]
        order = {"R1S": 0, "R1C": 1, "V2G": 2}
        choice = sorted(near, key=lambda c: (order[c.split("-")[0]], 0 if c.endswith("contrast") else 1))[0]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=True))
    (OUT / "judge.json").write_text(json.dumps({"protocol_sha256": h, "candidates": judge, "passing": passing, "selected": choice},
                                               ensure_ascii=False, indent=1))
    print(json.dumps({"candidates": judge, "passing": passing, "selected": choice}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
