#!/usr/bin/env python3
"""evaluate_v2.py — 按 retouch_v2_protocol.json 评测：3 个 CLIP 对照 + FB-R1S-contrast
================================================================================

  · V1（全部专家版本、同一批内不重复同一组、按组 bootstrap）：4 个候选都重新计算
  · V2–V5：CLIP 候选的定义与数据与 v1 完全相同，直接沿用 retouch_v1/summary.json；
    FB 候选计算 V3 / V4 / V5，不报告 V2（REPID train 被 FB 训练使用过）
输出（<bg>/retouch_v2/）：v1all_batches.csv、fb_v3.csv、fb_v4.csv、fb_v5.csv、summary.json、judge.json

用法（aes 环境）：python 6_aes_eval/batch_retouch/evaluate_v2.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402
from evaluate_v1 import human_curve, rho  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
V1 = BG / "retouch_v1"
OUT = BG / "retouch_v2"
PROTO = Path(__file__).resolve().parent / "retouch_v2_protocol.json"
SEED = 20261119
N = 64
SEEDS = [0, 1, 2, 3, 4]
FACTORS = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16]
SOURCES = ["AVA", "AADB", "PARA", "TAD", "FLICKR_AES"]


class KeyScorer:
    """把若干 (BankScorer, 键→坐标) 组合成一个打分器：对一组键给出各种子平均的 Q。"""

    def __init__(self, parts):
        self.parts = parts          # [(BankScorer, dict key->coords)]

    def score(self, keys):
        qs = [sc.score(np.stack([lk[k] for k in keys])) for sc, lk in self.parts]
        return {m: float(np.mean([q[f"Q_{m}"] for q in qs])) for m in ("contrast", "ideal", "low")}

    def delta(self, kb, ka):
        b, a = self.score(kb), self.score(ka)
        return {m: a[m] - b[m] for m in b}


def fb_scorer(extra=None):
    """FB-R1S：每个种子 rep() → 归一化 → R1S 第 0 次抽样 H∪L 拟合 16 维 PCA → 抽样×来源单元。
    extra：另一个原始特征文件（键 paths、特征 coords），只追加到查询表，不参与 PCA 与参考库。"""
    import torch
    sys.path.insert(0, str(REPO / "6_aes_eval"))
    import train_unified as tu  # noqa: E402
    cfg = json.loads((REPO / "6_aes_eval/runs/fusion_release/config.json").read_text())
    z = np.load(OUT / "fb_raw.npz")
    keys = z["keys"].astype(str)
    raw = z["fb"].astype(np.float32)
    if extra is not None:
        e = np.load(extra)
        ek = e["paths"].astype(str)
        new = ~np.isin(ek, keys)
        keys = np.concatenate([keys, ek[new]])
        raw = np.concatenate([raw, e["coords"].astype(np.float32)[new]])
    X = torch.from_numpy(raw).cuda()
    bank = pd.read_parquet(BG / "banks.parquet")
    bank = bank[bank.arm.eq("source_balanced")]
    pos = {k: i for i, k in enumerate(keys)}
    parts = []
    for k in cfg["种子"]:
        sc = tu.Scorer(768).cuda()
        sc.load_state_dict(torch.load(cfg["模型"][f"seed{k}"]["FB_l0.3"]["path"], map_location="cuda"))
        sc.eval()
        with torch.no_grad():
            R = torch.cat([sc.rep(X[i:i + 4096]) for i in range(0, len(X), 4096)])
            R = (R / R.norm(dim=-1, keepdim=True)).double().cpu().numpy()
        fit = R[[pos[p] for p in bank[bank.draw.eq(0)].path]]
        center = fit.mean(0)
        _, _, vt = np.linalg.svd(fit - center, full_matrices=False)
        W = vt[:16].T
        C = (R - center) @ W
        lookup = dict(zip(keys, C))
        units = []
        for draw in sorted(bank.draw.unique()):
            for src in SOURCES:
                cell = bank[bank.draw.eq(draw) & bank.source.eq(src)]
                h = bq.stats(np.stack([lookup[p] for p in cell.loc[cell.side.eq("H"), "path"]]))
                l = bq.stats(np.stack([lookup[p] for p in cell.loc[cell.side.eq("L"), "path"]]))
                zz = np.stack([lookup[p] for p in cell.path])
                units.append(bq.Unit(h, l, float((2 * zz @ (h[0] - l[0])).std(ddof=1)), f"FB{k}|{draw}|{src}"))
        parts.append((bq.BankScorer(units, f"FB-R1S seed{k}"), lookup))
        print(f"  FB 种子 {k}：PCA 解释方差 {float((np.linalg.svd(fit - center, compute_uv=False)[:16] ** 2).sum() / ((fit - center) ** 2).sum()):.3f}", flush=True)
    return KeyScorer(parts)


def clip_scorers():
    Z = bq.load_coords(REPO)
    cz = np.load(OUT / "clip_v1all.npz")
    lookup = dict(zip(cz["paths"].astype(str), cz["coords"]))
    return {"CLIP-R1S-contrast": KeyScorer([(bq.round1_scorer(Z, REPO, "source_balanced"), lookup)]),
            "CLIP-R1C-contrast": KeyScorer([(bq.round1_scorer(Z, REPO, "source_and_color_balanced"), lookup)]),
            "CLIP-V2G-contrast": KeyScorer([(bq.v2_global_scorer(Z, REPO), lookup)])}


# ---------------------------------------------------------------- V1：全部版本
def pack(order, families):
    """按 order 依次放入第一个还没有该组、且未满 64 的批；只返回装满的批（行号列表）。"""
    batches, fams = [], []
    for i in order:
        f = families[i]
        for b, fs in zip(batches, fams):
            if len(b) < N and f not in fs:
                b.append(i); fs.add(f)
                break
        else:
            batches.append([i]); fams.append({f})
    return [b for b in batches if len(b) == N]


def eval_v1(scorers):
    pairs = pd.read_parquet(OUT / "v1all_pairs.parquet")
    rows, boot = [], []
    for sub in ["vivid", "muted", "bw"]:
        g = pairs[pairs.subgroup.eq(sub)].reset_index(drop=True)
        n = len(g)
        fam = g.family.to_numpy()
        kin, kex = g.input_path.to_numpy(), g.expert_path.to_numpy()
        rev = np.zeros(n, bool)
        rev[np.random.default_rng(SEED).permutation(n)[: n // 2]] = True
        kb, ka = np.where(rev, kex, kin), np.where(rev, kin, kex)
        lab = np.where(rev, -1.0, 1.0)
        parts = {s: pack(np.random.default_rng(SEED + s).permutation(n), fam) for s in SEEDS}
        for name, sc in scorers.items():
            for s, bl in parts.items():
                for b, ix in enumerate(bl):
                    same = sc.delta(kin[ix], kex[ix])
                    mixd = sc.delta(kb[ix], ka[ix])
                    rows.append({"cand": name, "subgroup": sub, "seed": s, "batch": b, "label": float(lab[ix].mean()),
                                 "same": same["contrast"], "mixed": mixd["contrast"],
                                 "same_ideal": same["ideal"], "same_low": same["low"]})
            if sub == "bw":
                continue
            rng = np.random.default_rng(SEED + 1000)
            ufam = np.unique(fam)
            members = {f: np.flatnonzero(fam == f) for f in ufam}
            for t in range(200):
                draw = rng.choice(ufam, len(ufam), replace=True)
                idx = np.concatenate([members[f] for f in draw])
                tag = np.concatenate([[f"{f}#{j}"] * len(members[f]) for j, f in enumerate(draw)])
                bl = pack(np.random.default_rng(SEED).permutation(len(idx)), tag)
                dq, lb = [], []
                for ix in bl:
                    ii = idx[ix]
                    dq.append(sc.delta(kb[ii], ka[ii])["contrast"]); lb.append(lab[ii].mean())
                boot.append({"cand": name, "subgroup": sub, "rep": t, "rho": rho(dq, lb)})
            print(f"  V1 {sub} {name} 完成（每划分 {len(parts[0])} 批）", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v1all_batches.csv", index=False)
    return df, pd.DataFrame(boot)


# ---------------------------------------------------------------- FB 的 V3 / V4 / V5
def fb_v3(sc):
    lv = pd.read_parquet(V1 / "v3_levels.parquet")
    rows, summ = [], {}
    for axis, g in lv.groupby("axis"):
        votes = pd.read_csv(Path("/home/chen/Work/datasets/REPID/extreme_styles") / f"extreme_style_{axis}.csv")
        base = "con+0" if axis == "contrast" else "original"
        levels, scenes = sorted(g.level.unique()), sorted(g.scene.unique())
        path = {(r.level, r.scene): r.path for r in g.itertuples()}
        hc = human_curve(votes)

        def curve(sl):
            qb = sc.score([path[(base, s)] for s in sl])["contrast"]
            return {l: sc.score([path[(l, s)] for s in sl])["contrast"] - qb for l in levels}

        mc = curve(scenes)
        rows += [{"axis": axis, "level": l, "delta_Q": mc[l], "human": hc[l]} for l in levels]
        rng = np.random.default_rng(SEED + 7)
        bs = []
        for _ in range(1000):
            pick = rng.choice(scenes, len(scenes), replace=True)
            m2, h2 = curve(list(pick)), human_curve(votes, pd.Series(pick).value_counts().to_dict())
            bs.append(rho([m2[l] for l in levels], [h2[l] for l in levels]))
        bs = np.array([b for b in bs if not np.isnan(b)])
        summ[axis] = {"spearman": rho([mc[l] for l in levels], [hc[l] for l in levels]),
                      "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                      "metric_best": max(levels, key=mc.get), "human_best": max(hc, key=hc.get)}
        print(f"  FB V3 {axis}: Spearman {summ[axis]['spearman']:+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "fb_v3.csv", index=False)
    return summ


def fb_v4(sc):
    pairs = pd.read_parquet(BG / "calibration_pairs.parquet")
    bi = pd.read_parquet(BG / "bidirectional_pairs.parquet")
    assert (bi.family_id.values == pairs.family_id.values).all()
    batches = pd.read_parquet(BG / "batches.parquet")
    rows = []
    for (n, seed, b), g in batches.groupby(["n_requested", "seed", "batch"]):
        if n != N or g.partial.any():
            continue
        r, rb = pairs.iloc[g.pair_row], bi.iloc[g.pair_row]
        po, pe = r.path_original.to_numpy(), r.path_edited.to_numpy()
        rev = rb.reversed.to_numpy()
        same = sc.delta(po, pe)["contrast"]
        mixd = sc.delta(np.where(rev, pe, po), np.where(rev, po, pe))["contrast"]
        rows.append({"seed": int(seed), "batch": int(b), "delta_mos": float(r.delta_mos.mean()),
                     "oriented_delta_mos": float(rb.oriented_delta_mos.mean()), "same": same, "mixed": mixd})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fb_v4.csv", index=False)
    return {"same_spearman_mean": float(np.nanmean([rho(d.same, d.delta_mos) for _, d in df.groupby("seed")])),
            "mixed_spearman_mean": float(np.nanmean([rho(d.mixed, d.oriented_delta_mos) for _, d in df.groupby("seed")]))}


def fb_v5(sc):
    so = pd.read_parquet(BG / "synthetic_originals.parquet")
    fam = so.family_id.astype(str).to_numpy()
    perm = np.random.default_rng(SEED).permutation(len(fam))
    rows = []
    for b in range(len(fam) // N):
        ix = perm[b * N:(b + 1) * N]
        base = sc.score([f"SYN:{f}|1" for f in fam[ix]])["contrast"]
        for fac in FACTORS:
            rows.append({"batch": b, "factor": fac, "dQ": sc.score([f"SYN:{f}|{fac}" for f in fam[ix]])["contrast"] - base})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fb_v5.csv", index=False)
    m = df.groupby("factor").dQ.mean()
    return {"mean_dQ": {str(k): float(v) for k, v in m.items()}, "peak_factor": float(m.idxmax()),
            "dQ_8x_below_original": bool(m[8] < 0), "dQ_16x_below_original": bool(m[16] < 0)}


def main():
    h = hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    assert h == hashlib.sha256(PROTO.read_bytes()).hexdigest()
    scorers = clip_scorers()
    scorers["FB-R1S-contrast"] = fb_scorer()
    print("开始评测", flush=True)
    v1, v1b = eval_v1(scorers)
    fb = scorers["FB-R1S-contrast"]
    fb3, fb4, fb5 = fb_v3(fb), fb_v4(fb), fb_v5(fb)
    old = json.loads((V1 / "summary.json").read_text())

    summary, judge = {}, {}
    for cand in scorers:
        s = {}
        for sub in ["vivid", "muted", "bw"]:
            d = v1[(v1.cand == cand) & (v1.subgroup == sub)]
            bb = v1b[(v1b.cand == cand) & (v1b.subgroup == sub)].rho.dropna() if sub != "bw" else pd.Series(dtype=float)
            s[f"V1_{sub}"] = {"batches": int(len(d)), "same_mean_dQ": float(d.same.mean()), "same_frac_pos": float((d.same > 0).mean()),
                              "mixed_spearman_mean": float(np.nanmean([rho(dd.mixed, dd.label) for _, dd in d.groupby("seed")])),
                              "mixed_ci": [float(bb.quantile(.025)), float(bb.quantile(.975))] if len(bb) else None,
                              "decomp_same_ideal": float(d.same_ideal.mean()), "decomp_same_low": float(d.same_low.mean())}
        if cand.startswith("CLIP"):
            o = old[cand.replace("CLIP-", "")]
            s.update({"V2_expert_vs_expert": o["V2_expert_vs_expert"], "V2_vs_neutral": o["V2_vs_neutral"],
                      "V3": o["V3"], "V4": o["V4"], "V5": o["V5"], "note": "V2–V5 沿用 retouch-v1（定义与数据相同）"})
        else:
            s.update({"V3": fb3, "V4": fb4, "V5": fb5, "note": "V2 不报告：REPID train 被 FB 训练使用"})
        summary[cand] = s
    ref = summary["CLIP-R1S-contrast"]["V4"]["mixed_spearman_mean"]
    for cand, s in summary.items():
        k1 = all(s[f"V1_{g}"]["same_mean_dQ"] > 0 and s[f"V1_{g}"]["same_frac_pos"] >= 0.8 and
                 s[f"V1_{g}"]["mixed_spearman_mean"] > 0 and s[f"V1_{g}"]["mixed_ci"][0] > 0 for g in ["vivid", "muted"])
        k2 = all(v["spearman"] > 0 for v in s["V3"].values())
        own = ref if cand in ("FB-R1S-contrast", "CLIP-R1S-contrast") else s["V4"]["mixed_spearman_mean"]
        k3 = s["V4"]["mixed_spearman_mean"] >= 0.5 and s["V4"]["mixed_spearman_mean"] >= own - 0.05
        judge[cand] = {"K1": bool(k1), "K2": bool(k2), "K3": bool(k3), "pass": bool(k1 and k2 and k3)}
    passing = [c for c, j in judge.items() if j["pass"]]
    choice = None
    if passing:
        best = max(summary[c]["V1_muted"]["mixed_spearman_mean"] for c in passing)
        near = [c for c in passing if best - summary[c]["V1_muted"]["mixed_spearman_mean"] < 0.02]
        pri = {"CLIP-R1S-contrast": 0, "FB-R1S-contrast": 1, "CLIP-R1C-contrast": 2, "CLIP-V2G-contrast": 3}
        choice = sorted(near, key=pri.get)[0]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=True))
    (OUT / "judge.json").write_text(json.dumps({"protocol_sha256": h, "candidates": judge, "passing": passing, "selected": choice},
                                               ensure_ascii=False, indent=1))
    print(json.dumps({"candidates": judge, "passing": passing, "selected": choice}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
