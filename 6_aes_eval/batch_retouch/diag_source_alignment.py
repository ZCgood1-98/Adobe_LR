#!/usr/bin/env python3
"""diag_source_alignment.py — 事后诊断：FB 表征（AVA+REPID 训练）与五来源参考库是否「对不齐」
================================================================================

不属于 retouch-v2 协议的判定，只回答一个问题：FB 方案的效果是否依赖 AVA（FB 训练集）那部分参考库，
以及其他四个来源的高低分在 FB 空间里是否与 AVA 方向一致。

  ① 重叠：R1S 参考库里的 AVA 图有多少在 FB 的训练拟合集 / 开发集里
  ② 方向一致性（FB 与 CLIP 各算一遍）：
       · 各来源「高分均值 − 低分均值」方向两两余弦
       · 留一来源 AUC：用其余四个来源的方向去区分该来源自己的高 / 低分图
       · FB 评分器打分本身区分各来源高 / 低分的 AUC
  ③ 只用单个来源 / 去掉 AVA 的参考库，重算 V1（全部版本）、V3（极端风格，点估计）、V4（MSC 双向）
  ④ 去掉 AVA 后，全数据集各模型臂的总体 ΔQ 排名与原 FB 表的 Spearman

用法（aes 环境）：python 6_aes_eval/batch_retouch/diag_source_alignment.py
产物：<bg>/retouch_v2/diag_source_alignment.json
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
from evaluate_v1 import human_curve, rho  # noqa: E402
from evaluate_v2 import KeyScorer, fb_scorer, pack, SEED, N, SEEDS  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
V1D, V2D, FULL = BG / "retouch_v1", BG / "retouch_v2", BG / "retouch_full"
SOURCES = ["AVA", "AADB", "PARA", "TAD", "FLICKR_AES"]


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def subset(ks: KeyScorer, keep) -> KeyScorer:
    """只保留标签里来源属于 keep 的单元（标签形如 'xxx|draw|SRC'）。"""
    parts = []
    for sc, lk in ks.parts:
        units = [u for u in sc.units if u.tag.split("|")[-1] in keep]
        parts.append((bq.BankScorer(units, sc.name + "|" + "+".join(keep)), lk))
    return KeyScorer(parts)


def overlap():
    sys.path.insert(0, str(REPO / "6_aes_eval"))
    import train_unified as tu  # noqa: E402
    fit, dev, test = tu.ava_frames()
    b = pd.read_parquet(BG / "banks.parquet")
    ids = set(b[(b.arm == "source_balanced") & (b.source == "AVA")].id.astype(str))
    f, d, t = set(fit.id), set(dev.id), set(test.id)
    return {"ava_bank_images": len(ids), "in_fb_fit": len(ids & f), "in_fb_dev": len(ids & d),
            "in_ava_test": len(ids & t), "other": len(ids - f - d - t)}


def directions(lookup_of, bank):
    """lookup_of(path_or_idx) → 16 维坐标；返回 各来源方向余弦矩阵 与 留一来源 AUC。"""
    mu = {}
    pts = {}
    for src in SOURCES:
        g = bank[bank.source == src].drop_duplicates("key")
        H = np.stack([lookup_of(r) for r in g[g.side == "H"].itertuples()])
        L = np.stack([lookup_of(r) for r in g[g.side == "L"].itertuples()])
        mu[src] = H.mean(0) - L.mean(0)
        pts[src] = (H, L)
    cos = {a: {b: float(mu[a] @ mu[b] / np.linalg.norm(mu[a]) / np.linalg.norm(mu[b])) for b in SOURCES} for a in SOURCES}
    loso = {}
    for t in SOURCES:
        d = sum(mu[s] / np.linalg.norm(mu[s]) for s in SOURCES if s != t)
        H, L = pts[t]
        loso[t] = auc(H @ d, L @ d)
    own = {s: auc(pts[s][0] @ mu[s], pts[s][1] @ mu[s]) for s in SOURCES}
    return cos, loso, own


def eval_variant(sc, v1pairs, v3, v4):
    out = {}
    # V1：全部版本（与 evaluate_v2 同一装批规则），不做 bootstrap
    for sub in ["vivid", "muted", "bw"]:
        g = v1pairs[v1pairs.subgroup.eq(sub)].reset_index(drop=True)
        n, fam = len(g), g.family.to_numpy()
        kin, kex = g.input_path.to_numpy(), g.expert_path.to_numpy()
        rev = np.zeros(n, bool)
        rev[np.random.default_rng(SEED).permutation(n)[: n // 2]] = True
        kb, ka, lab = np.where(rev, kex, kin), np.where(rev, kin, kex), np.where(rev, -1.0, 1.0)
        same, mixed = [], []
        for s in SEEDS:
            dq, lb = [], []
            for ix in pack(np.random.default_rng(SEED + s).permutation(n), fam):
                same.append(sc.delta(kin[ix], kex[ix])["contrast"])
                dq.append(sc.delta(kb[ix], ka[ix])["contrast"]); lb.append(lab[ix].mean())
            mixed.append(rho(dq, lb))
        out[f"V1_{sub}"] = {"same_mean_dQ": float(np.mean(same)), "same_frac_pos": float(np.mean(np.array(same) > 0)),
                            "mixed_spearman_mean": float(np.nanmean(mixed))}
    # V3：极端风格，点估计
    lv, votes = v3
    for axis, g in lv.groupby("axis"):
        base = "con+0" if axis == "contrast" else "original"
        levels, scenes = sorted(g.level.unique()), sorted(g.scene.unique())
        path = {(r.level, r.scene): r.path for r in g.itertuples()}
        qb = sc.score([path[(base, s)] for s in scenes])["contrast"]
        mc = {l: sc.score([path[(l, s)] for s in scenes])["contrast"] - qb for l in levels}
        hc = human_curve(votes[axis])
        out[f"V3_{axis}"] = rho([mc[l] for l in levels], [hc[l] for l in levels])
    # V4：MSC 双向
    pairs, bi, batches = v4
    per = []
    for s in SEEDS:
        dq, lb = [], []
        for (n, seed, b), g in batches[(batches.seed == s)].groupby(["n_requested", "seed", "batch"]):
            if n != N or g.partial.any():
                continue
            r, rb = pairs.iloc[g.pair_row], bi.iloc[g.pair_row]
            po, pe, rv = r.path_original.to_numpy(), r.path_edited.to_numpy(), rb.reversed.to_numpy()
            dq.append(sc.delta(np.where(rv, pe, po), np.where(rv, po, pe))["contrast"]); lb.append(rb.oriented_delta_mos.mean())
        per.append(rho(dq, lb))
    out["V4_mixed_spearman_mean"] = float(np.nanmean(per))
    return out


def full_ranking(sc):
    """全数据集：各臂总体 ΔQ（公共组、5 种划分、每批 64，不做 bootstrap、不分类别）。"""
    m = pd.read_parquet(V1D / "full_manifest.parquet")
    m = m[m.exists]
    piv = m[m.cls != "专家"].pivot_table(index="family", columns="arm", values="path", aggfunc="first")
    arms = [a for a in m[~m.cls.isin(["对照", "专家"])].arm.unique()]
    common = piv.dropna(subset=arms + ["原图_jpg95"]).index.to_numpy()
    res = {}
    for a in arms:
        kb, ka = piv.loc[common, "原图_jpg95"].to_numpy(), piv.loc[common, a].to_numpy()
        d = []
        for s in SEEDS:
            ix_all = np.random.default_rng(SEED + s).permutation(len(common))
            for lo in range(0, len(common) - N + 1, N):
                ix = ix_all[lo:lo + N]
                d.append(sc.delta(kb[ix], ka[ix])["contrast"])
        res[a] = float(np.mean(d))
    return res


def main():
    out = {"note": "事后诊断，不属于 retouch-v2 的预注册判定"}
    out["overlap"] = overlap()
    print("① 重叠：", out["overlap"], flush=True)

    bank = pd.read_parquet(BG / "banks.parquet")
    bank = bank[bank.arm.eq("source_balanced")]
    fb = fb_scorer(extra=FULL / "full_fb_raw.npz")
    Z = bq.load_coords(REPO)
    clip_all = KeyScorer([(bq.round1_scorer(Z, REPO, "source_balanced"), None)])

    # ② 方向一致性
    fbdir = []
    for sc, lk in fb.parts:
        fbdir.append(directions(lambda r, lk=lk: lk[r.path], bank))
    cl = directions(lambda r: Z[int(r.feature_idx)], bank)
    avg = lambda key: {s: float(np.mean([d[key][s] for d in fbdir])) for s in SOURCES}  # noqa: E731
    out["direction"] = {
        "FB_cos_to_AVA": {s: float(np.mean([d[0]["AVA"][s] for d in fbdir])) for s in SOURCES},
        "CLIP_cos_to_AVA": {s: cl[0]["AVA"][s] for s in SOURCES},
        "FB_mean_offdiag_cos": float(np.mean([d[0][a][b] for d in fbdir for a in SOURCES for b in SOURCES if a != b])),
        "CLIP_mean_offdiag_cos": float(np.mean([cl[0][a][b] for a in SOURCES for b in SOURCES if a != b])),
        "FB_loso_auc": avg(1), "CLIP_loso_auc": cl[1], "FB_own_auc": avg(2), "CLIP_own_auc": cl[2]}

    # FB 评分器打分本身对各来源高低分的区分
    import torch
    sys.path.insert(0, str(REPO / "6_aes_eval"))
    import train_unified as tu  # noqa: E402
    cfg = json.loads((REPO / "6_aes_eval/runs/fusion_release/config.json").read_text())
    z = np.load(V2D / "fb_raw.npz")
    pos = {k: i for i, k in enumerate(z["keys"].astype(str))}
    X = torch.from_numpy(z["fb"].astype(np.float32)).cuda()
    score_auc = {s: [] for s in SOURCES}
    for k in cfg["种子"]:
        m = tu.Scorer(768).cuda()
        m.load_state_dict(torch.load(cfg["模型"][f"seed{k}"]["FB_l0.3"]["path"], map_location="cuda")); m.eval()
        with torch.no_grad():
            sco = torch.cat([m.score(m.rep(X[i:i + 4096]))[0] for i in range(0, len(X), 4096)]).cpu().numpy()
        for s in SOURCES:
            g = bank[bank.source == s].drop_duplicates("key")
            score_auc[s].append(auc(sco[[pos[p] for p in g[g.side == "H"].path]], sco[[pos[p] for p in g[g.side == "L"].path]]))
    out["direction"]["FB_score_auc"] = {s: float(np.mean(v)) for s, v in score_auc.items()}
    print("② 方向一致性：", json.dumps(out["direction"], ensure_ascii=False), flush=True)

    # ③ 单来源 / 去掉 AVA
    v1pairs = pd.read_parquet(V2D / "v1all_pairs.parquet")
    lv = pd.read_parquet(V1D / "v3_levels.parquet")
    votes = {a: pd.read_csv(Path("/home/chen/Work/datasets/REPID/extreme_styles") / f"extreme_style_{a}.csv") for a in lv.axis.unique()}
    v4 = (pd.read_parquet(BG / "calibration_pairs.parquet"), pd.read_parquet(BG / "bidirectional_pairs.parquet"),
          pd.read_parquet(BG / "batches.parquet"))
    variants = {"FB_all": fb, "FB_noAVA": subset(fb, SOURCES[1:])}
    for s in SOURCES:
        variants[f"FB_only_{s}"] = subset(fb, [s])
    out["variants"] = {}
    for name, sc in variants.items():
        out["variants"][name] = eval_variant(sc, v1pairs, (lv, votes), v4)
        r = out["variants"][name]
        print(f"③ {name}: V1 修淡同向 {r['V1_muted']['same_mean_dQ']:+.3f}（正 {r['V1_muted']['same_frac_pos']:.2f}）随机方向 {r['V1_muted']['mixed_spearman_mean']:+.2f}；"
              f"V3 色温 {r['V3_temperature']:+.2f} 曝光 {r['V3_exposure']:+.2f} 饱和 {r['V3_saturation']:+.2f} 对比 {r['V3_contrast']:+.2f}；V4 {r['V4_mixed_spearman_mean']:+.3f}", flush=True)

    # ④ 全数据集排名
    base = pd.read_csv(FULL / "table_fb_R1S.csv")
    base = base[(base.category == "全部") & ~base.cls.isin(["对照", "专家"])].set_index("arm").dQ
    out["full_ranking"] = {}
    for name in ["FB_noAVA", "FB_only_AVA"]:
        r = full_ranking(variants[name])
        arms = list(base.index)
        out["full_ranking"][name] = {"spearman_vs_FB_all": float(spearmanr([r[a] for a in arms], base.loc[arms]).statistic),
                                     "dQ": r}
        print(f"④ {name}：与原 FB 全量排名 Spearman {out['full_ranking'][name]['spearman_vs_FB_all']:.3f}", flush=True)
    (V2D / "diag_source_alignment.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
