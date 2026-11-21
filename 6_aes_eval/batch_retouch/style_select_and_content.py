#!/usr/bin/env python3
"""style_select_and_content.py — 论文写法所需的三项补充分析（最终评测方案 retouch-final）
================================================================================

① 评测集选风格：在评测集（eval_dev，202 张训练划分里的留出照片，未参与训练）上给 6 种风格打分，取最高者。
   选择只看评测集，不看测试集结果。
② 测试集配对检验：选出的风格、6 候选平均分别对比 DiffRetouch 等方法——同一批照片、同样划分，
   ΔQ 差值的批均值与按照片重抽 2,000 次的 95% 区间；另报「6 个风格都拿去比」时的 Bonferroni 校正区间（99.17%）。
③ 内容保持：测试照片上每个方法的输出与其输入图的结构相似度（长边 512 灰度；Sobel 梯度幅值的相关系数、SSIM），
   用来判断哪些方法改动了画面内容（专家终图存在构图裁剪差异，不参与）。
   ⚠ 这里没有做几何对齐：qwen-image / muse-image 的输出有不到 1% 长边的整体平移与缩放，未对齐的相关（0.58 / 0.80）
   会把它误当成内容改动。生图模型请以 align_b_outputs.py 的对齐后结果为准（qwen 0.967、muse 0.966、gpt-5-image-mini 0.309）。

产物：<bg>/retouch_final/final_model_test/{dev_style_selection.json, paired_tests.json, content_preservation.csv}
用法（aes 环境）：python 6_aes_eval/batch_retouch/style_select_and_content.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402
import score_final_model as sfm  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
FT, FD = REPO / "7_baselines/runs/FINAL_TEST", REPO / "7_baselines/runs/FINAL_DEV"
OUT = BG / "retouch_final" / "final_model_test"
STYLES = sfm.STYLES
BOOT = 2000


def setup_test(score):
    """与 score_final_model.py 相同的测试集数据结构（复用其划分与取条件规则）。"""
    lst = pd.DataFrame(json.loads((FT / "list.json").read_text()))
    o, c = lst[lst.arm == "原图_jpg95"], lst[lst.arm == "最终模型"]
    sfm.G.update(score=score, orig=dict(zip(o.cid, o.path)), cand={(r.cid, int(r.k)): r.path for r in c.itertuples()},
                 conds={g: sorted(d.cid) for g, d in o.groupby("group")})
    m = pd.read_parquet(BG / "retouch_v1" / "full_manifest.parquet")
    m = m[m.exists & (m.split == "test")]
    sfm.G["piv"] = m[m.cls != "专家"].pivot_table(index="family", columns="arm", values="path", aggfunc="first")
    rng = np.random.default_rng(sfm.SEED)
    exp_rows = m[m.cls == "专家"].sort_values(["family", "version"])
    sfm.G["ex"] = {f: d.path.iloc[rng.integers(len(d))] for f, d in exp_rows.groupby("family", sort=True)}
    ref = [a for a in m[~m.cls.isin(["对照", "专家"])].arm.unique()]
    common = sfm.G["piv"].dropna(subset=ref + ["原图_jpg95", "原图"]).index.intersection(list(sfm.G["conds"]))
    return np.array(sorted(p for p in common if p in sfm.G["ex"])), ref, m


def dq_batches(arm, photos, seeds, pick_seed=None):
    """arm 为 'k0'..'k5'、'mean6' 或基线名；返回每批 ΔQ（照片与条件的取法与主表一致）。"""
    sc, out = sfm.G["score"], []
    for s in seeds:
        cids = sfm.pick_cond(photos, s if pick_seed is None else pick_seed)
        for ix in sfm.batches_for(photos, s):
            if arm == "mean6":
                qb = sc([sfm.G["orig"][c] for c in cids[ix]])
                out.append(np.mean([sc([sfm.G["cand"][(c, k)] for c in cids[ix]]) - qb for k in range(6)]))
            else:
                kb, ka = sfm.arm_keys(arm, photos[ix], cids[ix])
                out.append(sc(ka) - sc(kb))
    return np.array(out)


def paired(task):
    a, b, photos = task
    d0 = dq_batches(a, photos, sfm.SEEDS) - dq_batches(b, photos, sfm.SEEDS)
    rng = np.random.default_rng(sfm.SEED + 3000)
    bs = []
    for _ in range(BOOT):
        ph = photos[rng.integers(0, len(photos), len(photos))]
        bs.append(np.mean(dq_batches(a, ph, [0]) - dq_batches(b, ph, [0])))
    bs = np.array(bs)
    return {"a": a, "b": b, "diff": float(d0.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "ci_bonf6": [float(np.quantile(bs, .025 / 6)), float(np.quantile(bs, 1 - .025 / 6))], "p_le0": float((bs <= 0).mean())}


def structure(pair):
    """输入、输出两张图：长边 512 灰度；Sobel 梯度幅值相关系数与 SSIM。"""
    from PIL import Image
    from scipy import ndimage
    from skimage.metrics import structural_similarity
    a_path, b_path = pair
    with Image.open(a_path) as A, Image.open(b_path) as B:
        A = A.convert("L"); B = B.convert("L")
        s = 512 / max(A.size)
        A = A.resize((round(A.width * s), round(A.height * s)), Image.BICUBIC)
        B = B.resize(A.size, Image.BICUBIC)            # 尺寸不同（生图模型可能改变构图）时对齐到输入尺寸
        a, b = np.asarray(A, np.float64), np.asarray(B, np.float64)
    ga = np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1)).ravel()
    gb = np.hypot(ndimage.sobel(b, 0), ndimage.sobel(b, 1)).ravel()
    return float(np.corrcoef(ga, gb)[0, 1]), float(structural_similarity(a, b, data_range=255))


def main():
    # ① 评测集选风格
    dl = pd.DataFrame(json.loads((FD / "list.json").read_text()))
    sfm.extract(sorted(dl.path.unique()), OUT / "final_dev_fb_raw.npz")
    from final_scheme import final_scorer
    ks = final_scorer(extra=[BG / "retouch_full/full_fb_raw.npz", OUT / "final_test_fb_raw.npz", OUT / "final_dev_fb_raw.npz"])
    score = lambda keys: ks.score(list(keys))["contrast"]  # noqa: E731

    o, c = dl[dl.arm == "原图_jpg95"], dl[dl.arm == "最终模型"]
    sfm.G.update(score=score, orig=dict(zip(o.cid, o.path)), cand={(r.cid, int(r.k)): r.path for r in c.itertuples()},
                 conds={g: sorted(d.cid) for g, d in o.groupby("group")})
    dev_ph = np.array(sorted(sfm.G["conds"]))
    dev = {}
    for k in range(6):
        d = dq_batches(f"k{k}", dev_ph, sfm.SEEDS)
        rng = np.random.default_rng(sfm.SEED + 4000)
        bs = [np.mean(dq_batches(f"k{k}", dev_ph[rng.integers(0, len(dev_ph), len(dev_ph))], [0])) for _ in range(200)]
        dev[f"风格{STYLES[k]}"] = {"cand": f"k{k}", "dQ": float(d.mean()), "batches": len(d),
                                   "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    best = max(dev, key=lambda s: dev[s]["dQ"])
    (OUT / "dev_style_selection.json").write_text(json.dumps({"dev_photos": int(len(dev_ph)), "per_style": dev, "selected": best,
                                                              "rule": "评测集上 ΔQ 最高的风格；只看评测集"}, ensure_ascii=False, indent=1))
    print(f"① 评测集（{len(dev_ph)} 张）各风格：", {s: round(v["dQ"], 3) for s, v in dev.items()}, "→ 选", best, flush=True)

    # ② 测试集配对检验
    photos, ref, m = setup_test(score)
    sel = dev[best]["cand"]
    tasks = [(sel, b, photos) for b in ["DiffRetouch", "CLIPtone_Beautiful", "Lightroom自动", "gemini-3.8-flash", "qwen-image-2.1-turbo"]]
    tasks += [("mean6", b, photos) for b in ["DiffRetouch", "Lightroom自动", "gemini-3.8-flash"]]
    tasks += [(f"k{k}", "DiffRetouch", photos) for k in range(6) if f"k{k}" != sel]
    with mp.get_context("fork").Pool(14) as pool:
        res = pool.map(paired, tasks)
    for r in res:
        r["a_label"] = "6候选平均" if r["a"] == "mean6" else f"风格{STYLES[int(r['a'][1:])]}"
    (OUT / "paired_tests.json").write_text(json.dumps({"test_photos": int(len(photos)), "selected_on_dev": best, "rows": res},
                                                      ensure_ascii=False, indent=1))
    for r in res:
        print(f"② {r['a_label']} − {r['b']}: {r['diff']:+.3f}  95% {r['ci95'][0]:+.3f}～{r['ci95'][1]:+.3f}  "
              f"Bonferroni(6) {r['ci_bonf6'][0]:+.3f}～{r['ci_bonf6'][1]:+.3f}", flush=True)

    # ③ 内容保持
    pairs, labels = [], []
    cids0 = sfm.pick_cond(photos, 0)
    for p, cid in zip(photos, cids0):
        for k in range(6):
            pairs.append((sfm.G["orig"][cid], sfm.G["cand"][(cid, k)])); labels.append(("我们", f"风格{STYLES[k]}"))
        for a in ref:
            pairs.append((sfm.G["piv"].loc[p, "原图_jpg95"], sfm.G["piv"].loc[p, a]))
            labels.append((m[m.arm == a].cls.iloc[0], a))
    with mp.get_context("fork").Pool(16) as pool:
        vals = pool.map(structure, pairs, chunksize=32)
    df = pd.DataFrame([{"cls": c, "arm": a, "edge_corr": e, "ssim": s} for (c, a), (e, s) in zip(labels, vals)])
    summ = df.groupby(["cls", "arm"]).agg(edge_median=("edge_corr", "median"), edge_p10=("edge_corr", lambda x: x.quantile(.1)),
                                          frac_edge_lt_0p8=("edge_corr", lambda x: (x < 0.8).mean()),
                                          ssim_median=("ssim", "median")).reset_index().sort_values("edge_median")
    summ.to_csv(OUT / "content_preservation.csv", index=False)
    print("③ 内容保持（梯度相关中位数，越低改动越大）:")
    print(summ.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
