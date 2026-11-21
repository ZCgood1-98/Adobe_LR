#!/usr/bin/env python3
"""align_b_outputs.py — 把生图模型的输出几何对齐到原图后再算结构相似度，区分「整体位移 / 缩放」与「内容真的变了」
================================================================================

背景：未对齐的 Sobel 梯度相关对整体位移非常敏感。qwen / muse 的输出相对原图有不到 1% 长边的整体平移与缩放
（重采样到模型自己的分辨率造成），未对齐指标会把它当成内容改动（11-21 的 content_preservation.csv 即如此）。

  · 原图与输出都转灰度、长边 512；输出先缩放到原图尺寸（与未对齐指标一致）
  · 候选对齐：不变换；ECC 仿射（128→256→512 逐级）；SIFT 匹配 + RANSAC 仿射；SIFT 初值再 ECC 细化。
    只接受缩放在 0.5–2、各向异性 <1.5、有效重叠 ≥30% 的候选；取对齐后梯度相关最高的一个
    （仿射只能整体搬动、缩放画面，造不出原图里没有的内容，所以取最优不会把内容改动「对齐掉」）
  · 指标在有效重叠区（去掉对齐后露出的边，再收缩 8 像素）上算：Sobel 梯度幅值相关、SSIM；
    记录平均位移（全图网格点经对齐矩阵的位移，占长边 %）、缩放、旋转、所选候选
  · 对象：B 类 5 个生图模型 + DiffRetouch（不改几何的修图对照）

用法（aes 环境）：python 6_aes_eval/batch_retouch/align_b_outputs.py
产物：<bg>/retouch_final/final_model_test/{b_alignment_metrics.csv（含 2×3 对齐矩阵 w00…w12，512 尺度）, b_alignment_summary.json}
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
RUNS = REPO / "7_baselines" / "runs"
OUT = bq.bg_dir(REPO) / "retouch_final" / "final_model_test"
ARMS = {"qwen-image-2.1-turbo": RUNS / "B_821e477950/qwen__qwen-image-2.1-turbo",
        "muse-image": RUNS / "B_821e477950/meta__muse-image",
        "gpt-5-image-mini": RUNS / "B_821e477950/openai__gpt-5-image-mini",
        "seedream-5-0-flash": RUNS / "B_821e477950/bytedance-seed__seedream-5-0-flash",
        "gemini-3.1-flash-lite-image": RUNS / "B_821e477950/google__gemini-3.1-flash-lite-image",
        "DiffRetouch": RUNS / "A/DiffRetouch"}
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
_SIFT = {}


def gray512(path, size=None):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if size is None:
        s = 512 / max(im.shape)
        size = (round(im.shape[1] * s), round(im.shape[0] * s))
    return cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32), size


def ecc(a, b, W0, scales):
    """ECC 仿射，从 W0 出发逐级细化；返回 W（b(W·x) ≈ a(x)，512 尺度），失败返回 None。"""
    W = W0.astype(np.float32).copy()
    for f in scales:
        sa = cv2.GaussianBlur(cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        sb = cv2.GaussianBlur(cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        Wf = W.copy(); Wf[:, 2] *= f
        try:
            _, Wf = cv2.findTransformECC(sa, sb, Wf, cv2.MOTION_AFFINE, CRIT, None, 1)
        except cv2.error:
            return None
        W = Wf.copy(); W[:, 2] /= f
    return W


def sift_init(a, b):
    """SIFT 匹配（比值检验 0.75）+ RANSAC 仿射；返回 (W, 内点数)，失败返回 (None, 0)。"""
    if "s" not in _SIFT:
        _SIFT["s"] = cv2.SIFT_create(nfeatures=4000)
    sift = _SIFT["s"]
    ka, da = sift.detectAndCompute(a.astype(np.uint8), None)
    kb, db = sift.detectAndCompute(b.astype(np.uint8), None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 10:
        return None, 0
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    W, inl = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000, confidence=0.999)
    if W is None or inl is None or int(inl.sum()) < 10:
        return None, 0
    return W.astype(np.float32), int(inl.sum())


def valid(W):
    sv = np.linalg.svd(W[:, :2], compute_uv=False)
    return bool(sv.min() > 0.5 and sv.max() < 2.0 and sv.max() / sv.min() < 1.5)


def edge_corr(a, b, mask):
    ga = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))[mask]
    gb = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))[mask]
    return float(np.corrcoef(ga, gb)[0, 1])


def warp_eval(a, b, W, size):
    bw = cv2.warpAffine(b, W, size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    m = cv2.warpAffine(np.ones_like(a), W, size, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0.5
    m = cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if m.mean() < 0.3:
        return None
    return edge_corr(a, bw, m), bw, m


def mean_disp_pct(W, size):
    """全图 16×16 网格点经 W 的平均位移，占长边的百分比。"""
    xs, ys = np.meshgrid(np.linspace(0, size[0] - 1, 16), np.linspace(0, size[1] - 1, 16))
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    return float(np.linalg.norm(W.astype(np.float64) @ P - P[:2], axis=0).mean() / max(size) * 100)


def one(job):
    from skimage.metrics import structural_similarity
    cv2.setNumThreads(1)
    uuid, arm, orig, out = job
    a, size = gray512(orig)
    b, _ = gray512(out, size)
    raw_e = edge_corr(a, b, np.ones_like(a, bool))
    raw_s = float(structural_similarity(a, b, data_range=255))
    eye = np.eye(2, 3, dtype=np.float32)
    cands = {"identity": eye}
    W = ecc(a, b, eye, (0.25, 0.5, 1.0))
    if W is not None:
        cands["ecc"] = W
    Wf, n_inl = sift_init(a, b)
    if Wf is not None:
        cands["sift"] = Wf
        W2 = ecc(a, b, Wf, (0.5, 1.0))
        if W2 is not None:
            cands["sift+ecc"] = W2
    scores, best = {}, None
    for k, W in cands.items():
        if not valid(W):
            continue
        r = warp_eval(a, b, W, size)
        if r is None:
            continue
        scores[k] = r[0]
        if best is None or r[0] > best[1]:
            best = (k, r[0], r[1], r[2], W)
    k, e, bw, m, W = best
    _, smap = structural_similarity(a, bw, data_range=255, full=True)
    A = W[:, :2]
    row = {"uuid": uuid, "arm": arm, "raw_edge": raw_e, "raw_ssim": raw_s, "aligned_edge": e,
           "aligned_ssim": float(smap[m].mean()), "method": k, "sift_inliers": n_inl,
           "disp_pct": mean_disp_pct(W, size), "scale": float(np.sqrt(abs(np.linalg.det(A)))),
           "rotation_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), "overlap": float(m.mean()),
           "w512": size[0], "h512": size[1]}
    row.update({f"edge_{c}": scores.get(c, np.nan) for c in ["identity", "ecc", "sift", "sift+ecc"]})
    row.update({f"w{i}{j}": float(W[i, j]) for i in range(2) for j in range(3)})
    return row


def main():
    jobs = []
    for p in sorted((RUNS / "A/_inputs_jpg95/test").glob("*.jpg")):
        for arm, d in ARMS.items():
            o = d / "test" / f"{p.stem}.jpg"
            if o.exists():
                jobs.append((p.stem, arm, str(p), str(o)))
    with mp.get_context("fork").Pool(16) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "b_alignment_metrics.csv", index=False)
    summ = {}
    for arm, g in df.groupby("arm"):
        summ[arm] = {"photos": int(len(g)), "raw_edge_median": round(float(g.raw_edge.median()), 3),
                     "aligned_edge_median": round(float(g.aligned_edge.median()), 3),
                     "frac_raw_lt_0p8": round(float((g.raw_edge < 0.8).mean()), 3),
                     "frac_aligned_lt_0p8": round(float((g.aligned_edge < 0.8).mean()), 3),
                     "aligned_ssim_median": round(float(g.aligned_ssim.median()), 3),
                     "disp_pct_median": round(float(g.disp_pct.median()), 3),
                     "scale_dev_median": round(float((g.scale - 1).abs().median()), 4),
                     "method_counts": {k: int(v) for k, v in g.method.value_counts().items()}}
    (OUT / "b_alignment_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(pd.DataFrame(summ).T.drop(columns=["method_counts"]).to_string())
    for arm, s in summ.items():
        print(arm, s["method_counts"])


if __name__ == "__main__":
    main()
