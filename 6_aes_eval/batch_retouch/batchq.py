#!/usr/bin/env python3
"""batchq.py — 批量双库评分的可复用入口（retouch_v1 的公共模块）
================================================================================

复用 `6_aes_eval/batch_global_20261118` 已冻结的全部组件，不改动其中任何文件：
  · 特征：OpenAI CLIP ViT-L/14（quickgelu），长边缩到 512 → 短边 224 + 中心裁剪，L2 归一化，
          以 float16 存于 all_features.npy；16 维 PCA（pca.npz：center, W）→ coords.npy
  · 参考库：首轮 banks.parquet（source_balanced / source_and_color_balanced），
          content_v2/banks.parquet（filtered_global：按参考库自身类别比例混合）
  · 距离：16 维高斯平方 Wasserstein-2；尺度 = 参考库图片在 (μ_H − μ_L) 方向上投影的标准差

一个「打分单元」= 一组 (高分分布 H, 低分分布 L, 尺度)。首轮按 抽样×来源 拆成 25 个单元，
content_v2 的 filtered_global 按抽样拆成 5 个单元。所有分数都是对单元等权平均。

对一批图 z（n×16）同时给出三种组合方式（同一组距离，只是组合不同）：
  · contrast（原方法）：Q = (D_L − D_H) / 尺度      —— 离低分库越远、离高分库越近越好
  · ideal（理想点）：  Q = −D_H / 尺度              —— 只看离高分库多近
  · low（只看低分库）：Q = D_L / 尺度               —— 只作诊断
contrast = ideal + low，可以用来拆解原方法的得分来源。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DEFAULT = Path("/home/chen/Work/adobe_lr")     # 数据与冻结产物所在的检出（工作区里没有这些大文件）
MODES = ("contrast", "ideal", "low")


def bg_dir(repo: Path = REPO_DEFAULT) -> Path:
    return Path(repo) / "6_aes_eval" / "batch_global_20261118"


# ---------------------------------------------------------------- 与 run.py 逐行等价的数学
def stats(z: np.ndarray):
    z = np.asarray(z, dtype=np.float64)
    return z.mean(0), (np.cov(z, rowvar=False, ddof=1) if len(z) > 1 else np.zeros((z.shape[1], z.shape[1])))


def sqrtm_psd(c: np.ndarray) -> np.ndarray:
    e, v = np.linalg.eigh((c + c.T) / 2)
    return (v * np.sqrt(np.maximum(e, 0))) @ v.T


def w2(mu, c, mr, cr, sr) -> float:
    """高斯平方 W2：|μ−μr|² + tr(C) + tr(Cr) − 2·tr((Cr^½ C Cr^½)^½)；sr = Cr^½。"""
    m = sr @ c @ sr
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((m + m.T) / 2), 0)).sum()
    return float(np.sum((mu - mr) ** 2) + np.trace(c) + np.trace(cr) - 2 * cross)


def mix(components, weights):
    """content_v2 的混合矩：μ = Σwμ_c，Σ = Σw(Σ_c + (μ_c−μ)(μ_c−μ)ᵀ)。"""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu = sum(ww * m for ww, (m, c) in zip(w, components))
    cov = sum(ww * (c + np.outer(m - mu, m - mu)) for ww, (m, c) in zip(w, components))
    return mu, cov


class Unit:
    """一个打分单元：高分分布、低分分布与尺度。"""

    def __init__(self, h, l, scale: float, tag: str):
        self.mh, self.ch = h
        self.ml, self.cl = l
        self.sh, self.sl = sqrtm_psd(self.ch), sqrtm_psd(self.cl)
        self.scale, self.tag = float(scale), tag
        assert self.scale > 1e-10


class BankScorer:
    def __init__(self, units: list[Unit], name: str):
        self.units, self.name = units, name

    def score(self, z: np.ndarray) -> dict:
        mu, c = stats(z)
        dh = np.array([w2(mu, c, u.mh, u.ch, u.sh) for u in self.units])
        dl = np.array([w2(mu, c, u.ml, u.cl, u.sl) for u in self.units])
        s = np.array([u.scale for u in self.units])
        return {"Q_contrast": float(np.mean((dl - dh) / s)), "Q_ideal": float(np.mean(-dh / s)),
                "Q_low": float(np.mean(dl / s)), "D_H": float(dh.mean()), "D_L": float(dl.mean())}


# ---------------------------------------------------------------- 加载冻结产物
def load_coords(repo: Path = REPO_DEFAULT) -> np.ndarray:
    return np.load(bg_dir(repo) / "coords.npy").astype(np.float64)


def round1_scorer(Z: np.ndarray, repo: Path = REPO_DEFAULT, arm: str = "source_balanced") -> BankScorer:
    """首轮：每个 (抽样, 来源) 一个单元；尺度在该来源该次抽样的 H∪L 上计算（与 run.py Scorer 相同）。"""
    bank = pd.read_parquet(bg_dir(repo) / "banks.parquet")
    bank = bank[bank.arm.eq(arm)]
    units = []
    for draw in sorted(bank.draw.unique()):
        for src in ["AVA", "AADB", "PARA", "TAD", "FLICKR_AES"]:
            cell = bank[bank.draw.eq(draw) & bank.source.eq(src)]
            h = stats(Z[cell.loc[cell.side.eq("H"), "feature_idx"].to_numpy()])
            l = stats(Z[cell.loc[cell.side.eq("L"), "feature_idx"].to_numpy()])
            zz = Z[cell.feature_idx.to_numpy()]
            scale = float((2 * zz @ (h[0] - l[0])).std(ddof=1))
            units.append(Unit(h, l, scale, f"{arm}|draw{draw}|{src}"))
    return BankScorer(units, f"round1:{arm}")


def v2_global_scorer(Z: np.ndarray, repo: Path = REPO_DEFAULT) -> BankScorer:
    """content_v2 filtered_global：每次抽样一个单元，H/L 为 类别×来源 单元的混合矩（与 pipeline.py Matcher 相同）。"""
    banks = pd.read_parquet(bg_dir(repo) / "content_v2" / "banks.parquet")
    units = []
    for draw, g in banks.groupby("draw"):
        comps, mass = {}, {}
        for cat, cg in g.groupby("content_category"):
            cells, weights = [], []
            for src, sg in cg.groupby("source"):
                hh, ll = sg[sg.side.eq("H")], sg[sg.side.eq("L")]
                assert len(hh) == len(ll)
                cells.append({"H": stats(Z[hh.feature_idx.astype(int)]), "L": stats(Z[ll.feature_idx.astype(int)])})
                weights.append(len(hh))
            comps[cat] = {s: mix([c[s] for c in cells], weights) for s in ["H", "L"]}
            mass[cat] = sum(weights)
        cats = list(comps)
        glob = {s: mix([comps[c][s] for c in cats], [mass[c] for c in cats]) for s in ["H", "L"]}
        zz = Z[g.feature_idx.astype(int)]
        scale = float((2 * zz @ (glob["H"][0] - glob["L"][0])).std(ddof=1))
        units.append(Unit(glob["H"], glob["L"], scale, f"v2_global|draw{draw}"))
    return BankScorer(units, "content_v2:filtered_global")


# ---------------------------------------------------------------- 新图片编码（与缓存同一条路径）
class ClipCoords:
    """把图片编码成与 coords.npy 同一空间的 16 维坐标：长边 512（bicubic）→ open_clip 预处理 → L2 归一化
    → 转 float16（与 all_features.npy 一致）→ (f − center) @ W。"""

    def __init__(self, repo: Path = REPO_DEFAULT, device: str = "cuda"):
        sys.path.insert(0, str(Path(repo) / "6_aes_eval"))
        from common import resize_long          # noqa: E402
        from encoders import build              # noqa: E402
        self.resize_long, self.enc = resize_long, build("clip_vitl14", device=device)
        p = np.load(bg_dir(repo) / "pca.npz")
        self.center, self.W = p["center"].astype(np.float64), p["W"].astype(np.float64)

    def load(self, path):
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return self.enc.preprocess(self.resize_long(im))

    def features(self, paths, batch: int = 32, workers: int = 6) -> np.ndarray:
        """768 维 L2 归一化特征（float32）；用线程池并行解码。"""
        import torch
        from concurrent.futures import ThreadPoolExecutor
        out = []
        with ThreadPoolExecutor(workers) as ex:
            for i in range(0, len(paths), batch):
                xs = list(ex.map(self.load, paths[i:i + batch]))
                out.append(self.enc.encode(torch.stack(xs)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def project(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float32)
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        return (f.astype(np.float16).astype(np.float64) - self.center) @ self.W
