#!/usr/bin/env python3
"""final_scheme.py — 最终评测方案 retouch-final：批量双库 · FB 表征 · 四来源参考库（不含 AVA）
================================================================================

用户 2026-11-21 决定采纳。配置见 retouch_final_protocol.json：
  · 表征：AesCLIP 投影前 768 维 L2 归一化 → 发布版 FB_l0.3 的 rep()（种子 0/1/2）→ L2 归一化 → 16 维 PCA
  · PCA：只用首轮 R1S 参考库第 0 次抽样中 AADB / PARA / TAD / FLICKR_AES 的 H∪L 拟合
  · 参考库：R1S 成员中上述四个来源（每来源每侧 300 张 × 5 次抽样），AVA 全部移除（其参考图 100% 在 FB 训练集内）
  · 分数：每批 64 张，Q = (D_L − D_H)/尺度，20 个（抽样×来源）单元等权平均，再对 3 个种子平均

用法：from final_scheme import final_scorer; sc = final_scorer(extra=...); sc.delta(原图键, 输出键)["contrast"]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v2 import fb_scorer  # noqa: E402

FINAL_SOURCES = ["AADB", "PARA", "TAD", "FLICKR_AES"]


def final_scorer(extra=None):
    """extra：附加的原始 AesCLIP 特征文件（如全数据集 full_fb_raw.npz），只进查询表，不影响 PCA 与参考库。"""
    return fb_scorer(extra=extra, sources=FINAL_SOURCES, pca_sources=FINAL_SOURCES)
