#!/usr/bin/env python3
"""extract_full.py — 为 full_manifest.parquet 里的全部图片提取 CLIP 16 维坐标（与 coords.npy 同一空间）
================================================================================

可续跑：每 2,048 张写一个分片，已存在的分片跳过。产物在 <bg>/retouch_full/：
  默认    full_coords.npz（path → CLIP 16 维坐标）
  --fb    full_fb_raw.npz（path → AesCLIP 投影前 768 维 L2 归一化特征，float16，供 FB 方案）

用法（aes 环境）：python 6_aes_eval/batch_retouch/extract_full.py [--fb]
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

MAN = bq.bg_dir() / "retouch_v1" / "full_manifest.parquet"
OUT = bq.bg_dir() / "retouch_full"
SHARD = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb", action="store_true")
    args = ap.parse_args()
    m = pd.read_parquet(MAN)
    paths = sorted(set(m.loc[m.exists, "path"]))
    OUT.mkdir(exist_ok=True)
    sd = OUT / ("full_fb" if args.fb else "full_coords")
    sd.mkdir(exist_ok=True)
    import torch
    from PIL import Image, ImageOps
    cc = bq.ClipCoords()
    if args.fb:
        sys.path.insert(0, str(bq.REPO_DEFAULT / "6_aes_eval"))
        from encoders import build
        enc = build("aesclip_pre")
    else:
        enc = cc.enc

    def load(p):
        with Image.open(p) as im:
            return enc.preprocess(cc.resize_long(ImageOps.exif_transpose(im).convert("RGB")))

    t0 = time.time()
    for s in range(0, len(paths), SHARD):
        f = sd / f"{s:06d}.npz"
        if f.exists():
            continue
        part = paths[s:s + SHARD]
        feats = []
        with ThreadPoolExecutor(10) as ex:
            for i in range(0, len(part), 64):
                xs = list(ex.map(load, part[i:i + 64]))
                feats.append(enc.encode(torch.stack(xs)).cpu().numpy())
        F = np.concatenate(feats)
        if args.fb:
            np.savez(f, paths=np.array(part), coords=(F / np.linalg.norm(F, axis=1, keepdims=True)).astype(np.float16))
        else:
            np.savez(f, paths=np.array(part), coords=cc.project(F))
        print(f"  {min(s + SHARD, len(paths))}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
    shards = [np.load(sd / f"{s:06d}.npz") for s in range(0, len(paths), SHARD)]
    P = np.concatenate([z["paths"] for z in shards])
    C = np.concatenate([z["coords"] for z in shards])
    assert len(P) == len(paths) and (P == np.array(paths)).all()
    name = "full_fb_raw.npz" if args.fb else "full_coords.npz"
    np.savez(OUT / name, paths=P, coords=C)
    print(f"完成：{len(P)} 张 → {OUT / name}", flush=True)


if __name__ == "__main__":
    main()
