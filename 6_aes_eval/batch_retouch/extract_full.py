#!/usr/bin/env python3
"""extract_full.py — 为 full_manifest.parquet 里的全部图片提取 CLIP 16 维坐标（与 coords.npy 同一空间）
================================================================================

可续跑：每 2,048 张写一个分片，已存在的分片跳过。产物 <bg>/retouch_v1/full_coords/（分片）与 full_coords.npz（path → 坐标）。

用法（aes 环境）：python 6_aes_eval/batch_retouch/extract_full.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

OUT = bq.bg_dir() / "retouch_v1"
SHARD = 2048


def main():
    m = pd.read_parquet(OUT / "full_manifest.parquet")
    paths = sorted(set(m.loc[m.exists, "path"]))
    sd = OUT / "full_coords"
    sd.mkdir(exist_ok=True)
    cc = bq.ClipCoords()
    import torch
    from PIL import Image, ImageOps

    def load(p):
        with Image.open(p) as im:
            return cc.enc.preprocess(cc.resize_long(ImageOps.exif_transpose(im).convert("RGB")))

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
                feats.append(cc.enc.encode(torch.stack(xs)).cpu().numpy())
        np.savez(f, paths=np.array(part), coords=cc.project(np.concatenate(feats)))
        print(f"  {min(s + SHARD, len(paths))}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
    shards = [np.load(sd / f"{s:06d}.npz") for s in range(0, len(paths), SHARD)]
    P = np.concatenate([z["paths"] for z in shards])
    C = np.concatenate([z["coords"] for z in shards])
    assert len(P) == len(paths) and (P == np.array(paths)).all()
    np.savez(OUT / "full_coords.npz", paths=P, coords=C)
    print(f"完成：{len(P)} 张 → {OUT / 'full_coords.npz'}", flush=True)


if __name__ == "__main__":
    main()
