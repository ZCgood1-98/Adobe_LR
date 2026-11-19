#!/usr/bin/env python3
"""build_sets_v2.py — retouch-v2：V1 全部专家版本 + FB 臂所需的 AesCLIP 特征
================================================================================

产物（<bg>/retouch_v2/）：
  v1all_pairs.parquet   训练分区全部 (条件, 版本)：输入图、专家终图、色彩丰富度、子组
  clip_v1all.npz        上述图片的 CLIP 16 维坐标（v1 已提取的按路径复用，其余新提取）
  fb_raw.npz            AesCLIP 投影前 768 维 L2 归一化特征（float16），键 = 图片绝对路径；
                        合成饱和度图的键 = 'SYN:<family_id>|<倍数>'
                        覆盖：首轮 R1S 参考库全部成员、V1 全部图片、极端风格、MSC 配对、合成饱和度
可续跑：FB 特征按 4,096 张分片写入 fb_shards/。

用法（aes 环境）：python 6_aes_eval/batch_retouch/build_sets_v2.py
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
BG = bq.bg_dir(REPO)
V1 = BG / "retouch_v1"
OUT = BG / "retouch_v2"
PROTO = Path(__file__).resolve().parent / "retouch_v2_protocol.json"
FACTORS = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16]
SHARD = 4096


def colorfulness(rgb_u8):
    x = rgb_u8.astype(np.float64)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def v1all_pairs():
    rows = [json.loads(line) for line in (REPO / "sft_data_v2/set_sft/conditions.jsonl").read_text().splitlines()]
    out = []
    for c in rows:
        if c["split"] != "train":
            continue
        uuid = c["source_group"].split("/")[1]
        inp = REPO / c["input_path"]
        if not inp.exists():
            continue
        for v in c["versions"]:
            ex = next((p for sp in ("train", "test")
                       if (p := REPO / "scraped_data_v4" / sp / uuid / f"version_{v['version']}" / "main_image.png").exists()), None)
            if ex is not None:
                out.append({"family": c["source_group"], "condition_id": c["condition_id"], "version": str(v["version"]),
                            "bw": bool(c["bw"]), "category": c["category"], "input_path": str(inp), "expert_path": str(ex)})
    return pd.DataFrame(out).sort_values(["family", "condition_id", "version"]).reset_index(drop=True)


def main():
    assert hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest() == hashlib.sha256(PROTO.read_bytes()).hexdigest()
    import torch
    from PIL import Image, ImageEnhance, ImageOps
    sys.path.insert(0, str(REPO / "6_aes_eval"))
    from encoders import build  # noqa: E402

    pairs = v1all_pairs()
    v1p = pd.read_parquet(V1 / "v1_pairs.parquet")
    v1c = np.load(V1 / "v1_coords.npy")
    have_clip, have_cf = {}, {}
    for r in v1p.itertuples():
        have_clip[r.input_path], have_clip[r.expert_path] = v1c[r.idx_in], v1c[r.idx_ex]
        have_cf[r.input_path], have_cf[r.expert_path] = r.cf_in, r.cf_ex
    v1_paths = sorted(set(pairs.input_path) | set(pairs.expert_path))

    bank = pd.read_parquet(BG / "banks.parquet")
    bank_paths = sorted(set(bank.loc[bank.arm.eq("source_balanced"), "path"]))
    msc = pd.read_parquet(BG / "calibration_pairs.parquet")
    msc_paths = sorted(set(msc.path_original) | set(msc.path_edited))
    v3_paths = sorted(pd.read_parquet(V1 / "v3_levels.parquet").path)
    real = sorted(set(v1_paths) | set(bank_paths) | set(msc_paths) | set(v3_paths))
    print(f"V1 全部配对 {len(pairs)}（组 {pairs.family.nunique()}）；图片：V1 {len(v1_paths)}、参考库 {len(bank_paths)}、"
          f"MSC {len(msc_paths)}、极端风格 {len(v3_paths)}；合计不同图片 {len(real)}", flush=True)

    cc = bq.ClipCoords(REPO)
    fb = build("aesclip_pre")
    need_clip = {p for p in v1_paths if p not in have_clip}
    need_cf = {p for p in v1_paths if p not in have_cf}

    def load(p):
        with Image.open(p) as im:
            im = cc.resize_long(ImageOps.exif_transpose(im).convert("RGB"))
            return (fb.preprocess(im), cc.enc.preprocess(im) if p in need_clip else None,
                    colorfulness(np.asarray(im)) if p in need_cf else None)

    sd = OUT / "fb_shards"
    sd.mkdir(exist_ok=True)
    t0 = time.time()
    for s in range(0, len(real), SHARD):
        f = sd / f"{s:06d}.npz"
        if f.exists():
            continue
        part = real[s:s + SHARD]
        F, CP, CC, CF, CFP = [], [], [], [], []
        with ThreadPoolExecutor(10) as ex:
            for i in range(0, len(part), 64):
                chunk = part[i:i + 64]
                got = list(ex.map(load, chunk))
                z = fb.encode(torch.stack([g[0] for g in got])).cpu().numpy()
                F.append(z / np.linalg.norm(z, axis=1, keepdims=True))
                ci = [j for j, g in enumerate(got) if g[1] is not None]
                if ci:
                    CC.append(cc.project(cc.enc.encode(torch.stack([got[j][1] for j in ci])).cpu().numpy()))
                    CP += [chunk[j] for j in ci]
                for j, g in enumerate(got):
                    if g[2] is not None:
                        CF.append(g[2]); CFP.append(chunk[j])
        np.savez(f, paths=np.array(part), fb=np.concatenate(F).astype(np.float16),
                 clip_paths=np.array(CP), clip=np.concatenate(CC) if CC else np.zeros((0, 16)),
                 cf_paths=np.array(CFP), cf=np.array(CF))
        print(f"  真实图片 {min(s + SHARD, len(real))}/{len(real)}  {time.time() - t0:.0f}s", flush=True)

    # 合成饱和度：与首轮 oversaturation 相同的生成方式
    syn_file = sd / "synthetic.npz"
    if not syn_file.exists():
        so = pd.read_parquet(BG / "synthetic_originals.parquet")
        ims = []
        for r in so.itertuples():
            with Image.open(r.path_original) as im:
                ims.append(cc.resize_long(ImageOps.exif_transpose(im).convert("RGB")))
        keys, F = [], []
        for fac in FACTORS:
            xs = [fb.preprocess(ImageEnhance.Color(im).enhance(fac)) for im in ims]
            for i in range(0, len(xs), 64):
                z = fb.encode(torch.stack(xs[i:i + 64])).cpu().numpy()
                F.append(z / np.linalg.norm(z, axis=1, keepdims=True))
            keys += [f"SYN:{fid}|{fac}" for fid in so.family_id]
        np.savez(syn_file, keys=np.array(keys), fb=np.concatenate(F).astype(np.float16))
        print(f"  合成饱和度 {len(keys)} 张", flush=True)

    shards = [np.load(sd / f"{s:06d}.npz") for s in range(0, len(real), SHARD)]
    syn = np.load(syn_file)
    keys = np.concatenate([z["paths"] for z in shards] + [syn["keys"]])
    feats = np.concatenate([z["fb"] for z in shards] + [syn["fb"]])
    np.savez(OUT / "fb_raw.npz", keys=keys, fb=feats)
    for z in shards:
        have_clip.update(dict(zip(z["clip_paths"].astype(str), z["clip"])))
        have_cf.update(dict(zip(z["cf_paths"].astype(str), z["cf"])))
    np.savez(OUT / "clip_v1all.npz", paths=np.array(v1_paths), coords=np.stack([have_clip[p] for p in v1_paths]))
    pairs["cf_in"] = pairs.input_path.map(have_cf)
    pairs["cf_ex"] = pairs.expert_path.map(have_cf)
    pairs["d_cf"] = pairs.cf_ex - pairs.cf_in
    pairs["subgroup"] = np.where(pairs.bw, "bw", np.where(pairs.d_cf > 0, "vivid", "muted"))
    pairs.to_parquet(OUT / "v1all_pairs.parquet", index=False)
    print("V1 全部版本子组：", pairs.subgroup.value_counts().to_dict(), "；FB 特征", feats.shape, flush=True)


if __name__ == "__main__":
    main()
