#!/usr/bin/env python3
"""build_pairs.py — 备齐 500 个「原图 vs version_0」图对，两侧统一成同一规格的 jpg
================================================================================

图对来源：`7_baselines/Analysis_votes.zip` 里第二轮人工投票的 manifest.json（500 个唯一 ID，
全部在 scraped_data_v4/test）。

**两侧必须同规格**，否则分辨率 / 压缩痕迹本身就会暴露哪张是原图：
  规格 = RGB、长边缩到 1024（LANCZOS，不放大）、JPEG quality=95
  —— 与 `7_baselines/runs/A/_inputs_jpg95` 的生成方式逐步相同（run_a.py 缩放 + score_aes_a.py 另存）。

  · 原图      优先从 `_inputs_jpg95/<split>/<id>.jpg` 直接复制；那里没有的（不在条件表里的组）
              用 scraped_data_v4/test/<id>/original_captured_cropped.png 按同一规格生成
  · version_0 一律用 scraped_data_v4/test/<id>/version_0/main_image.png 按同一规格生成

自检：对复制来的原图随机抽若干张，用本脚本重新生成一遍并逐像素比对，确认「复制的」与
「本脚本生成的」确实是同一规格。

产物（默认 eval_results/human500/，该目录不入库）：
  images/<id>/original.jpg、images/<id>/version_0.jpg
  pairs.json   500 条：展示顺序、哪张放 A 位（固定种子、250/250 均衡）、尺寸、构图是否不一致

用法（仓库根目录）：
  python 3_llm_eval/human500/build_pairs.py
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
LONG_SIDE = 1024
JPG_QUALITY = 95
SEED = 20261112
MANIFEST_IN_ZIP = "Analysis_votes/manifest.json"


def to_spec(src: Path) -> Image.Image:
    """读图 → RGB → 长边缩到 1024（不放大）。与 run_a.py 的缩放写法一致。"""
    im = Image.open(src).convert("RGB")
    k = min(1.0, LONG_SIDE / max(im.size))
    if k < 1.0:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return im


def save_jpg(im: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=JPG_QUALITY)


def jpg_pixels(im: Image.Image) -> np.ndarray:
    """按规格编码成 jpg 再解码，得到最终像素（用于自检比对）。"""
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPG_QUALITY)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser(description="备齐 500 个图对并统一成 jpg95")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="仓库根目录（数据所在的那份检出）")
    ap.add_argument("--out", type=Path, default=None, help="输出目录，默认 <repo>/eval_results/human500")
    ap.add_argument("--verify", type=int, default=40, help="自检抽查的张数")
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = (args.out or repo / "eval_results" / "human500").resolve()

    with zipfile.ZipFile(repo / "7_baselines" / "Analysis_votes.zip") as z:
        manifest = json.loads(z.read(MANIFEST_IN_ZIP).decode("utf-8"))
    ids = sorted({x["sample_id"] for x in manifest})
    assert len(ids) == 500, f"期望 500 个唯一 ID，实际 {len(ids)}"

    jdir = repo / "7_baselines" / "runs" / "A" / "_inputs_jpg95"
    scraped = repo / "scraped_data_v4" / "test"
    copied, generated, entries = [], [], []

    for i, sid in enumerate(ids):
        src_orig = scraped / sid / "original_captured_cropped.png"
        src_v0 = scraped / sid / "version_0" / "main_image.png"
        assert src_orig.exists() and src_v0.exists(), f"缺源图：{sid}"
        dst_orig = out / "images" / sid / "original.jpg"
        dst_v0 = out / "images" / sid / "version_0.jpg"

        # 原图：有现成的 jpg95 就复制，没有就按同规格生成
        hit = next((p for p in sorted(jdir.glob(f"*/{sid}.jpg"))), None)
        if hit is not None:
            dst_orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hit, dst_orig)
            copied.append(sid)
            orig_from = str(hit.relative_to(repo))
        else:
            save_jpg(to_spec(src_orig), dst_orig)
            generated.append(sid)
            orig_from = str(src_orig.relative_to(repo))

        save_jpg(to_spec(src_v0), dst_v0)

        wo, ho = Image.open(dst_orig).size
        wv, hv = Image.open(dst_v0).size
        entries.append({
            "sample_id": sid,
            "original": {"file": f"images/{sid}/original.jpg", "size": [wo, ho], "from": orig_from},
            "version_0": {"file": f"images/{sid}/version_0.jpg", "size": [wv, hv],
                          "from": str(src_v0.relative_to(repo))},
            # 专家另外裁剪过的图对：两张图构图不同，结果里可以分开统计
            "aspect_mismatch": abs(wo / ho - wv / hv) > 0.01 * (wo / ho),
        })
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/500")

    # 自检：复制来的原图 == 本脚本按规格生成的结果？
    rng = random.Random(SEED)
    sample = rng.sample(copied, min(args.verify, len(copied)))
    identical, max_diff = 0, 0
    for sid in sample:
        a = np.asarray(Image.open(out / "images" / sid / "original.jpg").convert("RGB"))
        b = jpg_pixels(to_spec(scraped / sid / "original_captured_cropped.png"))
        if a.shape == b.shape:
            d = int(np.abs(a.astype(int) - b.astype(int)).max())
            max_diff = max(max_diff, d)
            identical += d == 0
        else:
            max_diff = 255
    print(f"自检：抽 {len(sample)} 张复制来的原图，与本脚本重新生成的逐像素完全一致 {identical} 张，"
          f"最大像素差 {max_diff}")

    # 展示顺序与 A/B 位置：固定种子，原图放 A 位恰好 250 对
    order = list(range(500))
    rng.shuffle(order)
    sides = ["original"] * 250 + ["version_0"] * 250
    rng.shuffle(sides)
    pairs = []
    for k, (idx, a_is) in enumerate(zip(order, sides)):
        e = dict(entries[idx])
        e.update({"k": k, "pair_id": f"{e['sample_id']}__original_vs_version_0", "a_is": a_is})
        pairs.append(e)

    meta = {
        "spec": {"long_side": LONG_SIDE, "resample": "LANCZOS", "format": "JPEG", "quality": JPG_QUALITY},
        "seed": SEED,
        "source_manifest": f"7_baselines/Analysis_votes.zip::{MANIFEST_IN_ZIP}",
        "n_pairs": len(pairs),
        "n_original_copied": len(copied),
        "n_original_generated": len(generated),
        "n_aspect_mismatch": sum(p["aspect_mismatch"] for p in pairs),
        "verify": {"checked": len(sample), "pixel_identical": identical, "max_abs_diff": max_diff},
    }
    (out / "pairs.json").write_text(json.dumps({"meta": meta, "pairs": pairs}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    (out / "sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"已写出：{out}/pairs.json、sample_ids.txt、images/（{2 * len(pairs)} 张）")


if __name__ == "__main__":
    main()
