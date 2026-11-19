#!/usr/bin/env python3
"""manifest_full.py — 全数据集评测清单：3,081 组 × 全部臂（与编码器无关）
================================================================================

每行一张图：family（fit_annotations 的组键）、uuid、split、category（10 类，缺失记「其他」）、
cls（对照/A/B/C/D/专家）、arm、path、version（专家版本号，其余为空）。

  原图        runs/A/_inputs/<split>/<uuid>.png        —— 专家终图（png）的同格式基准
  原图_jpg95  runs/A/_inputs_jpg95/<split>/<uuid>.jpg  —— 各模型输出（q95 jpg）的同格式基准
  A / B / C   runs/<目录>/<模型>/<split>/<uuid>.jpg    （B/C 失败任务缺图，记 exists=False，不补）
  D           runs/LR_PS/<工具>/<split>/<uuid>.jpg      （上传的 PNG 已按 q95 另存）
  专家        scraped_data_v4/<旧划分>/<uuid>/version_N/main_image.png，版本取 conditions.jsonl 收录的版本

用法：python 6_aes_eval/batch_retouch/manifest_full.py   → <bg>/retouch_v1/full_manifest.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batchq as bq  # noqa: E402

REPO = bq.REPO_DEFAULT
RUNS = REPO / "7_baselines" / "runs"
OUT = bq.bg_dir(REPO) / "retouch_v1"

A_MODELS = ["DeepLPF", "AdaInt", "CLIPtone_Beautiful", "CLIPtone_Professional", "DiffRetouch", "VeraRetouch"]
B_DIR, C_DIR = RUNS / "B_821e477950", RUNS / "C_d08a24a148"
D_TOOLS = {"lightroom_auto": "Lightroom自动", "photoshop_auto_tone": "PS自动色调"}


def short(model_dir: str) -> str:
    return model_dir.split("__", 1)[1] if "__" in model_dir else model_dir


def main():
    ann = json.loads((REPO / "fit_annotations.json").read_text())
    key_of = {k.split("/", 1)[1]: k for k in ann}
    versions = {}
    for line in (REPO / "sft_data_v2/set_sft/conditions.jsonl").read_text().splitlines():
        c = json.loads(line)
        versions.setdefault(c["source_group"].split("/")[1], set()).update(str(v["version"]) for v in c["versions"])
    rows = []
    for p in sorted((RUNS / "A" / "_inputs_jpg95").glob("*/*.jpg")):
        split, uuid = p.parent.name, p.stem
        fam = key_of[uuid]
        base = {"family": fam, "uuid": uuid, "split": split, "category": ann[fam].get("category") or "其他"}

        def add(cls, arm, path, version=None):
            rows.append({**base, "cls": cls, "arm": arm, "path": str(path), "version": version, "exists": Path(path).exists()})

        add("对照", "原图", RUNS / "A" / "_inputs" / split / f"{uuid}.png")
        add("对照", "原图_jpg95", p)
        for m in A_MODELS:
            add("A", m, RUNS / "A" / m / split / f"{uuid}.jpg")
        for d in sorted(x.name for x in B_DIR.iterdir() if x.is_dir()):
            add("B", short(d), B_DIR / d / split / f"{uuid}.jpg")
        for d in sorted(x.name for x in C_DIR.iterdir() if x.is_dir()):
            add("C", short(d), C_DIR / d / split / f"{uuid}.jpg")
        for tool, arm in D_TOOLS.items():
            add("D", arm, RUNS / "LR_PS" / tool / split / f"{uuid}.jpg")
        for v in sorted(versions.get(uuid, ()), key=int):
            ex = next((q for sp in ("train", "test")
                       if (q := REPO / "scraped_data_v4" / sp / uuid / f"version_{v}" / "main_image.png").exists()), None)
            if ex is not None:
                add("专家", "专家", ex, v)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "full_manifest.parquet", index=False)
    cov = df.groupby(["cls", "arm"]).agg(images=("exists", "sum"), groups=("family", "nunique")).reset_index()
    print(f"组数 {df.family.nunique()}；图片 {int(df.exists.sum())}（缺 {int((~df.exists).sum())}）")
    print(cov.to_string(index=False))
    print(df.drop_duplicates("family").category.value_counts().to_dict())


if __name__ == "__main__":
    main()
