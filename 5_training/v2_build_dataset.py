#!/usr/bin/env python3
"""v2_build_dataset.py — SFT-v2 数据构建（工作分解 C）
====================================================

产出两门课的 ShareGPT 样本（token 输出格式，非 JSON）：

  品味课（taste，学「什么」）：复用 v1 已在磁盘的 s1..s4 中间图，只把标签重编码为
      token 序列、换 prompt。复用 build_sft_dataset.collect_jobs → 质量门 / 白名单 /
      val 钉死全部继承，与 v1 逐 version 对齐，零重渲。

  机制课（mech，学「怎么」）：合成逆向任务。train 非 val 原图 × 随机裁剪 →
      bin 空间均匀采参（70% 组合 / 30% 单参隔离，color 阶段 10% 黑白）→ 渲「改后」图 →
      (改前, 改后) → 该 stage 的 token 标签。喂饱 63 桶尾部、教「参数→效果」。
      解码标签→渲染 ≡ 存盘改后图（門①自检）；不带 category（物理与题材解耦）。

诊断课（diag，学「为何」）由 v2_diagnosis_gen.py 单独产出（依赖 OpenRouter）。

用法（仓库根目录，Chen 环境）：
  python 5_training/v2_build_dataset.py                 # 品味课 + 机制课
  python 5_training/v2_build_dataset.py --only taste    # 只重建品味课（快）
  python 5_training/v2_build_dataset.py --mech_target 30000 --selfcheck 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))

import cv2  # noqa: E402
from differentiable_renderer.opencv_renderer import render  # noqa: E402
from build_sft_dataset import (collect_jobs, extract_stages,  # noqa: E402
                               _imread_rgb, _resize_max_side, _save_jpg)
from v2_codec import (STAGE_NAMES, STAGE_SLOTS, BIN_GROUPS,  # noqa: E402
                      slot_nbins, slot_neutral, slot_vtok, decode_stage)


def _save_jpg444(path: Path, rgb: np.ndarray, quality: int = 92):
    """存机制课图用 4:4:4 无色度子采样 —— color 阶段高饱和色移在 4:2:0 下的色度误差
    实测 max 6.5→3.1（减半），代价 +57% 体积；不给 color 物理课再叠一层 JPEG 噪声。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality,
                 cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])

# ============================================================
# v2 prompts（token 输出；与 v1 JSON 版本刻意区分）
# ============================================================
SYSTEM_PROMPT_V2 = (
    "You are a professional photo retouching agent. You edit a photo in four "
    "sequential stages: light, tone, color, detail. For the requested stage you "
    "output parameter tokens — each parameter-name token is immediately followed "
    "by one value token: <ev_N> for exposure (N from 0 to 126; 63 = no change) "
    "and <v_N> for all other parameters (N from 0 to 62; 31 = no change). "
    "Lower = decrease, higher = increase. Output only the parameter tokens, "
    "nothing else."
)

STAGE_DESC = {
    "light":  "Stage 1/4 (light): global exposure and white balance.",
    "tone":   "Stage 2/4 (tone): contrast, highlights, shadows, whites, blacks, tone curve, dehaze.",
    "color":  "Stage 3/4 (color): vibrance, saturation, per-hue color mixer, color grading.",
    "detail": "Stage 4/4 (detail): texture, clarity, sharpening, noise reduction, vignette.",
}

TASTE_USER = ("<image>\nCategory: {cat}. Black & white target: {bw}.\n{desc}\n"
              "Output the parameter tokens for this stage.")

MECH_USER = ("<image><image>\nIMAGE 1 is the original photo; IMAGE 2 is the same "
             "photo after an edit that changed ONLY the {stage} controls "
             "({names}). Black & white target: {bw}.\n"
             "Output the {stage} parameter tokens that produced IMAGE 2 from IMAGE 1.")

STAGE_PARAM_NAMES = {
    "light":  "exposure, temperature, tint",
    "tone":   "contrast, highlights/shadows, whites/blacks, tone curve, dehaze",
    "color":  "vibrance, saturation, color mixer, color grading",
    "detail": "texture, clarity, sharpening, noise reduction, vignette",
}


def _tokens_for_stage(stage: str, bins: dict[str, int]) -> str:
    """{slot_token: bin} → 键值相间 token 串（按 STAGE_SLOTS 固定顺序，值 token 按桶组）。"""
    out = []
    for slot in STAGE_SLOTS[stage]:
        out += [slot.token, slot_vtok(slot, bins[slot.token])]
    return "".join(out)


# ============================================================
# 品味课：复用 v1 中间图 + 重编码标签
# ============================================================
def build_taste(out: Path, splits, max_final_loss, max_render_diff, limit):
    args = SimpleNamespace(
        out=out, splits=splits, max_side=1024, jpeg_q=92,
        max_final_loss=max_final_loss, max_render_diff=max_render_diff,
        consistency_file=out / "render_consistency.json",
        loss_whitelist=out / "loss_whitelist.txt", limit=limit)
    jobs, report = collect_jobs(args)
    print(f"[taste] collect_jobs: {report['versions']} version"
          f"（loss过滤 {report['loss_filtered']} / 白名单 {report['loss_whitelisted']}）")

    # val 钉死沿用（与 v1 同一份，保证跨版本可比）
    pin = json.loads((out / "val_groups.json").read_text())["groups"]
    val_gids = set(pin)
    print(f"[taste] 沿用 val_groups.json：{len(val_gids)} 组")

    from v2_codec import encode_stage
    buckets = {"train": [], "val": [], "test": []}
    n_missing = 0
    for j in jobs:
        key, vn = j["key"], j["vn"]
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        params = json.loads((vdir / "params.json").read_text())
        stages = extract_stages(params)
        bucket = "test" if j["split"] == "test" else ("val" if key in val_gids else "train")
        for i, sname in enumerate(STAGE_NAMES):
            img_rel = f"images/{key}/v{vn}/s{i + 1}.jpg"
            if not (out / img_rel).exists():
                n_missing += 1
                continue
            label = "".join(encode_stage(sname, stages[i][1]))
            buckets[bucket].append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": TASTE_USER.format(
                        cat=j["category"], bw="yes" if j["bw"] else "no",
                        desc=STAGE_DESC[sname])},
                    {"role": "assistant", "content": label},
                ],
                "images": [img_rel],
                "_meta": {"course": "taste", "group": key, "version": str(vn),
                          "stage": sname, "category": j["category"], "bw": j["bw"]},
            })
    random.Random(42).shuffle(buckets["train"])
    for split, data in buckets.items():
        (out / f"task_v2_taste_{split}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1))
        print(f"[taste] task_v2_taste_{split}.json: {len(data)} 样本")
    if n_missing:
        print(f"[taste] ⚠ {n_missing} 个 stage 图缺失（v1 中间图未生成？先跑 build_sft_dataset.py）")
    return {k: len(v) for k, v in buckets.items()}


# ============================================================
# 机制课：合成逆向任务
# ============================================================
def _augment_crop(img: np.ndarray, rng: np.random.Generator, max_side=1024) -> np.ndarray:
    """温和裁剪(0.85–1.0,躲开切碎主体/切进平坦区的退化) + 50% 水平翻转
    (保标签、不产离分布图、免费加倍;所有渲染参数水平对称,翻转不损物理课信号)。"""
    h, w = img.shape[:2]
    s = rng.uniform(0.85, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0 = rng.integers(0, h - ch + 1)
    x0 = rng.integers(0, w - cw + 1)
    crop = img[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:                       # 水平翻转
        crop = crop[:, ::-1]
    return _resize_max_side(np.ascontiguousarray(crop), max_side)


def _sample_bin(slot, rng: np.random.Generator, isolated: bool) -> int:
    """单向参数只用上半轴；隔离模式排除近中性 ±3 保证可见。桶数按 slot 桶组。"""
    nb, neu = slot_nbins(slot), slot_neutral(slot)
    lo_b = neu if slot.one_sided else 0
    if not isolated:
        return int(rng.integers(lo_b, nb))
    while True:
        b = int(rng.integers(lo_b, nb))
        if abs(b - neu) > 3:
            return b


def _mech_worker(task: dict) -> dict:
    """一个 (原图, 裁剪) → 1 张 before + S 张 after + S 条样本。"""
    try:
        rng = np.random.default_rng(task["seed"])
        out = Path(task["out"])
        img = _imread_rgb(task["src"])
        before = _augment_crop(img, rng)
        odir = out / "images_mech" / task["oid"] / f"c{task['crop']}"
        before_rel = f"images_mech/{task['oid']}/c{task['crop']}/before.jpg"
        _save_jpg444(out / before_rel, before)

        samples, cover = [], []
        # stage 采样加权：light 提到 0.4（曝光专属 127 桶信号稀，机制课定向增采喂密；
        # 其余三 stage 均分 0.2）
        stage_p = [0.4, 0.2, 0.2, 0.2]
        for j in range(task["n_samples"]):
            stage = STAGE_NAMES[int(rng.choice(4, p=stage_p))]
            slots = STAGE_SLOTS[stage]
            isolated = rng.random() < 0.30
            bins = {s.token: slot_neutral(s) for s in slots}   # 中性桶按桶组
            if isolated:
                pick = slots[int(rng.integers(0, len(slots)))]
                bins[pick.token] = _sample_bin(pick, rng, isolated=True)
            else:
                for s in slots:
                    bins[s.token] = _sample_bin(s, rng, isolated=False)
            bw = False
            if stage == "color" and rng.random() < 0.10:      # 黑白 regime
                bins["<saturation>"] = 0                       # bin0 → -100
                bins["<vibrance>"] = 0
                bw = True
            cover.extend((s.group, bins[s.token]) for s in slots)  # (桶组, bin)

            token_str = _tokens_for_stage(stage, bins)
            _, zh = decode_stage(stage, [t for s in slots
                                         for t in (s.token, slot_vtok(s, bins[s.token]))])
            after = render(before, zh)
            after_rel = f"images_mech/{task['oid']}/c{task['crop']}/m{j}_after.jpg"
            _save_jpg444(out / after_rel, after)
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": MECH_USER.format(
                        stage=stage, names=STAGE_PARAM_NAMES[stage],
                        bw="yes" if bw else "no")},
                    {"role": "assistant", "content": token_str},
                ],
                "images": [before_rel, after_rel],
                "_meta": {"course": "mech", "stage": stage, "bw": bw,
                          "isolated": isolated, "src": task["oid"]},
            })
        return {"ok": True, "samples": samples, "cover": cover}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"ok": False, "err": traceback.format_exc(limit=2), "src": task.get("oid")}


def build_mech(out: Path, target: int, crops_per_orig: int, seed: int, workers: int):
    val_gids = set(json.loads((out / "val_groups.json").read_text())["groups"])
    origs = []
    for gdir in sorted((REPO_ROOT / "fit_results" / "train").iterdir()):
        if not gdir.is_dir():
            continue
        key = f"train/{gdir.name}"
        if key in val_gids:                          # 排除 val 组，防泄漏
            continue
        p = gdir / "original_captured_cropped.png"
        if p.exists():
            origs.append((gdir.name, str(p)))
    n_units = len(origs) * crops_per_orig
    n_samples = max(1, round(target / n_units))
    print(f"[mech] 原图 {len(origs)} 张（已排除 {len(val_gids)} val 组）× {crops_per_orig} 裁剪 "
          f"× {n_samples} 样本/裁剪 ≈ {n_units * n_samples} 条")

    rng = np.random.default_rng(seed)
    tasks = []
    for oid, src in origs:
        for c in range(crops_per_orig):
            tasks.append({"oid": oid, "src": src, "crop": c, "n_samples": n_samples,
                          "seed": int(rng.integers(0, 2**31)), "out": str(out)})

    t0, all_samples, cover, errs = time.time(), [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_mech_worker, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["ok"]:
                all_samples.extend(r["samples"])
                cover.extend(r["cover"])
            else:
                errs.append(r)
            if n % 200 == 0 or n == len(tasks):
                rate = n / (time.time() - t0)
                print(f"  [{n}/{len(tasks)}] {rate:.1f} unit/s  "
                      f"eta {(len(tasks) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)

    random.Random(seed).shuffle(all_samples)
    (out / "task_v2_mech_train.json").write_text(
        json.dumps(all_samples, ensure_ascii=False, indent=1))
    print(f"[mech] task_v2_mech_train.json: {len(all_samples)} 样本  错误 {len(errs)}")

    # 門②：bin 覆盖直方图（按桶组分别报，应近均匀、尾部无空桶）
    hists = {}
    for g, nb in BIN_GROUPS.items():
        gb = [b for gg, b in cover if gg == g]
        h = np.bincount(np.array(gb), minlength=nb) if gb else np.zeros(nb, int)
        hists[g] = h.tolist()
        print(f"[mech] 桶组 {g}({nb}桶): {(h > 0).sum()}/{nb} 非空  "
              f"最稀 {int(h.min())}  最密 {int(h.max())}  中位 {int(np.median(h))}")
    if errs:
        print(f"[mech] ⚠ 首个错误: {errs[0]['err'][:200]}")
    return len(all_samples), hists


# ============================================================
# 自检 門①：机制课解码标签渲染 ≡ 存盘 after
# ============================================================
def selfcheck_mech(out: Path, n: int):
    data = json.loads((out / "task_v2_mech_train.json").read_text())
    rng = random.Random(0)
    diffs = []
    for s in rng.sample(data, min(n, len(data))):
        before = _imread_rgb(str(out / s["images"][0]))
        after = _imread_rgb(str(out / s["images"][1])).astype(np.float64)
        stage = s["_meta"]["stage"]
        toks = s["messages"][2]["content"]
        # 解析 token 串
        import re
        seq = re.findall(r"<[^>]+>", toks)
        _, zh = decode_stage(stage, seq)
        re_render = render(before, zh).astype(np.float64)
        diffs.append(float(np.abs(re_render - after).mean()))
    d = np.array(diffs)
    print(f"[selfcheck門①] n={len(d)}  解码渲染 vs 存盘after |Δ|: "
          f"p50={np.percentile(d, 50):.3f} p95={np.percentile(d, 95):.3f} max={d.max():.3f} /255")
    # 标签由构造自洽（after 就是 decode(label)→zh 渲的）；此 diff 是 before/after 两次
    # JPEG round-trip 的底噪，与 v1 selfcheck「<6 算正常」同类。用稳健 p95 设门，max
    # 的离群由强锐化 detail 样本在高频内容上的 JPEG 压缩造成、无害。
    assert np.percentile(d, 95) < 6.0, f"机制课标签自洽性异常（p95={np.percentile(d, 95):.2f}）"
    print("  ✅ 標籤自洽（diff = JPEG 底噪，p95 < 6）")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "sft_data_v2")
    ap.add_argument("--only", choices=["taste", "mech"], default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--max_final_loss", type=float, default=8.0)
    ap.add_argument("--max_render_diff", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mech_target", type=int, default=30000)
    ap.add_argument("--crops_per_orig", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--selfcheck", type=int, default=200)
    args = ap.parse_args()

    report = {}
    if args.only != "mech":
        report["taste"] = build_taste(args.out, args.splits, args.max_final_loss,
                                       args.max_render_diff, args.limit)
    if args.only != "taste":
        n, hists = build_mech(args.out, args.mech_target, args.crops_per_orig,
                              args.seed, args.workers)
        report["mech"] = {"n": n, "bin_hist": hists}
        if args.selfcheck:
            selfcheck_mech(args.out, args.selfcheck)

    (args.out / "build_v2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("[done] 见 build_v2_report.json")


if __name__ == "__main__":
    main()
