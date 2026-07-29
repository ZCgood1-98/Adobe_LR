#!/usr/bin/env python3
"""
build_sft_dataset.py — 线5 · 任务A「逐维直出」SFT 数据构建
==========================================================

按 TRAINING_PLAN「★ 设计修订(2026-06-23/07-04/07-12)」构建 SFT 训练集：
  数据源   : fit_results/{split}/{gid}/version_N/params.json（拟合参数当 GT）
             + fit_annotations.json（类别 / 黑白 / keep-discard）
  样本形态 : (当前图 + 类别 + 黑白flag + stage指令) → 该 stage 参数 JSON（无 CoT 直出）
  阶段划分 : 4 阶段（light → tone → color → detail），顺序=plan 的感知杠杆优先级
  中间图   : 用「已决定阶段的累积参数」从原图一次性渲染（渲染器管线顺序固定，
             参数子集渲染即得一致中间态；不做链式渲染，避免 detail_transfer 重复叠加）
  输入约定 : version 目录有 original_cropped.png 就用它，否则组级 original_captured_cropped.png

产出（--out 目录，默认 sft_data_v2/）：
  images/{split}/{gid}/v{N}/s1.jpg..s4.jpg   各 stage 的输入图（s1=原图缩放）
  task_a_train.json / task_a_val.json / task_a_test.json   ShareGPT 多模态样本
  dataset_info.json                          LLaMA-Factory 数据集注册
  build_report.json                          构建统计与跳过明细

用法（仓库根目录运行）：
  python 5_training/build_sft_dataset.py                    # 全量
  python 5_training/build_sft_dataset.py --limit 6          # 冒烟：只取前 6 组
  python 5_training/build_sft_dataset.py --selfcheck 20     # 抽 20 个 version 验证
                                                            # 累积渲染 ≈ rendered.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from differentiable_renderer.opencv_renderer import render  # noqa: E402

# ============================================================
# 阶段定义（v1 拍板：4 阶段，顺序=TRAINING_PLAN 四·优先级表）
# 改分组只需动这里 + STAGE_PROMPTS，构建器其余部分与分组无关
# ============================================================

STAGE_NAMES = ["light", "tone", "color", "detail"]

MIXER_CHANNELS = ["Red", "Orange", "Yellow", "Green",
                  "Aqua", "Blue", "Purple", "Magenta"]
CG_ZONES = ["shadow", "midtone", "highlight"]

CATEGORY_EN = {
    "人物": "portrait", "风景": "landscape", "都市": "urban",
    "动物": "animal", "食物": "food", "植物": "plant", "其他": "other",
}

SYSTEM_PROMPT = """You are a professional photo retouching agent. You edit a photo in 4 sequential stages of a Lightroom-style pipeline:
Stage 1 (light): exposure and white balance
Stage 2 (tone): contrast, highlights/shadows, whites/blacks, tone curve, dehaze
Stage 3 (color): vibrance, saturation, per-hue color mixer, color grading
Stage 4 (detail): texture, clarity, sharpening, noise reduction, vignette
Each turn you see the CURRENT image (earlier stages already applied), the subject category, and whether the target style is black & white. Decide only the requested stage's parameters. Output a single JSON object and nothing else. Use 0 for parameters that need no change."""

STAGE_PROMPTS = {
    "light": ("Stage 1/4 (light): decide global exposure and white balance for this photo.\n"
              "Output JSON keys: exposure (EV, [-5.00, 5.00]), "
              "temperature (target white point in Kelvin, [2000, 12000], 5500 = as shot), "
              "tint ([-150.0, 150.0])."),
    "tone":  ("Stage 2/4 (tone): shape the tonal response of this photo.\n"
              "Output JSON keys: contrast, highlights, shadows, whites, blacks (each [-100, 100]), "
              "tone_curve (8 curve-point deltas from dark to bright, each [-100, 100]), "
              "dehaze ([-100, 100])."),
    "color": ("Stage 3/4 (color): adjust global and per-hue color of this photo.\n"
              "Output JSON keys: vibrance, saturation (each [-100, 100]; saturation -100 = fully "
              "desaturated for black & white), color_mixer ({red, orange, yellow, green, aqua, "
              "blue, purple, magenta}, each with h/s/l in [-100, 100]), "
              "color_grading ({shadow, midtone, highlight}, each with r/g/b in [-100, 100])."),
    "detail": ("Stage 4/4 (detail): finishing touches for this photo.\n"
               "Output JSON keys: texture ([0, 100]), clarity ([0, 100]), sharpen ([0, 150]), "
               "noise_reduction ([0, 100]), vignette ([-150, 150])."),
}

USER_TEMPLATE = ("<image>\nCategory: {category}. Black & white target: {bw}.\n"
                 "{stage_prompt}\nOutput only the JSON object.")


def _r(v, nd: int = 1) -> float:
    """按输出精度舍入；渲染与 JSON 标签用同一份舍入值，保证二者一致。"""
    out = round(float(v), nd)
    return 0.0 if out == 0 else out  # 规避 -0.0


def _r_pos(v, nd: int = 1) -> float:
    """非负参数：拟合的微小负值（如 噪点消除 -0.1）按 0 处理。"""
    return max(0.0, _r(v, nd))


def extract_stages(p: dict) -> list[tuple[dict, dict]]:
    """params.json → [(渲染用中文参数子集, 英文标签 JSON), ...] × 4 阶段。

    两份 dict 的数值完全相同（同一舍入），中文键喂 opencv_renderer.render，
    英文键做模型输出标签；标签键序固定=语义序（不排序）。
    """
    # ---- Stage 1 light ----
    zh1 = {"曝光度": _r(p.get("曝光度", 0), 2),
           "色温": float(int(round(p.get("色温", 5500)))),
           "色调": _r(p.get("色调", 0))}
    en1 = {"exposure": zh1["曝光度"], "temperature": int(zh1["色温"]), "tint": zh1["色调"]}

    # ---- Stage 2 tone ----
    tc = [_r(v) for v in (p.get("tone_curve_dy") or [0.0] * 8)]
    zh2 = {"对比度": _r(p.get("对比度", 0)), "高光": _r(p.get("高光", 0)),
           "阴影": _r(p.get("阴影", 0)), "白色色阶": _r(p.get("白色色阶", 0)),
           "黑色色阶": _r(p.get("黑色色阶", 0)), "tone_curve_dy": tc,
           "去朦胧": _r(p.get("去朦胧", 0))}
    en2 = {"contrast": zh2["对比度"], "highlights": zh2["高光"], "shadows": zh2["阴影"],
           "whites": zh2["白色色阶"], "blacks": zh2["黑色色阶"],
           "tone_curve": tc, "dehaze": zh2["去朦胧"]}

    # ---- Stage 3 color ----
    raw_mixer = {e.get("channel"): e for e in (p.get("color_mixer") or [])
                 if isinstance(e, dict)}
    mix_zh, mix_en = [], {}
    for ch in MIXER_CHANNELS:
        e = raw_mixer.get(ch, {})
        h, s, l = _r(e.get("H", 0)), _r(e.get("S", 0)), _r(e.get("L", 0))
        mix_zh.append({"channel": ch, "H": h, "S": s, "L": l})
        mix_en[ch.lower()] = {"h": h, "s": s, "l": l}
    cg = p.get("颜色分级") or {}
    cg_zh, cg_en = {}, {}
    for z in CG_ZONES:
        d = cg.get(z) or {}
        r, g, b = _r(d.get("R", 0)), _r(d.get("G", 0)), _r(d.get("B", 0))
        cg_zh[z] = {"R": r, "G": g, "B": b}
        cg_en[z] = {"r": r, "g": g, "b": b}
    zh3 = {"自然饱和度": _r(p.get("自然饱和度", 0)), "饱和度": _r(p.get("饱和度", 0)),
           "color_mixer": mix_zh, "颜色分级": cg_zh}
    en3 = {"vibrance": zh3["自然饱和度"], "saturation": zh3["饱和度"],
           "color_mixer": mix_en, "color_grading": cg_en}

    # ---- Stage 4 detail ----
    zh4 = {"纹理": _r_pos(p.get("纹理", 0)), "清晰度": _r_pos(p.get("清晰度", 0)),
           "锐化": _r_pos(p.get("锐化", 0)), "噪点消除": _r_pos(p.get("噪点消除", 0)),
           "晕影": _r(p.get("晕影", 0))}
    en4 = {"texture": zh4["纹理"], "clarity": zh4["清晰度"], "sharpen": zh4["锐化"],
           "noise_reduction": zh4["噪点消除"], "vignette": zh4["晕影"]}

    return [(zh1, en1), (zh2, en2), (zh3, en3), (zh4, en4)]


# ============================================================
# 单 version 处理（进程池 worker）
# ============================================================

def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_LANCZOS4)


def _save_jpg(path: Path, rgb: np.ndarray, quality: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def process_version(job: dict) -> dict:
    """渲染并保存 4 张 stage 输入图，返回英文标签等组装素材。幂等：图已存在则跳过渲染。"""
    try:
        key, vn = job["key"], job["vn"]          # key = "train/<gid>"
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        params = json.loads((vdir / "params.json").read_text())
        stages = extract_stages(params)

        img_dir = Path(job["out_dir"]) / "images" / key / f"v{vn}"
        img_paths = [img_dir / f"s{i + 1}.jpg" for i in range(4)]

        in_png = vdir / "original_cropped.png"
        if not in_png.exists():
            in_png = REPO_ROOT / "fit_results" / key / "original_captured_cropped.png"

        # 过期检测：params.json 或输入图比已有中间图新（重拟合/几何救援之后），
        # 说明缓存是旧输入/旧参数渲的 → 强制全部重渲
        all_exist = all(p.exists() for p in img_paths)
        stale = False
        if all_exist:
            src_mtime = max((vdir / "params.json").stat().st_mtime,
                            in_png.stat().st_mtime)
            stale = min(p.stat().st_mtime for p in img_paths) < src_mtime
        need_render = (not all_exist) or stale

        selfcheck_diff = None
        if need_render or job.get("selfcheck"):
            img0 = _resize_max_side(_imread_rgb(str(in_png)), job["max_side"])

            if stale or not img_paths[0].exists():
                _save_jpg(img_paths[0], img0, job["jpeg_q"])
            cum: dict = {}
            for i in range(3):                    # s2..s4 = 累积到 stage1..3 的渲染
                cum.update(stages[i][0])
                if stale or not img_paths[i + 1].exists():
                    _save_jpg(img_paths[i + 1], render(img0, dict(cum)), job["jpeg_q"])

            if job.get("selfcheck"):              # 全 4 阶段累积 ≈ 拟合的 rendered.png
                cum.update(stages[3][0])
                full = render(img0, dict(cum)).astype(np.float64)
                ref = _imread_rgb(str(vdir / "rendered.png"))
                ref = cv2.resize(ref, (full.shape[1], full.shape[0]),
                                 interpolation=cv2.INTER_AREA).astype(np.float64)
                selfcheck_diff = float(np.abs(full - ref).mean())

        rel = [str(p.relative_to(job["out_dir"])) for p in img_paths]
        return {"ok": True, "key": key, "vn": vn, "images": rel,
                "labels": [json.dumps(en, ensure_ascii=False) for _, en in stages],
                "selfcheck_diff": selfcheck_diff}
    except Exception:
        return {"ok": False, "key": job.get("key"), "vn": job.get("vn"),
                "err": traceback.format_exc(limit=3)}


# ============================================================
# 主流程
# ============================================================

def collect_jobs(args) -> tuple[list[dict], dict]:
    """按标注规则收集有效 (组, version)，套 final_loss 质量门。"""
    ann = json.loads((REPO_ROOT / "fit_annotations.json").read_text())
    jobs, report = [], {"skipped_group": {}, "skipped_version": {}, "loss_filtered": 0,
                        "loss_whitelisted": 0}

    # 渲染一致性过滤（scan_render_consistency.py 产出）：文件存在才生效
    cons, max_rd = {}, float(getattr(args, "max_render_diff", 0) or 0)
    cons_file = getattr(args, "consistency_file", None)
    if max_rd > 0 and cons_file and Path(cons_file).exists():
        cons = json.loads(Path(cons_file).read_text())["diffs"]
        print(f"[collect] 渲染一致性过滤生效: |Δ|>{max_rd} 弃用（{len(cons)} 条记录）")

    # loss 白名单（人工审美背书的 version）：final_loss 超门槛仍保留，文件存在才生效
    wl, wl_file = set(), getattr(args, "loss_whitelist", None)
    if wl_file and Path(wl_file).exists():
        wl = {ln.strip() for ln in Path(wl_file).read_text().splitlines()
              if ln.strip() and not ln.startswith("#")}
        print(f"[collect] loss 白名单生效: {len(wl)} 条（final_loss 门对其豁免，渲染一致性门照常）")

    def skip_g(reason):
        report["skipped_group"][reason] = report["skipped_group"].get(reason, 0) + 1

    def skip_v(reason):
        report["skipped_version"][reason] = report["skipped_version"].get(reason, 0) + 1

    n_groups = {}
    for key in sorted(ann.keys()):
        a = ann[key]
        split = key.split("/", 1)[0]
        if split not in args.splits:
            continue
        if args.limit and n_groups.get(split, 0) >= args.limit:
            continue
        if a.get("discarded"):
            skip_g("整组舍弃"); continue
        cat = a.get("category")
        if not cat:
            skip_g("无类别"); continue
        gdir = REPO_ROOT / "fit_results" / key
        if not (gdir / "original_captured_cropped.png").exists():
            skip_g("缺组级原图"); continue

        group_jobs = []
        for vn, vinfo in sorted(a.get("versions", {}).items(), key=lambda x: int(x[0])):
            if vinfo.get("decision") != "keep":
                skip_v("version舍弃"); continue
            vdir = gdir / f"version_{vn}"
            if not ((vdir / "params.json").exists() and (vdir / "fit_meta.json").exists()):
                skip_v("缺拟合文件"); continue
            if args.max_final_loss > 0:
                meta = json.loads((vdir / "fit_meta.json").read_text())
                if meta.get("final_loss", 0) > args.max_final_loss:
                    if f"{key}/v{vn}" in wl:
                        report["loss_whitelisted"] += 1  # 人工背书豁免
                    else:
                        report["loss_filtered"] += 1; continue
                if meta.get("degenerate"):
                    skip_v("degenerate"); continue
            d = cons.get(f"{key}/v{vn}")
            if max_rd > 0 and d is not None and d > max_rd:
                skip_v("渲染不一致"); continue
            group_jobs.append({
                "key": key, "vn": vn, "split": split,
                "category": CATEGORY_EN.get(cat, "other"), "bw": bool(vinfo.get("bw")),
                "out_dir": str(args.out), "max_side": args.max_side,
                "jpeg_q": args.jpeg_q,
            })
        if not group_jobs:
            skip_g("无可用version"); continue
        n_groups[split] = n_groups.get(split, 0) + 1
        jobs.extend(group_jobs)

    report["groups"] = n_groups
    report["versions"] = len(jobs)
    return jobs, report


def build_samples(job: dict, result: dict) -> list[dict]:
    """一个 version → 4 条 ShareGPT 样本。"""
    samples = []
    for i, sname in enumerate(STAGE_NAMES):
        user = USER_TEMPLATE.format(
            category=job["category"], bw="yes" if job["bw"] else "no",
            stage_prompt=STAGE_PROMPTS[sname])
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": result["labels"][i]},
            ],
            "images": [result["images"][i]],
            "_meta": {"group": job["key"], "version": str(job["vn"]),
                      "stage": sname, "category": job["category"],
                      "bw": job["bw"]},
        })
    return samples


DATASET_INFO = {
    name: {
        "file_name": f"task_a_{split}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant",
                 "system_tag": "system"},
    }
    for name, split in [("photo_edit_sft_train", "train"),
                        ("photo_edit_sft_val", "val"),
                        ("photo_edit_sft_test", "test")]
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "sft_data_v2")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--max_side", type=int, default=1024, help="stage 输入图长边")
    ap.add_argument("--jpeg_q", type=int, default=92)
    ap.add_argument("--max_final_loss", type=float, default=8.0,
                    help="拟合质量门：final_loss 超过则弃用该 version；0=关闭")
    ap.add_argument("--consistency_file", type=Path, default=None,
                    help="render_consistency.json 路径（默认 <out>/render_consistency.json，"
                         "文件存在才过滤）")
    ap.add_argument("--loss_whitelist", type=Path, default=None,
                    help="loss 门白名单（每行 <split>/<gid>/v<N>，# 开头为注释）：人工复审背书的 "
                         "version 豁免 final_loss 门（默认 <out>/loss_whitelist.txt，文件存在才生效）")
    ap.add_argument("--max_render_diff", type=float, default=8.0,
                    help="渲染一致性门：累积渲染 vs rendered.png 平均|Δ|超过则弃用；0=关闭")
    ap.add_argument("--val_frac", type=float, default=0.02,
                    help="从 train 按「组」切出的 val 比例（防同组跨 split 泄漏）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只取前 N 组（冒烟）")
    ap.add_argument("--selfcheck", type=int, default=0,
                    help="抽 N 个 version 校验 4 阶段累积渲染 ≈ 拟合 rendered.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.consistency_file is None:
        args.consistency_file = args.out / "render_consistency.json"
    if args.loss_whitelist is None:
        args.loss_whitelist = args.out / "loss_whitelist.txt"

    t0 = time.time()
    jobs, report = collect_jobs(args)
    print(f"[collect] 组: {report['groups']}  version: {report['versions']}  "
          f"loss门槛过滤: {report['loss_filtered']}  白名单豁免: {report['loss_whitelisted']}")
    print(f"[collect] 跳过组: {report['skipped_group']}  跳过version: {report['skipped_version']}")

    if args.selfcheck:
        rng = random.Random(args.seed)
        for j in rng.sample(jobs, min(args.selfcheck, len(jobs))):
            j["selfcheck"] = True

    args.out.mkdir(parents=True, exist_ok=True)
    results, errors, checks = {}, [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_version, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["ok"]:
                results[(r["key"], r["vn"])] = r
                if r.get("selfcheck_diff") is not None:
                    checks.append((r["key"], r["vn"], r["selfcheck_diff"]))
            else:
                errors.append(r)
            if n % 200 == 0 or n == len(jobs):
                rate = n / (time.time() - t0)
                print(f"  [{n}/{len(jobs)}] {rate:.1f} it/s  "
                      f"eta {(len(jobs) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)

    # ---- 组装样本，train 按组切 val ----
    # val 钉死：val_groups.json 已存在则沿用其组清单（数据集增量重建时 val 不漂移，
    # 保证训练开始后的跨阶段可比性）；想重新抽样就先删该文件
    rng = random.Random(args.seed)
    train_gids = sorted({j["key"] for j in jobs if j["split"] == "train"})
    pin_file = args.out / "val_groups.json"
    if pin_file.exists():
        pinned = set(json.loads(pin_file.read_text())["groups"])
        val_gids = pinned & set(train_gids)
        print(f"[val] 沿用 val_groups.json 钉死的 {len(val_gids)} 组"
              f"（清单 {len(pinned)} 组，缺席 {len(pinned) - len(val_gids)}）")
    else:
        val_gids = set(rng.sample(train_gids, int(len(train_gids) * args.val_frac)))

    buckets = {"train": [], "val": [], "test": []}
    for j in jobs:
        r = results.get((j["key"], j["vn"]))
        if not r:
            continue
        b = "test" if j["split"] == "test" else ("val" if j["key"] in val_gids else "train")
        buckets[b].extend(build_samples(j, r))
    rng.shuffle(buckets["train"])

    for split, data in buckets.items():
        path = args.out / f"task_a_{split}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(f"[write] {path.name}: {len(data)} 样本")

    # val 组清单：GRPO / 任务B / 线路2 的训练 prompt 必须同样排除这些组，
    # 保证 val 对"SFT vs 后续阶段"的比较始终干净
    val_used = sorted({s["_meta"]["group"] for s in buckets["val"]})
    (args.out / "val_groups.json").write_text(json.dumps(
        {"note": "SFT val 组（按组切分 seed42，经全部质量门后实际入 val 的组）。"
                 "下游 RL/任务B/偏好对不得使用这些组。",
         "n_groups": len(val_used), "groups": val_used}, ensure_ascii=False, indent=1))
    print(f"[write] val_groups.json: {len(val_used)} 组")
    (args.out / "dataset_info.json").write_text(
        json.dumps(DATASET_INFO, ensure_ascii=False, indent=2))

    report["samples"] = {k: len(v) for k, v in buckets.items()}
    report["val_groups"] = len(val_gids)
    report["errors"] = errors
    if checks:
        diffs = [c[2] for c in checks]
        report["selfcheck"] = {"n": len(checks), "mean_abs_diff_255": round(np.mean(diffs), 2),
                               "max_abs_diff_255": round(max(diffs), 2)}
        print(f"[selfcheck] n={len(checks)}  平均|Δ|={np.mean(diffs):.2f}/255  "
              f"最大={max(diffs):.2f}/255（含分辨率/JPEG/舍入差异，<6 算正常）")
    (args.out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        print(f"[warn] {len(errors)} 个 version 处理失败，明细见 build_report.json")
    print(f"[done] 总耗时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
