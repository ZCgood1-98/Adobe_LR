#!/usr/bin/env python3
"""v2_diagnosis_gen.py — 审美诊断课产线（工作分解 2b / D）
=========================================================

老师 VLM（gemini-3.1-flash-lite-image via OpenRouter）看 (原图, 拟合渲染图, GT参数, bw)
写三段式诊断（Subject&mood / Current look / Target look；方向隐含在 current→target 对比里，不再单列 Plan），
末行附机器可读方向 JSON。方向一致性过滤器用 direction_calibration.json 把 JSON 逐轴对照
GT 参数符号（带死区），不符即弃 → 幻觉机械拦截。通过的样本只留散文（去掉 JSON 行）作学生标签。

学生样本复用现有 s1.jpg（原图），不新增图片存储；只从 train 非 val 组生成（防泄漏）。

d2 断点续跑：成功结果逐条 flush 进 diag_d2_checkpoint.jsonl，重跑跳过已完成、只补未完成/失败；
过滤判定不入 checkpoint（assembly 按当前 consistency_check 重算，改过滤器不必重烧 API）；签名文件挡
「换了老师 prompt 却复用旧 checkpoint」，改 prompt 后须 --fresh。

用法（仓库根目录，Chen 环境，先在 openrouter_config.json 填 key）：
  python 5_training/v2_diagnosis_gen.py d1              # 50 条试产 → 人工翻阅
  python 5_training/v2_diagnosis_gen.py d2              # 全量 ~7474;断了/崩了重跑同一命令即断点续跑
  python 5_training/v2_diagnosis_gen.py d2 --fresh      # 改了老师 prompt 后从头重跑(清 checkpoint)
  python 5_training/v2_diagnosis_gen.py d1 --workers 8
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))

from build_sft_dataset import collect_jobs  # noqa: E402

CONFIG = json.loads((REPO_ROOT / "5_training" / "openrouter_config.json").read_text())
CALIB = json.loads((REPO_ROOT / "sft_data_v2" / "direction_calibration.json").read_text())

# ============================================================
# Prompts（2026-07-20 定稿）
# ============================================================
TEACHER_PROMPT = """You are a professional photo retoucher writing a short diagnostic note for a training dataset.

You are given:
- IMAGE 1: the original, unedited photo
- IMAGE 2: the same photo after the expert's edit (the target result)
- The expert's edit directions (GROUND TRUTH — use them only to calibrate what you describe and the final summary; do NOT restate them as an explicit list of edits)
Black & white target: {bw}

Expert directions:
{skeleton}

Write a note in EXACTLY this three-part structure (about 60-100 words, English):

Subject & mood: one sentence — what IMAGE 1 shows and the feeling it carries.

Current look: 1-2 sentences — describe IMAGE 1's ACTUAL state as you genuinely see it in the pixels: its brightness (is it dark / medium / bright?), white balance (warm / neutral / cool?), saturation (muted / normal / vivid?), and contrast. Describe ONLY what is truly visible. Do NOT invent flaws and do NOT infer problems from the edit directions — many edits are stylistic choices, not corrections, so IMAGE 1 may already be well-exposed, neutral, or vivid. If it already looks fine in some respect, say so plainly.

Target look: one sentence — the aesthetic achieved in IMAGE 2, stated as the goal (describe the LOOK itself, e.g. "a cooler, moodier rendition"; do NOT narrate a before/after comparison and do NOT list editing steps).

Then, on the last line, output the machine-readable summary of the change from IMAGE 1's current look to this target look (consistent with the Expert directions above; use exactly "keep" for anything not adjusted):
{{"exposure":"brighten|darken|keep", "wb":"warm|cool|keep", "tint":"magenta|green|keep", "contrast":"up|down|keep", "highlights":"lower|raise|keep", "shadows":"lift|lower|keep", "saturation":"up|down|keep", "bw":true|false, "vignette":"darken|brighten|keep", "dehaze":"up|down|keep"}}"""


# axis → 该轴属于哪个 stage
STAGE_AXES = {"light": ["exposure", "wb", "tint"],
              "tone": ["contrast", "highlights", "shadows", "dehaze"],
              "color": ["saturation"], "detail": ["vignette"]}
# (axis, 方向词) → 自然语言短语
PHRASE = {
    ("exposure", "brighten"): "brighten the exposure", ("exposure", "darken"): "darken the exposure",
    ("wb", "warm"): "warm the white balance", ("wb", "cool"): "cool the white balance",
    ("tint", "magenta"): "shift the tint toward magenta", ("tint", "green"): "shift the tint toward green",
    ("contrast", "up"): "increase contrast", ("contrast", "down"): "reduce contrast",
    ("highlights", "raise"): "raise the highlights", ("highlights", "lower"): "recover the highlights",
    ("shadows", "lift"): "lift the shadows", ("shadows", "lower"): "deepen the shadows",
    ("saturation", "up"): "boost the saturation", ("saturation", "down"): "reduce the saturation",
    ("vignette", "darken"): "darken the corners with a vignette", ("vignette", "brighten"): "brighten the corners",
    ("dehaze", "up"): "add dehaze for clarity", ("dehaze", "down"): "reduce dehaze",
}
# 纳入骨架的最小幅度（略高于过滤器死区，避免把近中性的混淆轴写进去）
_MENTION = {"exposure": 0.15, "wb": 250., "tint": 5., "contrast": 8., "highlights": 8.,
            "shadows": 8., "saturation": 8., "vignette": 15., "dehaze": 8.}


def _mag_word(axis: str, m: float) -> str:
    t = {"exposure": (0.6, 1.5), "wb": (600, 1500), "vignette": (40, 90)}.get(axis, (20, 50))
    return "slightly" if m < t[0] else ("moderately" if m < t[1] else "strongly")


def direction_skeleton(params: dict, bw: bool) -> str:
    """GT 参数 → 按 stage 分组的自然语言方向骨架（喂给老师当 ground truth）。"""
    lines = []
    for stage, axes in STAGE_AXES.items():
        if stage == "color" and bw:
            lines.append("  color: convert to black & white (remove all saturation)")
            continue
        parts = []
        for axis in axes:
            pkey, neutral, _, _ = AXIS_MAP[axis]
            val = float(params.get(pkey, neutral))
            m = abs(val - neutral)
            if m < _MENTION[axis]:
                continue
            word = expected_word(axis, val)
            if word in ("keep", "__skip__"):
                continue
            ph = PHRASE.get((axis, word))
            if ph:
                parts.append(f"{ph} {_mag_word(axis, m)}")
        lines.append(f"  {stage}: " + ("; ".join(parts) if parts else "minimal, keep natural"))
    return "\n".join(lines)

STUDENT_PROMPT = ("<image>\nBlack & white target: {bw}.\n"
                  "Write a short diagnostic note for this photo in three parts — "
                  "Subject & mood; Current look; Target look (the aesthetic to aim for). "
                  "Do not output any parameter values or editing steps.")

# ============================================================
# 一致性过滤器：诊断 JSON 轴 → GT 参数键 + 死区
# ============================================================
# axis: (params.json 键, 中性值, 死区, calib 名)
AXIS_MAP = {
    "exposure":   ("曝光度", 0.0,   0.1, "exposure"),
    "wb":         ("色温",   5500., 150., "wb_temp"),
    "tint":       ("色调",   0.0,   3.0, "tint"),
    "contrast":   ("对比度", 0.0,   5.0, "contrast"),
    "highlights": ("高光",   0.0,   5.0, "highlights"),
    "shadows":    ("阴影",   0.0,   5.0, "shadows"),
    "saturation": ("饱和度", 0.0,   5.0, "saturation"),
    "vignette":   ("晕影",   0.0,  10.0, "vignette"),
    "dehaze":     ("去朦胧", 0.0,   5.0, "dehaze"),
}


def expected_word(axis: str, gt_val: float) -> str:
    """GT 参数值 → 期望方向词（用校准表；死区内 = keep）。"""
    _, neutral, dz, calib_name = AXIS_MAP[axis]
    c = CALIB[calib_name]
    if not c.get("consistent"):
        return "__skip__"                       # 校准不可靠的轴不判
    d = gt_val - neutral
    if abs(d) < dz:
        return "keep"
    return c["param_up_means"] if d > 0 else c["param_down_means"]


# 显著轴：散文会明确承诺、老师在缩略图上也能看清的方向(亮/暖/艳/黑白)。只有这些轴(+bw)方向搞反
# 才否决散文；高光/阴影/色调/对比/晕影/去朦胧属细微轴(散文不承诺、缩略图看不清、JSON 易滑)，反向
# 只记录不弃。2026-07-21 去 Plan 后收窄:方向由参数课去教,这里只挡"目标 look 明显讲反"。
SALIENT_AXES = {"exposure", "wb", "saturation"}


def consistency_check(teacher_json: dict, params: dict, gt_bw: bool) -> tuple[bool, list]:
    """逐轴对照，返回 (是否通过, 违规明细)。只有【显著轴(SALIENT_AXES)+bw】方向反向(fatal)才否决；
    细微轴反向仍记进 violations(复审可见)但不否决。老师漏报(说keep但GT动了)记 soft、从不 fatal。"""
    violations = []
    for axis, (pkey, neutral, dz, _) in AXIS_MAP.items():
        exp = expected_word(axis, float(params.get(pkey, neutral)))
        if exp == "__skip__":
            continue
        got = str(teacher_json.get(axis, "keep")).lower()
        if got != exp:
            # keep↔direction 的软违规与 方向反向 的硬违规都记；反向更严重
            hard = exp not in ("keep",) and got not in ("keep",) and got != exp
            violations.append({"axis": axis, "expected": exp, "got": got,
                               "severity": "hard" if hard else "soft",
                               "fatal": hard and axis in SALIENT_AXES})
    # bw 单独判
    got_bw = bool(teacher_json.get("bw", False))
    if got_bw != gt_bw:
        violations.append({"axis": "bw", "expected": gt_bw, "got": got_bw,
                           "severity": "hard", "fatal": True})
    n_fatal = sum(1 for v in violations if v["fatal"])
    return (n_fatal == 0), violations


# ============================================================
# 老师调用
# ============================================================
def _b64(path: Path, max_side: int) -> str:
    import cv2
    img = cv2.imread(str(path))
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            import openai
            key = CONFIG["api_key"]
            if not key or key.startswith("在这里") or key.startswith("<"):
                raise SystemExit("请先在 5_training/openrouter_config.json 填入 api_key")
            _client = openai.OpenAI(api_key=key, base_url=CONFIG["base_url"],
                                    timeout=CONFIG.get("timeout_s", 120))
    return _client


def parse_teacher(text: str) -> tuple[str, dict | None]:
    """拆 (散文, 方向JSON)。JSON = 最后一个 {...}；散文 = 其前的正文。"""
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    if not matches:
        return text.strip(), None
    m = matches[-1]
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError:
        return text.strip(), None
    prose = text[:m.start()].strip()
    # 去掉可能残留的 "machine-readable summary:" 之类尾注
    prose = re.sub(r"(?i)\n*.*machine[- ]readable.*$", "", prose).strip()
    return prose, js


def call_one(job: dict) -> dict:
    vid = f"{job['key']}/v{job['vn']}"
    vdir = REPO_ROOT / "fit_results" / job["key"] / f"version_{job['vn']}"
    try:
        params = json.loads((vdir / "params.json").read_text())
        orig = vdir / "original_cropped.png"
        if not orig.exists():
            orig = REPO_ROOT / "fit_results" / job["key"] / "original_captured_cropped.png"
        after = vdir / "rendered.png"
        ms = CONFIG.get("image_max_side", 768)
        img1, img2 = _b64(orig, ms), _b64(after, ms)           # 原图 + 终图
        # GT 方向骨架当 ground truth 喂给老师（它叙述、不猜方向；终图只用来描述 target look）
        skeleton = direction_skeleton(params, bool(job["bw"]))
        prompt = TEACHER_PROMPT.format(bw="yes" if job["bw"] else "no", skeleton=skeleton)
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2}"}},
        ]}]
        client = get_client()
        last_err = ""
        for attempt in range(CONFIG.get("max_retries", 3)):
            try:
                resp = client.chat.completions.create(
                    model=CONFIG["model"], messages=msgs, max_tokens=400, temperature=0.4,
                    extra_body={"usage": {"include": True}})
                text = resp.choices[0].message.content.strip()
                prose, js = parse_teacher(text)
                if js is None:
                    last_err = "no_json"; continue
                passed, viols = consistency_check(js, params, bool(job["bw"]))
                u = resp.usage
                cost = getattr(u, "cost", None) or (getattr(u, "model_extra", None) or {}).get("cost") or 0.0
                return {"vid": vid, "ok": True, "passed": passed, "prose": prose,
                        "json": js, "violations": viols, "category": job["category"],
                        "bw": bool(job["bw"]), "raw_len": len(text),
                        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                        "cost": float(cost)}
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)[:150]
                time.sleep(1.5 * (attempt + 1))
        return {"vid": vid, "ok": False, "err": last_err}
    except Exception as e:  # noqa: BLE001
        return {"vid": vid, "ok": False, "err": repr(e)[:150]}


# ============================================================
# 主流程
# ============================================================
def get_jobs(limit_stratified: int = 0):
    args = SimpleNamespace(
        out=REPO_ROOT / "sft_data_v2", splits=["train"], max_side=1024, jpeg_q=92,
        max_final_loss=8.0, max_render_diff=8.0,
        consistency_file=REPO_ROOT / "sft_data_v2" / "render_consistency.json",
        loss_whitelist=REPO_ROOT / "sft_data_v2" / "loss_whitelist.txt", limit=0)
    jobs, _ = collect_jobs(args)
    val_gids = set(json.loads((REPO_ROOT / "sft_data_v2" / "val_groups.json").read_text())["groups"])
    jobs = [j for j in jobs if j["key"] not in val_gids]     # train 非 val
    if limit_stratified:
        import random
        by_cat = {}
        for j in jobs:
            by_cat.setdefault((j["category"], j["bw"]), []).append(j)
        rng = random.Random(42)
        out, per = [], max(1, limit_stratified // max(len(by_cat), 1))
        for group in by_cat.values():
            out.extend(rng.sample(group, min(per, len(group))))
        rng.shuffle(out)
        return out[:limit_stratified]
    return jobs


def _vid(job: dict) -> str:
    return f"{job['key']}/v{job['vn']}"


def _params_of(vid: str) -> dict:
    key, vn = vid.rsplit("/v", 1)
    return json.loads((REPO_ROOT / "fit_results" / key / f"version_{vn}" / "params.json").read_text())


def _run_sig() -> str:
    """决定老师原始输出(散文+JSON)的因素签名;不含过滤器(过滤 assembly 时重算)。"""
    import hashlib
    return hashlib.md5((TEACHER_PROMPT + "|" + CONFIG["model"]).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["d1", "d2"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n_d1", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true",
                    help="d2:忽略并覆盖已有 checkpoint 从头重跑(改了老师 prompt/生成逻辑时用)")
    args = ap.parse_args()

    out = REPO_ROOT / "sft_data_v2"
    jobs = get_jobs(args.n_d1 if args.mode == "d1" else 0)
    if args.limit:
        jobs = jobs[:args.limit]

    # 断点续跑(仅 d2):成功结果逐条追加进 JSONL checkpoint,重跑跳过已完成、只补未完成/失败;
    # 失败不落盘→下次自动重试。过滤判定不进 checkpoint(assembly 时按当前 consistency_check 重算)。
    ckpt = out / "diag_d2_checkpoint.jsonl"
    sig_path = out / "diag_d2_checkpoint.sig"
    use_ckpt = (args.mode == "d2")
    if use_ckpt and args.fresh:
        for p in (ckpt, sig_path):
            if p.exists():
                p.unlink()
        print("[--fresh] 已清除旧 checkpoint")
    done: dict = {}
    if use_ckpt and ckpt.exists():
        if sig_path.exists() and sig_path.read_text().strip() != _run_sig():
            sys.exit(f"checkpoint 是用不同的老师 prompt/模型生成的,与当前不符。\n"
                     f"用新 prompt 重跑请加 --fresh,或删除 {ckpt.name}/{sig_path.name}。")
        for line in ckpt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                           # 崩溃残留的半行,跳过(该 job 会被重跑)
            if r.get("ok") and "vid" in r:
                done[r["vid"]] = r
    todo = [j for j in jobs if _vid(j) not in done]
    if use_ckpt and done:
        print(f"[续跑] checkpoint 已完成 {len(done)}，本次待补 {len(todo)}")
    print(f"[{args.mode}] 本次 {len(todo)} 个 version 待诊断（模型 {CONFIG['model']}）")

    t0, results, errs = time.time(), [], []
    ckpt_fh = ckpt.open("a", encoding="utf-8") if use_ckpt else None
    if use_ckpt and not sig_path.exists():
        sig_path.write_text(_run_sig())
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(call_one, j) for j in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r["ok"]:
                    results.append(r)
                    if ckpt_fh is not None:        # 成功即落盘 flush,崩溃不丢已完成
                        ckpt_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        ckpt_fh.flush()
                else:
                    errs.append(r)
                if n % 10 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    print(f"  [{n}/{len(todo)}] {rate:.2f}/s "
                          f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f}min", flush=True)
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()

    all_ok = list(done.values()) + results         # checkpoint + 本次成功 = 全部成功
    for r in all_ok:                                # 过滤判定按当前 consistency_check 重算(与 checkpoint 解耦)
        try:
            r["passed"], r["violations"] = consistency_check(r["json"], _params_of(r["vid"]), r["bw"])
        except Exception:                           # noqa: BLE001  params 缺失等,保留原判定
            pass
    passed = [r for r in all_ok if r["passed"]]
    print(f"\n成功 {len(all_ok)}（本次 {len(results)}）/ 本次失败 {len(errs)}；通过过滤 {len(passed)} "
          f"({len(passed) / max(len(all_ok), 1) * 100:.1f}%)")
    if all_ok:
        # 被拒=显著轴/bw 反向(fatal)；容忍=仅细微轴反向,已放行
        n_tol = sum(1 for r in passed
                    if any(v["severity"] == "hard" and not v.get("fatal") for v in r["violations"]))
        print(f"  被拒 {len(all_ok) - len(passed)}（显著轴/bw 方向反向）；"
              f"另 {n_tol} 组仅细微轴反向已容忍放行")

    # 用量账单(累计 checkpoint+本次)
    tin = sum(r.get("prompt_tokens", 0) for r in all_ok)
    tout = sum(r.get("completion_tokens", 0) for r in all_ok)
    tcost = sum(r.get("cost", 0.0) for r in all_ok)
    print(f"[用量·累计] {len(all_ok)} 次成功：输入 {tin:,} + 输出 {tout:,} = {tin + tout:,} tokens；"
          f"总成本 ${tcost:.4f}（均 ${tcost / max(len(all_ok), 1):.5f}/次）")
    if args.mode == "d1":
        review = [{"vid": r["vid"], "category": r["category"], "bw": r["bw"],
                   "passed": r["passed"], "violations": r["violations"],
                   "prose": r["prose"], "json": r["json"]} for r in all_ok]
        (out / "diag_d1_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1))
        print(f"[d1] 复审文件 → sft_data_v2/diag_d1_review.json（{len(review)} 条，请人工翻阅）")
        for r in passed[:2]:
            print(f"\n--- 样例 {r['vid']} ({r['category']}, bw={r['bw']}) ---\n{r['prose'][:500]}")
    else:
        samples = [{
            "messages": [
                {"role": "user", "content": STUDENT_PROMPT.format(bw="yes" if r["bw"] else "no")},
                {"role": "assistant", "content": r["prose"]},
            ],
            "images": [f"images/{r['vid'].rsplit('/v', 1)[0]}/v{r['vid'].rsplit('/v', 1)[1]}/s1.jpg"],
            "_meta": {"course": "diag", "group": r["vid"].rsplit("/v", 1)[0],
                      "version": r["vid"].rsplit("/v", 1)[1], "category": r["category"],
                      "bw": r["bw"]},
        } for r in passed]
        (out / "task_v2_diag_train.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1))
        n_missing = len(jobs) - len(all_ok)
        print(f"[d2] task_v2_diag_train.json: {len(samples)} 样本"
              + (f"；仍有 {n_missing} 个未完成/失败——重跑同一命令即可续跑" if n_missing else "；全部完成"))
        (out / "diag_gen_report.json").write_text(json.dumps(
            {"model": CONFIG["model"], "total": len(jobs), "ok": len(all_ok),
             "missing": n_missing, "errors_this_run": len(errs), "passed": len(passed),
             "tokens_in": tin, "tokens_out": tout, "cost_usd": round(tcost, 4),
             "elapsed_min": round((time.time() - t0) / 60, 1)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
