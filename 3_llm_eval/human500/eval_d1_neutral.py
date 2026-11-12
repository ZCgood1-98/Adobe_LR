#!/usr/bin/env python3
"""eval_d1_neutral.py — 用 500 个图对重测 D1（中性标签、强制二选一、正反两种顺序各问一次）
================================================================================

与旧版 D1（evaluate_llms.py）的四处区别：
  ① 标签中性：只说「Image A / Image B」，不再告诉模型哪张是原图、哪张是修后图；
  ② 强制二选一：去掉「差不多」选项，与人工投票页（只能选 A 或 B）同口径，可逐对比较；
  ③ 每对正反各问一次：原图在前一次、修后图在前一次 ⇒ 位置偏好与真实偏好可以分开；
  ④ 图片两侧同规格（build_pairs.py：长边 1024、jpg95），原样发送、不再二次压缩。

模型 = 7_baselines 的 C 类 7 个视觉语言模型；每条结果都存下**接口实际返回的模型名**、
提示词哈希、原始回复、用量与费用，事后可核实。

调用量：500 对 × 2 顺序 × 7 模型 = 7000 次。可断点续跑（按 模型 × 图对 × 顺序 去重）。

用法（仓库根目录）：
  python 3_llm_eval/human500/eval_d1_neutral.py --dry-run          # 只列任务、不发请求
  python 3_llm_eval/human500/eval_d1_neutral.py --limit 2          # 先用 2 对试跑（28 次调用）
  python 3_llm_eval/human500/eval_d1_neutral.py                    # 全量
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import requests
except ImportError:      # 空跑不需要；正式跑请用带 requests 的环境（如 ~/miniconda3/envs/Chen/bin/python）
    requests = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recompute_d1_d2 import SUBDIMS, parse_d1  # noqa: E402  复用已核对过的稳健解析器

# 与 7_baselines/smoke/run_bc.py 的 C_MODELS 相同
C_MODELS = [
    "deepseek/deepseek-v4.1-flash-1017", "z-ai/glm-5.3-flash", "xiaomi/mimo-v2.6-flash",
    "openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "google/gemini-3.8-flash",
    "qwen/qwen3.6-35b-a3b",
]

PROMPT = """You are a professional photography reviewer. Carefully compare the aesthetic quality of the two photos above (Image A and Image B).

Judge each of the following four dimensions separately, then give an overall conclusion. For every line you must choose exactly one of A or B; ties are not allowed.

[Dimension 1: Light and Shadow] Is the distribution of light and dark natural? How well are highlight details and shadow layers handled?
PREFER_LIGHT: <A or B>

[Dimension 2: Color] Is the color temperature appropriate? Does the saturation harmonize with the overall tone?
PREFER_COLOR: <A or B>

[Dimension 3: Detail and Texture] Texture clarity, sharpness, and delicacy of the image.
PREFER_DETAIL: <A or B>

[Dimension 4: Overall Visual Impact] First impression, visual tension, and emotional expression.
PREFER_IMPACT: <A or B>

Overall aesthetic judgment:
PREFER: <A or B>
REASON: <one sentence giving the core reason>"""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]

TEMPERATURE = 0.0
MAX_TOKENS = 4000        # 带思考的模型会先消耗一部分，留足余量
EMPTY_TRIES = 3          # 回复里解析不出总体选择时的重试次数
RETRY_HTTP = (408, 429, 500, 502, 503, 504, 520, 524)
ORDERS = ("original_first", "version_0_first")


def slug(model: str) -> str:
    return model.replace("/", "__")


def img_part(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def build_messages(first: Path, second: Path) -> list:
    return [{"role": "user", "content": [
        {"type": "text", "text": "Image A:"}, img_part(first),
        {"type": "text", "text": "Image B:"}, img_part(second),
        {"type": "text", "text": PROMPT},
    ]}]


def post(base: str, hdr: dict, body: dict, timeout: int = 240, tries: int = 6) -> tuple[dict, float]:
    """429 / 5xx / 网络错误指数退避重试，其余非 200 直接抛出。"""
    for k in range(tries):
        t = time.time()
        try:
            r = requests.post(f"{base}/chat/completions", headers=hdr, json=body, timeout=timeout)
        except requests.RequestException as e:
            if k == tries - 1:
                raise RuntimeError(f"network error: {e}") from e
        else:
            if r.status_code == 200:
                return r.json(), time.time() - t
            if r.status_code not in RETRY_HTTP or k == tries - 1:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        time.sleep(min(120, 5 * 2 ** k) * (0.5 + random.random()))
    raise RuntimeError("unreachable")


def run_one(job: dict, base: str, hdr: dict, data: Path) -> dict:
    p, order, model = job["pair"], job["order"], job["model"]
    first_is = "original" if order == "original_first" else "version_0"
    second_is = "version_0" if first_is == "original" else "original"
    rec = {
        "model": model, "sample_id": p["sample_id"], "order": order,
        "a_is": first_is, "b_is": second_is, "aspect_mismatch": p["aspect_mismatch"],
        "prompt_sha": PROMPT_SHA, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "cost": 0.0, "seconds": 0.0, "n_calls": 0, "ok": False,
    }
    msgs = build_messages(data / p[first_is]["file"], data / p[second_is]["file"])
    try:
        for _ in range(EMPTY_TRIES):
            js, dt = post(base, hdr, {"model": model, "messages": msgs,
                                      "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE})
            use = js.get("usage") or {}
            rec["cost"] = round(rec["cost"] + (use.get("cost") or 0.0), 6)
            rec["seconds"] = round(rec["seconds"] + dt, 1)
            rec["n_calls"] += 1
            rec["returned_model"] = js.get("model")       # 接口实际路由到的模型
            rec["provider"] = js.get("provider")
            ch = (js.get("choices") or [{}])[0]
            rec["finish_reason"] = ch.get("finish_reason")
            rec["raw"] = (ch.get("message") or {}).get("content") or ""
            rec["usage"] = {k: use.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            parsed = parse_d1(rec["raw"])
            if parsed["overall"] in ("A", "B"):
                break
        rec["parsed"] = parsed
        # 把 A/B 翻译成 original / version_0（C 或解析不了记为 None）
        to_side = {"A": first_is, "B": second_is}
        rec["chosen"] = to_side.get(parsed["overall"])
        rec["chosen_sub"] = {k: to_side.get(parsed[k]) for k in SUBDIMS}
        rec["ok"] = rec["chosen"] is not None
    except Exception as e:  # noqa: BLE001  单条失败不影响其余任务，续跑时会重试
        rec["error"] = str(e)[:400]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="500 图对 D1 重测（中性标签、正反两序）")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="仓库根目录（放接口配置的那份检出）")
    ap.add_argument("--data", type=Path, default=None, help="build_pairs.py 的输出目录")
    ap.add_argument("--models", nargs="+", default=C_MODELS)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个图对（试跑用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="只列任务与请求体大小，不发任何请求")
    args = ap.parse_args()
    repo = args.repo.resolve()
    data = (args.data or repo / "eval_results" / "human500").resolve()
    out_dir = data / "llm_d1"

    pairs = sorted(json.loads((data / "pairs.json").read_text(encoding="utf-8"))["pairs"],
                   key=lambda p: p["sample_id"])
    if args.limit:
        pairs = pairs[: args.limit]

    # 断点续跑：已有成功记录的 (图对, 顺序) 跳过；失败或解析不了的会重跑
    jobs, done_n = [], 0
    for model in args.models:
        f = out_dir / f"{slug(model)}.jsonl"
        done = set()
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("ok") and r.get("prompt_sha") == PROMPT_SHA:
                        done.add((r["sample_id"], r["order"]))
        done_n += len(done)
        jobs += [{"model": model, "pair": p, "order": o}
                 for p in pairs for o in ORDERS if (p["sample_id"], o) not in done]

    print(f"图对 {len(pairs)} × 顺序 {len(ORDERS)} × 模型 {len(args.models)} = "
          f"{len(pairs) * len(ORDERS) * len(args.models)} 次；已完成 {done_n}，待跑 {len(jobs)}")
    print(f"提示词哈希 {PROMPT_SHA}；输出目录 {out_dir}")
    if args.dry_run:
        if jobs:
            j = jobs[0]
            msgs = build_messages(data / j["pair"]["original"]["file"], data / j["pair"]["version_0"]["file"])
            kinds = [c["type"] for c in msgs[0]["content"]]
            size = len(json.dumps({"model": j["model"], "messages": msgs}))
            print(f"请求体结构 {kinds}，首个请求约 {size / 1e6:.2f} MB")
        print("空跑结束，没有发送任何请求。")
        return
    if not jobs:
        return
    if requests is None:
        sys.exit("当前 Python 没有 requests，请换环境运行，例如 ~/miniconda3/envs/Chen/bin/python")

    cfg =json.loads((repo / "5_training" / "openrouter_config.json").read_text(encoding="utf-8"))
    hdr = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    out_dir.mkdir(parents=True, exist_ok=True)
    random.Random(0).shuffle(jobs)          # 打散，避免对单个模型连续施压
    lock = threading.Lock()
    stat = {"n": 0, "ok": 0, "cost": 0.0}
    t0 = time.time()

    def work(job):
        rec = run_one(job, cfg["base_url"], hdr, data)
        with lock:
            with open(out_dir / f"{slug(job['model'])}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stat["n"] += 1; stat["ok"] += rec["ok"]; stat["cost"] += rec["cost"]
            if stat["n"] % 50 == 0 or stat["n"] == len(jobs):
                print(f"  {stat['n']}/{len(jobs)}  成功 {stat['ok']}  累计费用 ${stat['cost']:.2f}  "
                      f"用时 {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))
    print(f"完成：成功 {stat['ok']}/{stat['n']}，费用 ${stat['cost']:.2f}。失败的再运行一次本脚本即可续跑。")


if __name__ == "__main__":
    main()
