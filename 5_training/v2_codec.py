#!/usr/bin/env python3
"""v2_codec.py — SFT-v2 值离散化编解码器 + 词表手术（工作分解 A）
================================================================

职责：
  1) 57 个参数槽位注册表（顺序 = 输出 schema，与 extract_stages 的英文标签逐位对齐）
  2) 值 ↔ bin 编解码：中性点锚定分段归一化 + μ-law(μ=8) + 温度走 mired 轴
  3) 120 个新 token（57 键 + 63 值）注入 Qwen3-VL 词表保留区（151669+，不 resize）
  4) 键 token 均值初始化 / 值 token 插值初始化（供训练器调用）

⚠ 桶数 = 63（非立项书原定 64）：偶数桶时 c=0 落在桶边界，精确中性值（曝光 0、色温 5500）
   会解码出 +半桶 的系统偏移（如 +0.02EV）；「不动」占标签 28%，不可接受。
   奇数 63 桶让 31 号桶中心 == 中性点，「不动」精确往返。粒度差异可忽略（2/63 vs 2/64）。

自检：
  python 5_training/v2_codec.py check                # 数值往返 + 全量标签 + 渲染抽检（Chen 环境）
  conda run -n lf python 5_training/v2_codec.py check-tokenizer   # 词表手术（lf 环境）
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 常量
# ============================================================
MU = 8.0                 # μ-law 强度（2026-07-19 拍板）

# 值 token「桶组」：曝光是最高杠杆维，专属 127 桶（2026-07-20）；其余共享 63 桶。
# 桶数取奇数 → 中心桶 == 中性点，「不动」精确往返。
BIN_GROUPS = {"v": 63, "ev": 127}
GROUP_NEUTRAL = {"v": 31, "ev": 63}
VALUE_TOKENS_V = [f"<v_{i}>" for i in range(BIN_GROUPS["v"])]
VALUE_TOKENS_EV = [f"<ev_{i}>" for i in range(BIN_GROUPS["ev"])]
ALL_VALUE_TOKENS = VALUE_TOKENS_V + VALUE_TOKENS_EV
# 兼容旧引用（共享组默认值）
NBINS = BIN_GROUPS["v"]
NEUTRAL_BIN = GROUP_NEUTRAL["v"]

MIXER_CHANNELS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
CG_ZONES = ["shadow", "midtone", "highlight"]
STAGE_NAMES = ["light", "tone", "color", "detail"]

# 温度的 mired 轴（1e6/K；注意 K 与 mired 反序）
_T_LO_K, _T_HI_K, _T_N_K = 2000.0, 12000.0, 5500.0
_M_LO, _M_HI, _M_N = 1e6 / _T_HI_K, 1e6 / _T_LO_K, 1e6 / _T_N_K   # 83.33, 500, 181.82


@dataclass(frozen=True)
class Slot:
    token: str          # 键 token，如 "<exposure>"
    stage: str
    en_path: tuple      # 英文标签 dict 里的取值路径（tone_curve 用 int 下标）
    lo: float
    hi: float
    neutral: float
    one_sided: bool
    ndp: int            # 解码舍入小数位（与 v1 标签一致：曝光 2、色温 0、其余 1）
    desc: str           # 均值初始化用的英文描述
    group: str = "v"    # 值 token 桶组（曝光 "ev"=127 桶，其余 "v"=63 桶）


def slot_nbins(slot: "Slot") -> int:
    return BIN_GROUPS[slot.group]


def slot_neutral(slot: "Slot") -> int:
    return GROUP_NEUTRAL[slot.group]


def slot_vtok(slot: "Slot", b: int) -> str:
    return f"<{slot.group}_{b}>"


def _build_slots() -> list[Slot]:
    s: list[Slot] = []
    # ---- light ----
    s.append(Slot("<exposure>", "light", ("exposure",), -5, 5, 0, False, 2,
                  "photo exposure brightness adjustment in EV stops", group="ev"))
    s.append(Slot("<temperature>", "light", ("temperature",), _T_LO_K, _T_HI_K, _T_N_K, False, 0,
                  "white balance color temperature adjustment"))
    s.append(Slot("<tint>", "light", ("tint",), -150, 150, 0, False, 1,
                  "white balance tint green magenta adjustment"))
    # ---- tone ----
    for k, d in [("contrast", "tonal contrast adjustment"),
                 ("highlights", "highlights region brightness adjustment"),
                 ("shadows", "shadows region brightness adjustment"),
                 ("whites", "white point level adjustment"),
                 ("blacks", "black point level adjustment")]:
        s.append(Slot(f"<{k}>", "tone", (k,), -100, 100, 0, False, 1, d))
    for i in range(8):
        s.append(Slot(f"<tc_{i}>", "tone", ("tone_curve", i), -100, 100, 0, False, 1,
                      f"tone curve control point {i} adjustment from dark to bright"))
    s.append(Slot("<dehaze>", "tone", ("dehaze",), -100, 100, 0, False, 1,
                  "dehaze haze removal strength adjustment"))
    # ---- color ----
    s.append(Slot("<vibrance>", "color", ("vibrance",), -100, 100, 0, False, 1,
                  "vibrance smart saturation adjustment"))
    s.append(Slot("<saturation>", "color", ("saturation",), -100, 100, 0, False, 1,
                  "global color saturation adjustment"))
    for ch in MIXER_CHANNELS:
        cl = ch.lower()
        for a, an in [("h", "hue shift"), ("s", "saturation"), ("l", "luminance")]:
            s.append(Slot(f"<mix_{cl}_{a}>", "color", ("color_mixer", cl, a), -100, 100, 0, False, 1,
                          f"color mixer {cl} channel {an} adjustment"))
    for z in CG_ZONES:
        for c, cn in [("r", "red"), ("g", "green"), ("b", "blue")]:
            s.append(Slot(f"<cg_{z}_{c}>", "color", ("color_grading", z, c), -100, 100, 0, False, 1,
                          f"color grading {z} region {cn} balance adjustment"))
    # ---- detail ----
    s.append(Slot("<texture>", "detail", ("texture",), 0, 100, 0, True, 1,
                  "texture fine detail enhancement strength"))
    s.append(Slot("<clarity>", "detail", ("clarity",), 0, 100, 0, True, 1,
                  "clarity midtone local contrast strength"))
    s.append(Slot("<sharpen>", "detail", ("sharpen",), 0, 150, 0, True, 1,
                  "sharpening strength"))
    s.append(Slot("<noise_reduction>", "detail", ("noise_reduction",), 0, 100, 0, True, 1,
                  "noise reduction strength"))
    s.append(Slot("<vignette>", "detail", ("vignette",), -150, 150, 0, False, 1,
                  "vignette corner brightness adjustment"))
    return s


SLOTS: list[Slot] = _build_slots()
assert len(SLOTS) == 57, len(SLOTS)
KEY_TOKENS = [s.token for s in SLOTS]
ALL_NEW_TOKENS = KEY_TOKENS + ALL_VALUE_TOKENS      # 57 + 63 + 127 = 247
STAGE_SLOTS = {st: [s for s in SLOTS if s.stage == st] for st in STAGE_NAMES}
_SLOT_BY_TOKEN = {s.token: s for s in SLOTS}


# ============================================================
# 编解码数学
# ============================================================
def _to_dev(slot: Slot, v: float) -> float:
    """原值 → 中性锚定偏差 d ∈ [-1,1]（单向参数 ∈ [0,1]）。温度先转 mired。"""
    if slot.token == "<temperature>":
        m = 1e6 / min(max(float(v), _T_LO_K), _T_HI_K)
        return (m - _M_N) / (_M_HI - _M_N) if m >= _M_N else (m - _M_N) / (_M_N - _M_LO)
    v = min(max(float(v), slot.lo), slot.hi)
    if slot.one_sided:
        return (v - slot.lo) / (slot.hi - slot.lo)
    return (v - slot.neutral) / (slot.hi - slot.neutral) if v >= slot.neutral \
        else (v - slot.neutral) / (slot.neutral - slot.lo)


def _from_dev(slot: Slot, d: float) -> float:
    """偏差 d → 原值（含舍入；温度经 mired 反变换取整）。"""
    if slot.token == "<temperature>":
        m = _M_N + d * (_M_HI - _M_N) if d >= 0 else _M_N + d * (_M_N - _M_LO)
        m = min(max(m, _M_LO), _M_HI)
        return float(int(round(1e6 / m)))
    if slot.one_sided:
        d = min(max(d, 0.0), 1.0)
        v = slot.lo + d * (slot.hi - slot.lo)
    else:
        v = slot.neutral + d * (slot.hi - slot.neutral) if d >= 0 \
            else slot.neutral + d * (slot.neutral - slot.lo)
    out = round(v, slot.ndp)
    return 0.0 if out == 0 else out


def dev_to_bin(d: float, nbins: int) -> int:
    c = np.sign(d) * np.log1p(MU * abs(d)) / np.log1p(MU)
    return int(min(nbins - 1, max(0, np.floor((c + 1) / 2 * nbins))))


def bin_to_dev(b: int, nbins: int) -> float:
    c = (b + 0.5) / nbins * 2 - 1
    return float(np.sign(c) * ((1 + MU) ** abs(c) - 1) / MU)


def value_to_bin(slot: Slot, v: float) -> int:
    return dev_to_bin(_to_dev(slot, v), slot_nbins(slot))


def bin_to_value(slot: Slot, b: int) -> float:
    return _from_dev(slot, bin_to_dev(b, slot_nbins(slot)))


# ============================================================
# stage 级 编 / 解码（英文标签 dict ↔ token 序列）
# ============================================================
def _en_get(en: dict, path: tuple):
    cur = en
    for p in path:
        cur = cur[p]
    return cur


def encode_stage(stage: str, en: dict) -> list[str]:
    """英文标签 dict（extract_stages 的 en 输出）→ token 序列（键值相间）。"""
    out = []
    for slot in STAGE_SLOTS[stage]:
        out += [slot.token, slot_vtok(slot, value_to_bin(slot, float(_en_get(en, slot.en_path))))]
    return out


def decode_stage(stage: str, tokens: list[str]) -> tuple[dict, dict]:
    """token 序列 → (英文标签 dict, 渲染器中文参数 dict)。容错：按键找值、缺槽视为中性。"""
    vals: dict[str, float] = {}
    i = 0
    while i < len(tokens) - 1:
        t, nxt = tokens[i], tokens[i + 1]
        if t in _SLOT_BY_TOKEN:
            slot = _SLOT_BY_TOKEN[t]
            pref = f"<{slot.group}_"                       # 值 token 前缀按键的桶组定
            if nxt.startswith(pref) and nxt.endswith(">"):
                if slot.stage == stage:
                    vals[t] = bin_to_value(slot, int(nxt[len(pref):-1]))
                i += 2
                continue
        i += 1
    get = lambda tok: vals.get(tok, _SLOT_BY_TOKEN[tok].neutral if tok != "<temperature>" else _T_N_K)  # noqa: E731

    if stage == "light":
        en = {"exposure": get("<exposure>"), "temperature": int(get("<temperature>")),
              "tint": get("<tint>")}
        zh = {"曝光度": en["exposure"], "色温": float(en["temperature"]), "色调": en["tint"]}
    elif stage == "tone":
        tc = [get(f"<tc_{i}>") for i in range(8)]
        en = {"contrast": get("<contrast>"), "highlights": get("<highlights>"),
              "shadows": get("<shadows>"), "whites": get("<whites>"), "blacks": get("<blacks>"),
              "tone_curve": tc, "dehaze": get("<dehaze>")}
        zh = {"对比度": en["contrast"], "高光": en["highlights"], "阴影": en["shadows"],
              "白色色阶": en["whites"], "黑色色阶": en["blacks"],
              "tone_curve_dy": tc, "去朦胧": en["dehaze"]}
    elif stage == "color":
        mix_en, mix_zh = {}, []
        for ch in MIXER_CHANNELS:
            cl = ch.lower()
            h, s_, l = (get(f"<mix_{cl}_{a}>") for a in "hsl")
            mix_en[cl] = {"h": h, "s": s_, "l": l}
            mix_zh.append({"channel": ch, "H": h, "S": s_, "L": l})
        cg_en, cg_zh = {}, {}
        for z in CG_ZONES:
            r, g, b = (get(f"<cg_{z}_{c}>") for c in "rgb")
            cg_en[z] = {"r": r, "g": g, "b": b}
            cg_zh[z] = {"R": r, "G": g, "B": b}
        en = {"vibrance": get("<vibrance>"), "saturation": get("<saturation>"),
              "color_mixer": mix_en, "color_grading": cg_en}
        zh = {"自然饱和度": en["vibrance"], "饱和度": en["saturation"],
              "color_mixer": mix_zh, "颜色分级": cg_zh}
    else:  # detail
        en = {"texture": get("<texture>"), "clarity": get("<clarity>"),
              "sharpen": get("<sharpen>"), "noise_reduction": get("<noise_reduction>"),
              "vignette": get("<vignette>")}
        zh = {"纹理": en["texture"], "清晰度": en["clarity"], "锐化": en["sharpen"],
              "噪点消除": en["noise_reduction"], "晕影": en["vignette"]}
    return en, zh


# ============================================================
# 词表手术 + 初始化（lazy import，训练器调用）
# ============================================================
VOCAB_ROWS = 151936          # Qwen3-VL embedding 行数
RESERVED_START = 151669      # tokenizer 已用词条数（新 token 应从这里开始）
CONTROL_GUARD = 151643       # 此 id 起是 <|endoftext|>/<|im_*|>/</think> 等控制符，绝不触碰


def inject_tokens(tokenizer) -> dict[str, int]:
    """把 120 个新 token 注入词表保留区。幂等；断言不 resize、不碰控制符。"""
    tokenizer.add_special_tokens({"additional_special_tokens": ALL_NEW_TOKENS})
    ids = tokenizer.convert_tokens_to_ids(ALL_NEW_TOKENS)
    assert min(ids) >= RESERVED_START, f"新 token 落进已用区: min_id={min(ids)}"
    assert max(ids) < VOCAB_ROWS, f"超出 embedding 行数需 resize: max_id={max(ids)}"
    for tok, i in [("<|im_end|>", 151645), ("</think>", 151668)]:
        assert tokenizer.convert_ids_to_tokens(i) == tok, f"控制符 {tok} 被破坏"
    return dict(zip(ALL_NEW_TOKENS, ids))


def init_new_embeddings(model, tokenizer, noise: float = 0.01, seed: int = 42):
    """键 token=描述均值初始化；值 token=两端锚插值 + 微噪。输入/输出侧分别做。返回新 id 列表。"""
    import torch
    g = torch.Generator().manual_seed(seed)
    mapping = inject_tokens(tokenizer)
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    tied = emb.data_ptr() == head.data_ptr()

    def phrase_ids(text):
        return tokenizer(text, add_special_tokens=False).input_ids

    with torch.no_grad():
        for w in ([emb] if tied else [emb, head]):
            std = float(w[:RESERVED_START].std())
            for slot in SLOTS:                                   # 键：描述子词均值
                w[mapping[slot.token]] = w[phrase_ids(slot.desc)].mean(0)
            lo_v = w[phrase_ids("decrease this value very strongly to the minimum")].mean(0)
            hi_v = w[phrase_ids("increase this value very strongly to the maximum")].mean(0)
            for toks in (VALUE_TOKENS_V, VALUE_TOKENS_EV):       # 每组各自沿全域插值 + 微噪
                n = len(toks)
                for i, tok in enumerate(toks):
                    t = (i + 0.5) / n
                    # 噪声在 CPU 上用种子生成（保证确定性），再搬到权重所在设备
                    eps = torch.randn(w.shape[1], generator=g).to(w.device, w.dtype)
                    w[mapping[tok]] = (1 - t) * lo_v + t * hi_v + noise * std * eps
    return sorted(mapping.values())


def value_token_specs():
    """给训练器：各桶组的 (token 列表, 桶数, 中性桶)。"""
    return [("v", VALUE_TOKENS_V, BIN_GROUPS["v"], GROUP_NEUTRAL["v"]),
            ("ev", VALUE_TOKENS_EV, BIN_GROUPS["ev"], GROUP_NEUTRAL["ev"])]


# ============================================================
# 自检
# ============================================================
def _check_numeric():
    print("== ① bin 稳定性 + 中性精确往返 ==")
    worst = 0.0
    for slot in SLOTS:
        nb = slot_nbins(slot)
        b0 = slot_neutral(slot)
        lo_b = b0 if slot.one_sided else 0
        for b in range(lo_b, nb):
            v = bin_to_value(slot, b)
            b2 = value_to_bin(slot, v)
            assert b2 == b, f"{slot.token} bin{b}→{v}→bin{b2} 不稳定"
        n_dec = bin_to_value(slot, b0)
        n_ref = slot.neutral if slot.token != "<temperature>" else _T_N_K
        assert abs(n_dec - n_ref) < 1e-9, f"{slot.token} 中性不精确: {n_dec}"
        # 半桶误差上界（随机采样）
        rng = np.random.default_rng(0)
        vs = rng.uniform(slot.lo, slot.hi, 500)
        for v in vs:
            b = value_to_bin(slot, v)
            dec = bin_to_value(slot, b)
            nb1 = bin_to_value(slot, max(b - 1, 0))
            nb2 = bin_to_value(slot, min(b + 1, nb - 1))
            bound = max(abs(dec - nb1), abs(dec - nb2))
            err = abs(dec - min(max(v, slot.lo), slot.hi))
            worst = max(worst, err / max(bound, 1e-9))
            assert err <= bound + 1e-6, f"{slot.token} v={v} err={err} > 邻桶距 {bound}"
    print(f"  57 槽全过（曝光 {slot_nbins(_SLOT_BY_TOKEN['<exposure>'])} 桶/其余 63）；"
          f"最大误差/邻桶距 = {worst:.3f} (<1 即合格)")

    print("== ② 全量真实标签往返 ==")
    stats = {}
    n_neutral, n_neutral_ok = 0, 0
    for split in ["train", "val", "test"]:
        f = REPO_ROOT / "sft_data_v2" / f"task_a_{split}.json"
        for s in json.loads(f.read_text()):
            st = s["_meta"]["stage"]
            en = json.loads(s["messages"][2]["content"])
            toks = encode_stage(st, en)
            dec, _ = decode_stage(st, toks)
            for slot in STAGE_SLOTS[st]:
                v0, v1 = float(_en_get(en, slot.en_path)), float(_en_get(dec, slot.en_path))
                key = slot.token
                d = abs(v1 - min(max(v0, slot.lo), slot.hi))
                st_ = stats.setdefault(key, [0.0, 0.0, 0])
                st_[0] = max(st_[0], d); st_[1] += d; st_[2] += 1
                is_n = abs(v0 - (slot.neutral if key != "<temperature>" else _T_N_K)) < 1e-9
                if is_n:
                    n_neutral += 1
                    n_neutral_ok += int(abs(v1 - v0) < 1e-9)
    bad = {k: v for k, v in stats.items() if v[0] > 12}   # 温度以外没有槽该超 12
    print(f"  标签值 {sum(v[2] for v in stats.values()):,} 个；中性值精确保留 "
          f"{n_neutral_ok:,}/{n_neutral:,} ({n_neutral_ok / max(n_neutral, 1) * 100:.2f}%)")
    top = sorted(stats.items(), key=lambda x: -x[1][0])[:5]
    for k, (mx, sm, n) in top:
        print(f"  最大误差 top: {k:18} max={mx:7.2f}  mean={sm / n:6.3f}")
    assert n_neutral_ok == n_neutral, "存在中性值往返不精确!"
    assert not {k: v for k, v in bad.items() if k != "<temperature>"}, f"超界: {bad}"

    print("== ③ 渲染抽检（量化视觉无害性）==")
    sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "5_training"))
    import cv2
    from differentiable_renderer.opencv_renderer import render
    from build_sft_dataset import extract_stages, _imread_rgb, _resize_max_side
    rng = np.random.default_rng(42)
    vids = sorted(str(p.parent.relative_to(REPO_ROOT / "sft_data_v2" / "images"))
                  for p in (REPO_ROOT / "sft_data_v2" / "images").glob("train/*/v*/s1.jpg"))
    diffs = []
    for vid in rng.choice(vids, 150, replace=False):
        key, vn = vid.rsplit("/v", 1)
        vdir = REPO_ROOT / "fit_results" / key / f"version_{vn}"
        try:
            p = json.loads((vdir / "params.json").read_text())
            in_png = vdir / "original_cropped.png"
            if not in_png.exists():
                in_png = REPO_ROOT / "fit_results" / key / "original_captured_cropped.png"
            img0 = _resize_max_side(_imread_rgb(str(in_png)), 1024)
            cum_gt, cum_dec = {}, {}
            for i, st in enumerate(STAGE_NAMES):
                zh, en = extract_stages(p)[i]
                cum_gt.update(zh)
                cum_dec.update(decode_stage(st, encode_stage(st, en))[1])
            a = render(img0, cum_gt).astype(np.float64)
            b = render(img0, cum_dec).astype(np.float64)
            diffs.append(float(np.abs(a - b).mean()))
        except Exception as e:  # noqa: BLE001
            print("  跳过", vid, repr(e))
    d = np.array(diffs)
    print(f"  n={len(d)}  GT渲染 vs 量化后渲染 |Δ|: p50={np.percentile(d, 50):.2f} "
          f"p95={np.percentile(d, 95):.2f} max={d.max():.2f} /255")
    assert np.percentile(d, 95) < 4, "量化视觉误差超预期(底噪≈2)"
    print("数值自检全部通过 ✅")


def _check_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    n0 = len(tok)
    mapping = inject_tokens(tok)
    ids = sorted(mapping.values())
    print(f"注入前 len={n0} → 注入后 len={len(tok)}；新 id 区间 [{ids[0]}, {ids[-1]}]")
    mapping2 = inject_tokens(tok)      # 幂等
    assert mapping2 == mapping, "重复注入 id 漂移!"
    # 曝光走 <ev_>（127 桶），其余走 <v_>（63 桶）；两种前缀都要原子整词化
    enc = tok.encode("<exposure><ev_63><temperature><v_29>", add_special_tokens=False)
    assert enc == [mapping["<exposure>"], mapping["<ev_63>"],
                   mapping["<temperature>"], mapping["<v_29>"]], f"整词化失败: {enc}"
    assert len(mapping) == 57 + 63 + 127, f"token 数异常: {len(mapping)}"
    print(f"幂等 ✓ / <v_>+<ev_> 原子整词化 ✓ / 控制符完好 ✓ / "
          f"共 {len(mapping)} 个新 token（57 键 + 63 共享值 + 127 曝光值）")
    print("词表手术自检通过 ✅")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        _check_numeric()
    elif cmd == "check-tokenizer":
        _check_tokenizer()
    else:
        raise SystemExit(f"未知命令 {cmd}（check / check-tokenizer）")
