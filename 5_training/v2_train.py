#!/usr/bin/env python3
"""v2_train.py — SFT-v2 自研训练管线（工作分解 E）
=================================================

不走 LLaMA-Factory（它不暴露 trainable_token_indices / 自定义 loss）。自研 HF Trainer：
  · Qwen3-VL-8B + LoRA；120 个新 token 注入词表保留区 + 均值/插值初始化（v2_codec）
  · 只训新 token 的 embedding/lm_head 行（peft trainable_token_indices，显存≈0）
  · 值 slot 上 **有序软标签 CE**（σ=1.0 高斯）+ 中性桶降权；键/格式/prompt 位不计 loss
  · 三课课程表 sampler（品味2/机制1/诊断2 遍;机制前置热身 → 中段均衡 → 收尾品味加倍）
  · 多模态 collator：品味课 1 图 / 机制课 2 图，用 Qwen3VLProcessor + assistant 掩码

用法（仓库根目录，lf 环境）：
  # 冒烟（验证只动 120 行 + loss 下降 + 双图显存）
  python 5_training/v2_train.py --smoke
  # 正式
  python 5_training/v2_train.py --output 5_training/output/sft_v2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "5_training"))

from v2_codec import (value_token_specs, init_new_embeddings,  # noqa: E402
                      STAGE_SLOTS)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DATA = REPO_ROOT / "sft_data_v2"
SIGMA = 1.0


# ============================================================
# 软标签 + 中性桶降权（每桶组一套：曝光 ev=127 / 共享 v=63）
# ============================================================
def build_soft_labels(nbins: int) -> torch.Tensor:
    """SL[t] = 以 t 为中心的高斯软标签（截断 ±3σ 重归一化），(nbins, nbins)。

    σ 按桶密度缩放：σ = SIGMA × nbins/63 —— 127 桶的曝光组 σ≈2，保证**值域上的平滑范围**
    与共享 63 桶组一致；顺带每次发射训练 ~2 倍邻域 token，对冲 ev 组的信号稀疏。
    """
    sigma = SIGMA * nbins / 63.0
    idx = np.arange(nbins)
    sl = np.exp(-((idx[None, :] - idx[:, None]) ** 2) / (2 * sigma ** 2))
    sl[np.abs(idx[None, :] - idx[:, None]) > 3 * sigma] = 0.0
    sl /= sl.sum(1, keepdims=True)
    return torch.tensor(sl, dtype=torch.float32)


def build_neutral_weights(group: str, prefix: str, nbins: int, neutral: int) -> torch.Tensor:
    """w[b] = clip(sqrt(median/count[b]), 0.15, 1)，count 取自品味课该桶组的真实标签分布。"""
    counts = np.zeros(nbins)
    data = json.loads((DATA / "task_v2_taste_train.json").read_text())
    import re
    pat = re.compile(rf"<{prefix}_(\d+)>")
    for s in data:
        for tok in pat.findall(s["messages"][2]["content"]):
            b = int(tok)
            if b < nbins:
                counts[b] += 1
    nz = counts[counts > 0]
    med = np.median(nz) if len(nz) else 1.0
    w = np.clip(np.sqrt(med / np.maximum(counts, 1)), 0.15, 1.0)
    print(f"[weights] 桶组 {group}: 中性桶({neutral}) count={int(counts[neutral])} "
          f"→ w={w[neutral]:.3f}；中位桶 w≈{np.median(w):.2f}")
    return torch.tensor(w, dtype=torch.float32)


# ============================================================
# 课程表：预生成 (course, idx) 序列
# ============================================================
# 计划(2026-07-21 定死):品味 2 遍 / 机制 1 遍 / 诊断 2 遍。
# 机制合成可再生、1 遍够;品味=行为克隆最重、2 遍;诊断=品味课 4:1 伴随(品味样本≈诊断×4,故各 2 遍即 4:1)。
COURSE_PASSES = {"taste": 2, "mech": 1, "diag": 2}


def build_curriculum(draws: dict, sizes: dict, seed: int = 42, spans: dict = None) -> list:
    """精确覆盖 + 位置塑形。draws[c]=该课总抽取次数(整遍打乱循环,不足一遍取前缀);
    spans[c]=该课发生的位置区间 → 机制前置(热身)、品味后置(收尾)、诊断伴随品味,
    形成「热身机制 → 中段均衡(≈47:40:12) → 收尾品味加倍」的形状,同时保证 draws 被精确消费。"""
    rng = random.Random(seed)
    spans = spans or {"mech": (0.0, 0.55), "taste": (0.10, 1.0), "diag": (0.15, 1.0)}
    items = []
    for c, nd in draws.items():
        n = sizes[c]; lo, hi = spans.get(c, (0.0, 1.0))
        order = []
        while len(order) < nd:                     # 整遍打乱、循环到够 nd 次
            perm = list(range(n)); rng.shuffle(perm); order += perm
        for idx in order[:nd]:
            items.append((lo + (hi - lo) * rng.random(), c, idx))   # 位置 = span 内均匀采样
    items.sort(key=lambda t: t[0])                 # 按位置排序即得课程序列
    return [(c, i) for _, c, i in items]


# ============================================================
# Dataset + Collator
# ============================================================
class V2Dataset(torch.utils.data.Dataset):
    def __init__(self, seq, courses):
        self.seq = seq
        self.courses = courses            # {course: [samples]}

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        course, idx = self.seq[i]
        return self.courses[course][idx]


def _to_msgs(sample):
    """ShareGPT messages（含 <image> 占位）→ 嵌入 PIL 图的 processor 消息。"""
    imgs = [Image.open(DATA / p).convert("RGB") for p in sample["images"]]
    ptr = 0
    msgs = []
    for m in sample["messages"]:
        c = m["content"]
        if isinstance(c, str) and "<image>" in c:
            content, parts = [], c.split("<image>")
            for k, seg in enumerate(parts):
                if k > 0:
                    content.append({"type": "image", "image": imgs[ptr]}); ptr += 1
                if seg:
                    content.append({"type": "text", "text": seg})
            msgs.append({"role": m["role"], "content": content})
        else:
            msgs.append({"role": m["role"], "content": [{"type": "text", "text": c}]})
    return msgs


class Collator:
    """逐样本 apply_chat_template（图嵌消息）→ 手动 pad(逐 token 场) + concat(逐图场)。

    labels = input_ids（不做 assistant 掩码）：值 token <v_N> 只出现在 assistant 输出里，
    损失函数只在值位算 loss → 天然只落 assistant 区，键/格式/prompt/图 全被过滤，最稳。
    """
    def __init__(self, processor, max_len=3072):
        self.p = processor
        self.pad_id = processor.tokenizer.pad_token_id
        self.max_len = max_len
        # assistant 头 token 串，用于定位 response 起点做标签掩码（散文/诊断课需要）
        self.head = processor.tokenizer("<|im_start|>assistant\n",
                                        add_special_tokens=False)["input_ids"]

    def _resp_start(self, ids: torch.Tensor) -> int:
        """input_ids 中最后一个 assistant 头之后的位置（response 起点）。"""
        h, n = self.head, len(self.head)
        ids_l = ids.tolist()
        for i in range(len(ids_l) - n, -1, -1):
            if ids_l[i:i + n] == h:
                return i + n
        return 0

    def __call__(self, batch):
        encs = []
        for s in batch:
            e = self.p.apply_chat_template(
                _to_msgs(s), tokenize=True, return_dict=True,
                add_generation_prompt=False, return_tensors="pt")
            encs.append({k: (v[0] if torch.is_tensor(v) and v.ndim and v.shape[0] == 1
                             and k in ("input_ids", "attention_mask", "mm_token_type_ids")
                             else v) for k, v in e.items()})
        T = min(self.max_len, max(e["input_ids"].shape[-1] for e in encs))

        def pad1(x, val):
            x = x[:T]
            return F.pad(x, (0, T - x.shape[0]), value=val)

        out = {}
        out["input_ids"] = torch.stack([pad1(e["input_ids"], self.pad_id) for e in encs])
        out["attention_mask"] = torch.stack([pad1(e["attention_mask"], 0) for e in encs])
        if "mm_token_type_ids" in encs[0]:
            out["mm_token_type_ids"] = torch.stack([pad1(e["mm_token_type_ids"], 0) for e in encs])
        labels = out["input_ids"].clone()
        for i, e in enumerate(encs):                       # 非 assistant 区 → -100
            labels[i, :self._resp_start(e["input_ids"])] = -100
        labels[out["attention_mask"] == 0] = -100
        out["labels"] = labels
        # 逐图场：拼接
        if "pixel_values" in encs[0]:
            out["pixel_values"] = torch.cat([e["pixel_values"] for e in encs], 0)
            out["image_grid_thw"] = torch.cat([e["image_grid_thw"] for e in encs], 0)
        return out


# ============================================================
# 自定义 Trainer：值 slot 软标签 CE
# ============================================================
def make_trainer_cls(value_groups: list, key_ids: torch.Tensor):
    """value_groups: [(value_ids, soft(nb×nb), nw(nb)), ...]，每桶组一套（曝光/共享）。"""
    from transformers import Trainer

    class V2Trainer(Trainer):
        """三分支损失：值 token → 该桶组的有序软标签 CE + 中性降权；键 token → 不计；
        其余 assistant token（诊断散文 / 结束符）→ 标准 CE。共同归一。"""
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits
            V = logits.shape[-1]
            dev = logits.device
            flat_logits = logits[:, :-1, :].reshape(-1, V).float()
            flat_labels = labels[:, 1:].reshape(-1)
            valid = flat_labels >= 0
            lab = flat_labels.clamp(min=0)

            num = flat_logits.new_zeros(())
            den = flat_logits.new_zeros(())
            is_value = torch.zeros_like(valid)
            for vids, soft, nw in value_groups:
                vids, soft, nw = vids.to(dev), soft.to(dev), nw.to(dev)
                nb = vids.shape[0]
                id2bin = torch.full((V,), -1, dtype=torch.long, device=dev)
                id2bin[vids] = torch.arange(nb, device=dev)
                tb = id2bin[lab]
                vpos = valid & (tb >= 0)
                is_value |= vpos
                if vpos.any():
                    sel = flat_logits[vpos]
                    b = tb[vpos]
                    logp = F.log_softmax(sel, -1)[:, vids]         # (Nv, nb)
                    ww = nw[b]
                    num = num + (-(soft[b] * logp).sum(-1) * ww).sum()
                    den = den + ww.sum()
            is_key = torch.zeros(V, dtype=torch.bool, device=dev)
            is_key[key_ids.to(dev)] = True
            kpos = valid & is_key[lab]                              # 键 token（丢弃）
            ppos = valid & ~is_value & ~kpos                       # 散文 / 结束符
            if ppos.any():
                num = num + F.cross_entropy(flat_logits[ppos], flat_labels[ppos], reduction="sum")
                den = den + ppos.sum()
            loss = num / den.clamp(min=1.0)
            return (loss, out) if return_outputs else loss

        def _get_train_sampler(self, *args, **kwargs):
            # 顺序消费预排好的课程表(热身机制→中段均衡→收尾品味)。默认 RandomSampler 会打散
            # 阶段调度(「0-20% 纯机制」等按步数的安排全失效)、也破坏 resume 确定性;
            # 用 SequentialSampler 让 build_curriculum 的顺序真正生效。
            ds = (args[0] if args else kwargs.get("train_dataset", None)) or self.train_dataset
            return torch.utils.data.SequentialSampler(ds)

    return V2Trainer


# ============================================================
# 模型构建
# ============================================================
def build_model(lora_r=32, lora_alpha=64):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa")
    new_ids = init_new_embeddings(model, processor.tokenizer)
    print(f"[model] 注入 {len(new_ids)} token；embedding 行 {model.get_input_embeddings().weight.shape[0]}")
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    print(f"[model] lm_head 与 embed {'共享(tied)' if tied else '独立(untied)'}")

    tti = new_ids if tied else {"embed_tokens": new_ids, "lm_head": new_ids}
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        trainable_token_indices=tti)
    model = get_peft_model(model, lora)
    # 冻结 ViT
    for n, p in model.named_parameters():
        if "visual" in n and "lora" not in n:
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 可训练参数 {trainable/1e6:.1f}M")
    return model, processor, new_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "5_training/output/sft_v2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--resume", action="store_true",
                    help="从 --output 最近的 checkpoint 续训(崩了/断了重跑同命令即续);无 checkpoint 则从头")
    args = ap.parse_args()

    from transformers import TrainingArguments

    # ---- 数据 ----
    courses = {}
    for c, f in [("taste", "task_v2_taste_train.json"), ("mech", "task_v2_mech_train.json"),
                 ("diag", "task_v2_diag_train.json")]:
        p = DATA / f
        courses[c] = json.loads(p.read_text()) if p.exists() else []
        print(f"[data] {c}: {len(courses[c])}")
    sizes = {c: len(v) for c, v in courses.items() if v}
    val = json.loads((DATA / "task_v2_taste_val.json").read_text())

    if args.smoke:
        draws = {c: min(24, sizes[c]) for c in sizes}          # 冒烟:每课抽点,确保三课(含诊断)都走到
    else:
        draws = {c: round(COURSE_PASSES.get(c, 0) * args.epochs * sizes[c]) for c in sizes}
    seq = build_curriculum(draws, sizes)
    print(f"[curriculum] 遍数×{args.epochs} → 抽取 {draws}  共 {len(seq)} 步"
          f"(≈{len(seq) // max(args.bs * args.grad_accum, 1)} optim step)")
    train_ds = V2Dataset(seq, courses)
    val_seq = [("taste", i) for i in range(min(len(val), 64 if args.smoke else len(val)))]
    val_ds = V2Dataset(val_seq, {"taste": val})

    # ---- 模型 ----
    model, processor, new_ids = build_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()   # grad ckpt + 冻结 base 时，输入需 require_grad 才能回传

    from v2_codec import KEY_TOKENS
    tok = processor.tokenizer
    value_groups = []
    for group, toks, nbins, neutral in value_token_specs():
        vids = torch.tensor(tok.convert_tokens_to_ids(toks))
        value_groups.append((vids, build_soft_labels(nbins),
                             build_neutral_weights(group, group, nbins, neutral)))
    key_ids = torch.tensor(tok.convert_tokens_to_ids(KEY_TOKENS))
    collator = Collator(processor, args.max_len)

    targs = TrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        num_train_epochs=1, max_steps=4 if args.smoke else -1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=1 if args.smoke else 10,
        save_steps=200, save_total_limit=3, eval_strategy="no" if args.smoke else "steps",
        eval_steps=200, per_device_eval_batch_size=args.bs,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=2)

    V2Trainer = make_trainer_cls(value_groups, key_ids)
    trainer = V2Trainer(model=model, args=targs, train_dataset=train_ds,
                        eval_dataset=None if args.smoke else val_ds, data_collator=collator)

    if args.smoke:
        _smoke_verify(model, new_ids)
        torch.cuda.reset_peak_memory_stats()
    resume = None
    if args.resume and not args.smoke:
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(str(args.output)) if args.output.exists() else None
        print(f"[resume] {'从 ' + resume + ' 续训' if resume else '未发现 checkpoint，从头开始'}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    if args.smoke:
        print(f"[smoke] 峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB / 32")
    if not args.smoke:
        trainer.save_model(str(args.output))
    print("[done]")


def _smoke_verify(model, new_ids):
    """冒烟前置：确认只训 LoRA + 120 新 token delta，622M 全量 embedding 未被训练。"""
    trn, emb_full = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trn += p.numel()
            # 全量 embedding/lm_head 矩阵（非 delta）若可训 = 灾难
            if p.numel() > 5e8 and "trainable_tokens" not in n:
                emb_full += p.numel()
    tt = [(n, tuple(p.shape)) for n, p in model.named_parameters()
          if p.requires_grad and "trainable_tokens" in n]
    print(f"[smoke] 可训参数 {trn/1e6:.1f}M；全量 embedding 被训练 = {emb_full/1e6:.0f}M（必须为 0）")
    for n, sh in tt:
        print(f"[smoke]   token delta: ...{n.split('.')[-3]} {sh}（应 (247, 4096)）")
    assert emb_full == 0, "灾难：整块 embedding 进了优化器！"


if __name__ == "__main__":
    main()
