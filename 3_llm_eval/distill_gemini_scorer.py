#!/usr/bin/env python3
"""
distill_gemini_scorer.py
========================
Distills Gemini's (google/gemini-3.1-flash-image-preview) aesthetic judgment
into a local CLIP ViT-L/14 + MLP scorer for use as a GRPO reward model.

Step 1 — generate:
    Call _run_pairwise from evaluate_llms.py for every version combination
    in the training split (original vs v0, v0 vs v1/v2/v3).
    Saves annotated preference pairs to a JSON file; fully resume-safe.

Step 2 — train:
    Load the clear preference pairs and train a Bradley-Terry CLIP+MLP scorer.
    Validates on a held-out clear subset; target ≥ 75% accuracy.

Step 3 — validate / route:
    Score individual images or route all version_N in a split into DPO/SFT tiers.

Usage:
    # Step 1: generate pairs (resume-safe, ~2h sequential or ~25min with --workers 5)
    python distill_gemini_scorer.py generate \\
        --dataset_root scraped_data_v4 --split train \\
        --output gemini_pairs.json [--workers 1]

    # Step 2: train scorer
    python distill_gemini_scorer.py train \\
        --pairs gemini_pairs.json \\
        --output_dir aesthetic_scorer_ckpt

    # Validate accuracy on clear pairs
    python distill_gemini_scorer.py validate \\
        --ckpt aesthetic_scorer_ckpt/best_scorer.pt --pairs gemini_pairs.json

    # Route version_N across the training split
    python distill_gemini_scorer.py route \\
        --ckpt aesthetic_scorer_ckpt/best_scorer.pt \\
        --dataset_root scraped_data_v4 --split train \\
        --output routing_result.json
"""

import json
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import clip
from tqdm import tqdm

# Reuse the existing pairwise machinery from the evaluator
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_llms import (
    _run_pairwise,
    OPENROUTER_API_KEY,
    MODELS,
)

# ============================================================
# SECTION 0: Configuration
# ============================================================

GEMINI_MODEL_ID = MODELS["gemini-nano-banana"]   # google/gemini-3.1-flash-image-preview

# Save the running cache every N completed pairs to guard against crashes
SAVE_INTERVAL = 20

# Each entry defines (pair_type_name, path_a_fn, path_b_fn).
# path_a maps to the "original_path" slot of _run_pairwise;
# path_b maps to the "edited_path" slot (position is randomized inside _run_pairwise).
PAIR_DEFINITIONS = [
    (
        "original_vs_v0",
        lambda d: d / "original_captured_cropped.png",
        lambda d: d / "version_0" / "main_image.png",
    ),
    (
        "v0_vs_v1",
        lambda d: d / "version_0" / "main_image.png",
        lambda d: d / "version_1" / "main_image.png",
    ),
    (
        "v0_vs_v2",
        lambda d: d / "version_0" / "main_image.png",
        lambda d: d / "version_2" / "main_image.png",
    ),
    (
        "v0_vs_v3",
        lambda d: d / "version_0" / "main_image.png",
        lambda d: d / "version_3" / "main_image.png",
    ),
]


# ============================================================
# SECTION 1: Dataset scanner
# ============================================================

def scan_pairs(dataset_root: Path, split: str) -> list[dict]:
    """
    Walk the dataset split and enumerate all (sample_id, pair_type, path_a, path_b)
    tuples whose image files actually exist on disk.
    """
    root = dataset_root / split
    if not root.exists():
        raise FileNotFoundError(f"Split directory not found: {root}")

    pairs = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = sample_dir.name
        for pair_name, get_a, get_b in PAIR_DEFINITIONS:
            path_a = get_a(sample_dir)
            path_b = get_b(sample_dir)
            if path_a.exists() and path_b.exists():
                pairs.append({
                    "sample_id": sid,
                    "pair_type": pair_name,
                    "path_a":    str(path_a),
                    "path_b":    str(path_b),
                })
    return pairs


# ============================================================
# SECTION 2: Preference pair generation  (Step 1)
# ============================================================

def _annotate_one_pair(pair: dict, api_key: str) -> dict:
    """
    Call Gemini on a single (path_a, path_b) pair via _run_pairwise and
    resolve the result into a (winner, loser, clarity) annotation.

    Clarity:
        "clear"     — Gemini chose A or B (confident preference)
        "ambiguous" — Gemini chose C (tie); pair is down-weighted at training time
        "error"     — API call failed; excluded from training
    """
    salt   = f"{pair['sample_id']}__{pair['pair_type']}"
    result = _run_pairwise(
        model_id=GEMINI_MODEL_ID,
        original_path=Path(pair["path_a"]),
        edited_path=Path(pair["path_b"]),
        salt=salt,
        api_key=api_key,
    )

    answer        = result.get("answer")         # "A", "B", or "C"
    edited_option = result.get("edited_option")  # which position holds path_b

    if answer in ("A", "B"):
        # Resolve winner/loser from position
        if answer == edited_option:
            winner, loser = pair["path_b"], pair["path_a"]
        else:
            winner, loser = pair["path_a"], pair["path_b"]
        clarity = "clear"
    else:
        winner = loser = None
        clarity = "ambiguous"

    return {
        "sample_id":            pair["sample_id"],
        "pair_type":            pair["pair_type"],
        "path_a":               pair["path_a"],
        "path_b":               pair["path_b"],
        "winner":               winner,
        "loser":                loser,
        "clarity":              clarity,
        "gemini_answer":        answer,
        "gemini_edited_option": edited_option,
        "gemini_subdim":        result.get("subdim"),
        "gemini_accuracy":      result.get("accuracy"),
    }


def generate_preference_pairs(
    dataset_root: str,
    split: str,
    output_path: str,
    api_key: str = None,
    max_workers: int = 1,
) -> list[dict]:
    """
    Generate Gemini preference annotations for all version combinations.

    Resume-safe: pairs already present in output_path are skipped.
    Saves incrementally every SAVE_INTERVAL completions.

    Args:
        dataset_root : root directory of the scraped dataset
        split        : "train" or "test"
        output_path  : JSON file to save/resume from
        api_key      : OpenRouter API key (falls back to evaluate_llms constant)
        max_workers  : parallel API threads (1 = sequential, safe for rate limits;
                       3–5 = reasonable for Gemini Flash)
    """
    api_key     = api_key or OPENROUTER_API_KEY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            done_list = json.load(f)
        print(f"Resuming: {len(done_list)} pairs already annotated.")
    else:
        done_list = []

    done_keys = {f"{r['sample_id']}__{r['pair_type']}" for r in done_list}

    # Enumerate all pairs that exist on disk and filter out already-done ones
    all_pairs = scan_pairs(Path(dataset_root), split)
    pending   = [p for p in all_pairs
                 if f"{p['sample_id']}__{p['pair_type']}" not in done_keys]

    print(f"Pairs on disk : {len(all_pairs)}")
    print(f"Already done  : {len(done_list)}")
    print(f"Pending       : {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return done_list

    results = list(done_list)  # working copy

    def _save():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if max_workers == 1:
        # Sequential — simplest, most reliable for rate-limited APIs
        for i, pair in enumerate(tqdm(pending, desc="Gemini pairwise")):
            try:
                annotated = _annotate_one_pair(pair, api_key)
            except Exception as e:
                print(f"\n  [ERR] {pair['sample_id']} {pair['pair_type']}: {e}")
                annotated = {
                    **pair,
                    "winner": None, "loser": None,
                    "clarity": "error", "gemini_answer": None,
                }
            results.append(annotated)
            if (i + 1) % SAVE_INTERVAL == 0:
                _save()
                clear = sum(1 for r in results if r.get("clarity") == "clear")
                print(f"  [{i+1}/{len(pending)}] {len(results)} total, {clear} clear")
    else:
        # Parallel — mind your rate limit; Gemini Flash is fairly generous
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_annotate_one_pair, pair, api_key): pair
                for pair in pending
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Gemini pairwise")
            for i, fut in enumerate(pbar):
                pair = futures[fut]
                try:
                    annotated = fut.result()
                except Exception as e:
                    print(f"\n  [ERR] {pair['sample_id']} {pair['pair_type']}: {e}")
                    annotated = {
                        **pair,
                        "winner": None, "loser": None,
                        "clarity": "error", "gemini_answer": None,
                    }
                results.append(annotated)
                if (i + 1) % SAVE_INTERVAL == 0:
                    _save()

    _save()

    # Summary
    clear_n = sum(1 for r in results if r.get("clarity") == "clear")
    ambig_n = sum(1 for r in results if r.get("clarity") == "ambiguous")
    error_n = sum(1 for r in results if r.get("clarity") == "error")
    total   = len(results)

    print(f"\nGeneration complete  ({output_path})")
    print(f"  Total     : {total}")
    print(f"  Clear     : {clear_n} ({100*clear_n/total:.1f}%)")
    print(f"  Ambiguous : {ambig_n} ({100*ambig_n/total:.1f}%)")
    print(f"  Errors    : {error_n}")

    # Per-pair-type breakdown
    for pt in ("original_vs_v0", "v0_vs_v1", "v0_vs_v2", "v0_vs_v3"):
        subset = [r for r in results if r.get("pair_type") == pt]
        c = sum(1 for r in subset if r.get("clarity") == "clear")
        print(f"  {pt:20s}: {len(subset):4d} total, {c:4d} clear")

    return results


# ============================================================
# SECTION 3: Training dataset
# ============================================================

class GeminiPairDataset(Dataset):
    """
    Loads (winner, loser) image pairs from the generated preference pairs JSON.

    By default only "clear" pairs are used (Gemini chose A or B definitively).
    Pass clear_only=False to also include "ambiguous" pairs if you want to
    experiment with soft labels or weighted sampling.
    """

    def __init__(self, pairs_path: str, clip_preprocess, clear_only: bool = True):
        with open(pairs_path, encoding="utf-8") as f:
            all_pairs = json.load(f)

        self.data = [
            p for p in all_pairs
            if p.get("clarity") == ("clear" if clear_only else p.get("clarity"))
            and p.get("winner") and p.get("loser")
        ]
        if clear_only:
            self.data = [p for p in all_pairs
                         if p.get("clarity") == "clear"
                         and p.get("winner") and p.get("loser")]

        self.preprocess = clip_preprocess
        tag = "clear only" if clear_only else "clear + ambiguous"
        print(f"GeminiPairDataset: {len(self.data)} pairs ({tag})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        winner = self.preprocess(Image.open(item["winner"]).convert("RGB"))
        loser  = self.preprocess(Image.open(item["loser"]).convert("RGB"))
        return winner, loser


class SingleImageDataset(Dataset):
    """Single-image dataset used for batch inference scoring."""

    def __init__(self, image_paths: list, clip_preprocess):
        self.paths      = [str(p) for p in image_paths]
        self.preprocess = clip_preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.paths[idx]).convert("RGB"))
        return img, self.paths[idx]


# ============================================================
# SECTION 4: Aesthetic scorer model
# ============================================================

class AestheticScorer(nn.Module):
    """
    Lightweight MLP head on top of CLIP ViT-L/14 features (dim=768).
    Outputs a scalar aesthetic score; higher = more aesthetically pleasing.
    CLIP backbone is frozen; only the MLP is trained.
    """

    def __init__(self, clip_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """clip_features: (B, clip_dim), L2-normalized. Returns (B,) scores."""
        return self.mlp(clip_features).squeeze(-1)


# ============================================================
# SECTION 5: Bradley-Terry loss
# ============================================================

def bradley_terry_loss(
    score_winner: torch.Tensor,
    score_loser: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry negative log-likelihood.
    P(winner > loser) = sigmoid(score_winner - score_loser)
    Loss = -log P = BCEWithLogitsLoss(score_w - score_l, ones)
    """
    return F.binary_cross_entropy_with_logits(
        score_winner - score_loser,
        torch.ones_like(score_winner),
    )


# ============================================================
# SECTION 6: Trainer  (Step 2)
# ============================================================

class AestheticScorerTrainer:
    def __init__(
        self,
        pairs_path: str,
        output_dir: str = "aesthetic_scorer_ckpt",
        clip_model_name: str = "ViT-L/14",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        batch_size: int = 32,
        num_epochs: int = 30,
        val_split: float = 0.15,
        clear_only: bool = True,
        device: str = None,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        print(f"Device : {self.device}")
        print(f"Loading CLIP {clip_model_name}...")
        self.clip_model, preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.clip_dim        = self.clip_model.visual.output_dim
        self.clip_model_name = clip_model_name
        self.scorer = AestheticScorer(self.clip_dim, hidden_dim, dropout).to(self.device)

        # Dataset split
        full_dataset = GeminiPairDataset(pairs_path, preprocess, clear_only=clear_only)
        val_size     = int(len(full_dataset) * val_split)
        train_size   = len(full_dataset) - val_size
        self.train_set, self.val_set = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Train : {train_size}  |  Val : {val_size}")

        self.train_loader = DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        self.optimizer = torch.optim.AdamW(
            self.scorer.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs,
        )

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        feats  = self.clip_model.encode_image(images).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.scorer.train(train)
        total_loss = total_correct = total = 0

        for winner_imgs, loser_imgs in loader:
            feat_w  = self._extract_features(winner_imgs)
            feat_l  = self._extract_features(loser_imgs)
            score_w = self.scorer(feat_w)
            score_l = self.scorer(feat_l)
            loss    = bradley_terry_loss(score_w, score_l)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
                self.optimizer.step()

            n             = len(winner_imgs)
            total_loss   += loss.item() * n
            total_correct += (score_w > score_l).sum().item()
            total        += n

        return total_loss / total, total_correct / total

    def train(self) -> float:
        best_val_acc = 0.0
        print("\nTraining...")
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = self.output_dir / "best_scorer.pt"
                torch.save(
                    {
                        "scorer_state":    self.scorer.state_dict(),
                        "clip_model_name": self.clip_model_name,
                        "clip_dim":        self.clip_dim,
                        "val_acc":         val_acc,
                        "epoch":           epoch,
                    },
                    ckpt_path,
                )
                print(f"  → Saved best checkpoint  (val acc = {val_acc:.3f})")

        print(f"\nTraining complete.  Best val accuracy: {best_val_acc:.3f}")
        if best_val_acc < 0.75:
            print("  ⚠ Target 75% not yet reached — consider more data or longer training.")
        return best_val_acc


# ============================================================
# SECTION 7: Inference interface
# ============================================================

class AestheticScorerInference:
    """Load a checkpoint and score arbitrary image paths."""

    def __init__(self, ckpt_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        clip_model_name = ckpt.get("clip_model_name", "ViT-L/14")
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.scorer = AestheticScorer(clip_dim=ckpt["clip_dim"]).to(self.device)
        self.scorer.load_state_dict(ckpt["scorer_state"])
        self.scorer.eval()

        print(
            f"Loaded scorer from {ckpt_path}  "
            f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.3f})"
        )

    @torch.no_grad()
    def score_images(self, image_paths: list, batch_size: int = 64) -> dict[str, float]:
        """Returns {path: score}. Higher score = more aesthetically pleasing."""
        dataset = SingleImageDataset(image_paths, self.preprocess)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        results: dict[str, float] = {}
        for imgs, paths in loader:
            imgs  = imgs.to(self.device)
            feats = self.clip_model.encode_image(imgs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for path, score in zip(paths, self.scorer(feats).tolist()):
                results[path] = score
        return results

    def score_sample_versions(self, sample_dir: str) -> dict[str, float]:
        """
        Score all available versions in one sample directory.
        Returns {"original": score, "version_0": score, ...}
        """
        sample_dir = Path(sample_dir)
        named: dict[str, Path] = {}

        orig = sample_dir / "original_captured_cropped.png"
        if orig.exists():
            named["original"] = orig
        for i in range(4):
            v = sample_dir / f"version_{i}" / "main_image.png"
            if v.exists():
                named[f"version_{i}"] = v

        if not named:
            return {}
        raw = self.score_images([str(p) for p in named.values()])
        return {name: raw[str(path)] for name, path in named.items()}


# ============================================================
# SECTION 8: Validation helper
# ============================================================

def validate_on_clear_pairs(ckpt_path: str, pairs_path: str, batch_size: int = 64) -> float:
    """
    Compute scorer accuracy on all clear pairs from the annotation file.
    Provides a per-pair-type breakdown so you can see where the scorer is weaker.
    Target: ≥ 75% overall accuracy.
    """
    scorer = AestheticScorerInference(ckpt_path)

    with open(pairs_path, encoding="utf-8") as f:
        all_pairs = json.load(f)

    clear = [p for p in all_pairs
             if p.get("clarity") == "clear"
             and p.get("winner") and p.get("loser")]
    print(f"Validating on {len(clear)} clear pairs...")

    # Score all unique images in one batched pass (efficient)
    all_paths = list({p["winner"] for p in clear} | {p["loser"] for p in clear})
    scores    = scorer.score_images(all_paths, batch_size=batch_size)

    correct_total = sum(
        1 for p in clear
        if scores.get(p["winner"], 0.0) > scores.get(p["loser"], 0.0)
    )
    acc = correct_total / len(clear)
    print(f"Overall accuracy: {correct_total}/{len(clear)} = {acc:.3f}")

    # Per-pair-type breakdown
    for pt in ("original_vs_v0", "v0_vs_v1", "v0_vs_v2", "v0_vs_v3"):
        subset = [p for p in clear if p.get("pair_type") == pt]
        if not subset:
            continue
        c = sum(1 for p in subset
                if scores.get(p["winner"], 0.0) > scores.get(p["loser"], 0.0))
        print(f"  {pt:20s}: {c}/{len(subset)} = {c/len(subset):.3f}")

    return acc


# ============================================================
# SECTION 9: Version routing (using trained scorer)
# ============================================================

def route_versions(
    scorer: AestheticScorerInference,
    dataset_root: str,
    split: str = "train",
    margin_high: float = 0.3,
    margin_low: float  = 0.1,
) -> dict:
    """
    Score all version_N images and sort samples into tiers for downstream DPO/SFT.

    Tier 1 — high-confidence DPO: |score_v0 - score_vN| > margin_high
    Tier 2 — ambiguous → SFT only: |score_v0 - score_vN| ≤ margin_low
    Tier 3 — flipped DPO: version_N clearly beats v0

    The margin thresholds are calibrated on the scorer's score distribution;
    0.3 / 0.1 are reasonable starting points.
    """
    root = Path(dataset_root) / split
    tier1, tier2, tier3 = [], [], []

    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"Routing {len(sample_dirs)} samples...")

    for sample_dir in tqdm(sample_dirs):
        scores = scorer.score_sample_versions(str(sample_dir))
        if "version_0" not in scores:
            continue

        score_v0 = scores["version_0"]
        path_v0  = str(sample_dir / "version_0" / "main_image.png")

        for vname in ("version_1", "version_2", "version_3"):
            if vname not in scores:
                continue
            score_vn = scores[vname]
            path_vn  = str(sample_dir / vname / "main_image.png")
            margin   = score_v0 - score_vn   # positive → v0 is better

            if margin > margin_high:
                tier1.append((path_v0, path_vn))   # v0 clearly better → chosen/rejected
            elif margin < -margin_high:
                tier3.append((path_vn, path_v0))   # vN clearly better → flip labels
            else:
                # Too close to call — keep only the higher-scoring one for SFT
                tier2.append(path_v0 if score_v0 >= score_vn else path_vn)

    stats = {
        "tier1_dpo_pairs":   len(tier1),
        "tier2_sft_samples": len(tier2),
        "tier3_flip_pairs":  len(tier3),
        "total":             len(tier1) + len(tier2) + len(tier3),
    }
    print(f"Tier 1 (high-confidence DPO): {len(tier1)}")
    print(f"Tier 2 (ambiguous → SFT):     {len(tier2)}")
    print(f"Tier 3 (flipped DPO):         {len(tier3)}")
    return {"tier1_dpo": tier1, "tier2_sft": tier2, "tier3_flip": tier3, "stats": stats}


# ============================================================
# SECTION 10: CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Distill Gemini aesthetic judgment into a local CLIP+MLP scorer."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Step 1: call Gemini and save preference pairs")
    gen.add_argument("--dataset_root", default="scraped_data_v4")
    gen.add_argument("--split",        default="train")
    gen.add_argument("--output",       default="gemini_pairs.json")
    gen.add_argument("--workers",      type=int, default=1,
                     help="Parallel API threads (1=sequential; 3-5 for Gemini Flash)")
    gen.add_argument("--api_key",      default=None,
                     help="OpenRouter key (defaults to evaluate_llms.OPENROUTER_API_KEY)")

    # ── train ─────────────────────────────────────────────────────────────────
    trn = sub.add_parser("train", help="Step 2: train CLIP+MLP scorer on preference pairs")
    trn.add_argument("--pairs",             required=True)
    trn.add_argument("--output_dir",        default="aesthetic_scorer_ckpt")
    trn.add_argument("--epochs",            type=int,   default=30)
    trn.add_argument("--batch_size",        type=int,   default=32)
    trn.add_argument("--lr",               type=float, default=1e-4)
    trn.add_argument("--include_ambiguous", action="store_true",
                     help="Also train on ambiguous (tie) pairs")

    # ── validate ──────────────────────────────────────────────────────────────
    val = sub.add_parser("validate", help="Compute accuracy on all clear pairs")
    val.add_argument("--ckpt",  required=True)
    val.add_argument("--pairs", required=True)

    # ── route ─────────────────────────────────────────────────────────────────
    rte = sub.add_parser("route", help="Score version_N and sort into DPO/SFT tiers")
    rte.add_argument("--ckpt",         required=True)
    rte.add_argument("--dataset_root", default="scraped_data_v4")
    rte.add_argument("--split",        default="train")
    rte.add_argument("--output",       default="routing_result.json")
    rte.add_argument("--margin_high",  type=float, default=0.3)
    rte.add_argument("--margin_low",   type=float, default=0.1)

    args = parser.parse_args()

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.mode == "generate":
        generate_preference_pairs(
            dataset_root=args.dataset_root,
            split=args.split,
            output_path=args.output,
            api_key=args.api_key,
            max_workers=args.workers,
        )

    elif args.mode == "train":
        trainer = AestheticScorerTrainer(
            pairs_path=args.pairs,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            clear_only=not args.include_ambiguous,
        )
        trainer.train()

    elif args.mode == "validate":
        acc = validate_on_clear_pairs(args.ckpt, args.pairs)
        if acc < 0.75:
            print("⚠ Target 75% not reached.")

    elif args.mode == "route":
        scorer = AestheticScorerInference(args.ckpt)
        result = route_versions(
            scorer,
            dataset_root=args.dataset_root,
            split=args.split,
            margin_high=args.margin_high,
            margin_low=args.margin_low,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Routing result saved to {args.output}")
        print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
