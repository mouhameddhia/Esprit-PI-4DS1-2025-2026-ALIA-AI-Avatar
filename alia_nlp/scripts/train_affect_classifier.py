"""
Affect classifier fine-tuning script — v3.

Improvements over v2:
  - Focal loss for binary heads (frustration, stress) — handles class imbalance
  - Layer-wise LR decay (LLRD) — bottom encoder layers get lower LR
  - Cosine annealing with warmup — better convergence than linear decay
  - Multi-turn context — optionally prepends previous turn for richer signal
  - Warnings filter for GradScaler false-positive

Local 4GB VRAM (GTX 1650):
    python -m alia_nlp.scripts.train_affect_classifier

High-quality run (Colab / more VRAM):
    python -m alia_nlp.scripts.train_affect_classifier \
        --model-name xlm-roberta-base --batch-size 16 --epochs 15

Output:
    alia_nlp/models/affect_classifier/
        tokenizer/   model.pt   config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

# GradScaler calls optimizer.step() internally; PyTorch's scheduler checker
# doesn't detect the indirect call and emits a false-positive warning.
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
    category=UserWarning,
)

_ENV_PATH = Path(__file__).resolve().parents[2] / "backend" / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except ImportError:
        for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Label mappings ────────────────────────────────────────────────────────────
CONFIDENCE_MAP = {"low": 0, "medium": 1, "high": 2}
ENGAGEMENT_MAP = {"passive": 0, "engaged": 1}
URGENCY_MAP    = {"routine": 0, "elevated": 1, "urgent": 2}

DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "affect_classifier"


# ── Dataset ───────────────────────────────────────────────────────────────────

class AffectDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r    = self.records[idx]
        text = r["text"]

        # Multi-turn: prepend previous turn if available
        context = r.get("context", "").strip()
        if context:
            text = context + f" {self.tokenizer.sep_token} " + text

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        eng_raw = r.get("engagement_level", "engaged")
        if eng_raw in ("active", "highly_engaged"):
            eng_raw = "engaged"

        return {
            "input_ids":         enc["input_ids"].squeeze(0),
            "attention_mask":    enc["attention_mask"].squeeze(0),
            "confidence_label":  torch.tensor(CONFIDENCE_MAP.get(r.get("rep_confidence", "medium"), 1), dtype=torch.long),
            "engagement_label":  torch.tensor(ENGAGEMENT_MAP.get(eng_raw, 1), dtype=torch.long),
            "urgency_label":     torch.tensor(URGENCY_MAP.get(r.get("query_urgency", "routine"), 0), dtype=torch.long),
            "frustration_label": torch.tensor(int(r.get("frustration_signal", False)), dtype=torch.long),
            "stress_label":      torch.tensor(int(r.get("stress_signal", False)),      dtype=torch.long),
        }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Model ─────────────────────────────────────────────────────────────────────

class AffectClassifier(nn.Module):
    def __init__(self, base_model_name: str, dropout: float = 0.25):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        h = self.encoder.config.hidden_size
        self.dropout          = nn.Dropout(dropout)
        self.confidence_head  = nn.Linear(h, 3)
        self.engagement_head  = nn.Linear(h, 2)
        self.urgency_head     = nn.Linear(h, 3)
        self.frustration_head = nn.Linear(h, 2)
        self.stress_head      = nn.Linear(h, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return {
            "confidence":  self.confidence_head(pooled),
            "engagement":  self.engagement_head(pooled),
            "urgency":     self.urgency_head(pooled),
            "frustration": self.frustration_head(pooled),
            "stress":      self.stress_head(pooled),
        }


# ── Class-weight helper ───────────────────────────────────────────────────────

def _class_weights(
    records: List[Dict[str, Any]],
    key: str,
    label_map: Dict,
    is_bool: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Inverse-frequency class weights, normalised so mean = 1."""
    n = len(label_map)
    counts = [0] * n
    for r in records:
        raw = r.get(key, None)
        idx = int(raw) if is_bool else label_map.get(raw, -1)
        if 0 <= idx < n:
            counts[idx] += 1
    total = sum(counts)
    w = [total / (n * max(c, 1)) for c in counts]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float, device=device)


# ── Loss functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights for imbalanced binary heads."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight: Optional[torch.Tensor] = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_loss(
    logits: Dict[str, torch.Tensor],
    batch:  Dict[str, torch.Tensor],
    losses: Dict[str, Any],
) -> torch.Tensor:
    return (
        losses["conf"] (logits["confidence"],  batch["confidence_label"])
        + losses["eng"](logits["engagement"],  batch["engagement_label"])
        + losses["urg"](logits["urgency"],     batch["urgency_label"])
        + losses["frust"](logits["frustration"], batch["frustration_label"])
        + losses["stress"](logits["stress"],     batch["stress_label"])
    )


# ── Layer-wise LR decay ───────────────────────────────────────────────────────

def _get_encoder_layers(encoder) -> List:
    """Return transformer layers regardless of model architecture."""
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return list(encoder.encoder.layer)          # BERT, XLM-R, DeBERTa
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return list(encoder.transformer.layer)      # DistilBERT
    return []


def build_llrd_optimizer(
    model: AffectClassifier,
    base_lr: float,
    head_lr: float,
    decay: float = 0.9,
) -> torch.optim.AdamW:
    """
    Layer-wise LR decay: heads get head_lr, top encoder layer gets base_lr,
    each lower layer gets base_lr * decay^i.
    """
    head_names = {"confidence_head", "engagement_head", "urgency_head",
                  "frustration_head", "stress_head", "dropout"}

    groups: List[Dict] = []

    # Classification heads
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in head_names)]
    groups.append({"params": head_params, "lr": head_lr})

    # Transformer layers (top → bottom with decay)
    layers = _get_encoder_layers(model.encoder)
    for i, layer in enumerate(reversed(layers)):
        layer_lr = base_lr * (decay ** i)
        groups.append({"params": list(layer.parameters()), "lr": layer_lr})

    # Embeddings + pooler (lowest LR)
    emb_lr = base_lr * (decay ** max(len(layers), 1))
    remaining = [
        p for n, p in model.encoder.named_parameters()
        if not any(f"layer.{j}" in n for j in range(len(layers)))
        and not any(h in n for h in head_names)
    ]
    if remaining:
        groups.append({"params": remaining, "lr": emb_lr})

    return torch.optim.AdamW(groups, weight_decay=0.01)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:  AffectClassifier,
    loader: DataLoader,
    device: torch.device,
    losses: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    correct    = {k: 0 for k in ("confidence", "engagement", "urgency", "frustration", "stress")}
    total      = 0

    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        total_loss += compute_loss(logits, batch, losses).item()
        total      += batch["input_ids"].size(0)
        for head, lkey in [
            ("confidence",  "confidence_label"),
            ("engagement",  "engagement_label"),
            ("urgency",     "urgency_label"),
            ("frustration", "frustration_label"),
            ("stress",      "stress_label"),
        ]:
            correct[head] += (logits[head].argmax(-1) == batch[lkey]).sum().item()

    return total_loss / len(loader), {k: v / total for k, v in correct.items()}


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model_name:      str   = "distilbert-base-multilingual-cased",
    batch_size:      int   = 8,
    grad_accum:      int   = 2,
    epochs:          int   = 15,
    lr:              float = 2e-5,
    llrd_decay:      float = 0.9,
    max_length:      int   = 128,
    warmup_ratio:    float = 0.1,
    label_smoothing: float = 0.1,
    focal_gamma:     float = 2.0,
    stress_gamma:    float = 4.0,
    patience:        int   = 3,
    seed:            int   = 42,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    logger.info("Device: %s | fp16: %s | model: %s", device, use_fp16, model_name)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_path = DATA_DIR / "affect_training.jsonl"
    val_path   = DATA_DIR / "affect_validation.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            "Run: python -m alia_nlp.scripts.generate_affect_dataset"
        )

    train_records = load_jsonl(train_path)
    val_records   = load_jsonl(val_path) if val_path.exists() else []
    logger.info("Loaded %d train / %d val records", len(train_records), len(val_records))

    multi_turn_count = sum(1 for r in train_records if r.get("context"))
    if multi_turn_count:
        logger.info("  %d records include multi-turn context", multi_turn_count)

    tokenizer    = AutoTokenizer.from_pretrained(model_name)
    train_ds     = AffectDataset(train_records, tokenizer, max_length)
    val_ds       = AffectDataset(val_records,   tokenizer, max_length) if val_records else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=use_fp16)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    # ── Model + LLRD optimizer ────────────────────────────────────────────────
    model     = AffectClassifier(model_name).to(device)
    optimizer = build_llrd_optimizer(model, base_lr=lr, head_lr=lr * 3, decay=llrd_decay)

    layers = _get_encoder_layers(model.encoder)
    logger.info(
        "LLRD: %d encoder layers | top-layer LR=%.1e | bottom-layer LR=%.1e | head LR=%.1e",
        len(layers), lr, lr * (llrd_decay ** max(len(layers) - 1, 0)), lr * 3,
    )

    effective_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = int(total_steps * warmup_ratio)

    # Cosine annealing with warmup (replaces linear decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ── Loss functions ────────────────────────────────────────────────────────
    # Rule: use focal OR class weights for each head, never both.
    # - frustration: focal γ=2.0 alone (was 79% in v2, proven working)
    # - stress: class weights alone (moderate 3× ratio) — focal made it collapse
    # - confidence/engagement/urgency: plain CE + label smoothing (weights hurt these)
    stress_w = _class_weights(
        train_records, "stress_signal", {0: 0, 1: 1}, is_bool=True, device=device
    )
    # Cap stress weight ratio at 3× to avoid overcorrection
    stress_w = torch.clamp(stress_w, max=3.0)
    stress_w = stress_w / stress_w.mean()   # re-normalise after clamping

    losses = {
        "conf":  nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        "eng":   nn.CrossEntropyLoss(),
        "urg":   nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        "frust": FocalLoss(gamma=focal_gamma),          # focal only, no weights
        "stress":nn.CrossEntropyLoss(weight=stress_w),  # weights only, no focal
    }

    logger.info(
        "Training: %d epochs | eff-batch %d | %d steps | %d warmup | patience %d | "
        "frust focal γ=%.1f | stress class-weights %s",
        epochs, effective_batch, total_steps, warmup_steps, patience,
        focal_gamma, [f"{w:.2f}" for w in stress_w.tolist()],
    )

    # ── Training loop with early stopping ─────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss   = compute_loss(logits, batch, losses) / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, device, losses)
            logger.info(
                "Epoch %2d/%d | train=%.4f | val=%.4f | "
                "conf=%.2f eng=%.2f urg=%.2f frust=%.2f stress=%.2f",
                epoch, epochs, avg_train_loss, val_loss,
                val_acc["confidence"], val_acc["engagement"],
                val_acc["urgency"],    val_acc["frustration"], val_acc["stress"],
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break
        else:
            logger.info("Epoch %2d/%d | train=%.4f", epoch, epochs, avg_train_loss)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_dir = OUTPUT_DIR / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(tok_dir))
    torch.save(best_state, OUTPUT_DIR / "model.pt")

    config = {
        "base_model_name": model_name,
        "max_length": max_length,
        "label_config": {
            "confidence":  {"map": CONFIDENCE_MAP,  "inv": {str(v): k for k, v in CONFIDENCE_MAP.items()}},
            "engagement":  {"map": ENGAGEMENT_MAP,   "inv": {str(v): k for k, v in ENGAGEMENT_MAP.items()}},
            "urgency":     {"map": URGENCY_MAP,      "inv": {str(v): k for k, v in URGENCY_MAP.items()}},
            "frustration": {"map": {"False": 0, "True": 1}, "inv": {"0": False, "1": True}},
            "stress":      {"map": {"False": 0, "True": 1}, "inv": {"0": False, "1": True}},
        },
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved to %s (best val_loss=%.4f)", OUTPUT_DIR, best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",      default="distilbert-base-multilingual-cased")
    parser.add_argument("--batch-size",      type=int,   default=8)
    parser.add_argument("--grad-accum",      type=int,   default=2)
    parser.add_argument("--epochs",          type=int,   default=15)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--llrd-decay",      type=float, default=0.9,
                        help="Layer-wise LR decay factor (default 0.9)")
    parser.add_argument("--max-length",      type=int,   default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma",     type=float, default=2.0,
                        help="Focal loss gamma for frustration head (default 2.0)")
    parser.add_argument("--stress-gamma",    type=float, default=4.0,
                        help="Focal loss gamma for stress head — higher = more focus on hard positives (default 4.0)")
    parser.add_argument("--patience",        type=int,   default=3)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()
    train(
        model_name=args.model_name,       batch_size=args.batch_size,
        grad_accum=args.grad_accum,       epochs=args.epochs,
        lr=args.lr,                       llrd_decay=args.llrd_decay,
        max_length=args.max_length,       label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,     stress_gamma=args.stress_gamma,
        patience=args.patience,
        seed=args.seed,
    )
