"""
Detailed evaluation of the trained affect classifier.

Prints per-class F1, precision, recall, confusion matrices,
and per-language accuracy breakdown.

Usage:
    python -m alia_nlp.scripts.evaluate_affect_classifier
    python -m alia_nlp.scripts.evaluate_affect_classifier --split train
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

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

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "affect_classifier"
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"

CONFIDENCE_MAP = {"low": 0, "medium": 1, "high": 2}
ENGAGEMENT_MAP = {"passive": 0, "engaged": 1}
URGENCY_MAP    = {"routine": 0, "elevated": 1, "urgent": 2}


# ── Model (mirrors train script exactly) ─────────────────────────────────────

class AffectClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.dropout          = nn.Dropout(0.25)
        self.confidence_head  = nn.Linear(h, 3)
        self.engagement_head  = nn.Linear(h, 2)
        self.urgency_head     = nn.Linear(h, 3)
        self.frustration_head = nn.Linear(h, 2)
        self.stress_head      = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return {
            "confidence":  self.confidence_head(pooled),
            "engagement":  self.engagement_head(pooled),
            "urgency":     self.urgency_head(pooled),
            "frustration": self.frustration_head(pooled),
            "stress":      self.stress_head(pooled),
        }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model() -> Tuple[AffectClassifier, Any, Dict]:
    with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    encoder   = AutoModel.from_pretrained(cfg["base_model_name"])
    model     = AffectClassifier(encoder)
    state     = torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, cfg


def encode(texts: List[str], tokenizer, max_length: int, batch_size: int = 32) -> Tuple:
    all_ids, all_masks = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    return torch.cat(all_ids), torch.cat(all_masks)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def classification_report(y_true: List[int], y_pred: List[int], labels: List[str]) -> str:
    n = len(labels)
    tp = [0] * n; fp = [0] * n; fn = [0] * n

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    lines = [f"  {'class':<18} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}"]
    lines.append("  " + "-" * 58)

    all_f1, total_support = [], sum(y_true.count(i) for i in range(n))

    for i, label in enumerate(labels):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup  = y_true.count(i)
        all_f1.append(f1)
        lines.append(f"  {label:<18} {prec:>9.3f} {rec:>9.3f} {f1:>9.3f} {sup:>9d}")

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = sum(all_f1) / len(all_f1)
    lines.append("  " + "-" * 58)
    lines.append(f"  {'accuracy':<18} {'':>9} {'':>9} {acc:>9.3f} {total_support:>9d}")
    lines.append(f"  {'macro F1':<18} {'':>9} {'':>9} {macro_f1:>9.3f} {total_support:>9d}")
    return "\n".join(lines)


def confusion_matrix_str(y_true: List[int], y_pred: List[int], labels: List[str]) -> str:
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        mat[t][p] += 1
    header = "        " + "".join(f"{l[:6]:>8}" for l in labels)
    lines  = [header]
    for i, row in enumerate(mat):
        lines.append(f"  {labels[i][:6]:<6}" + "".join(f"{v:>8}" for v in row))
    return "\n".join(lines)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(split: str = "val") -> None:
    path = DATA_DIR / ("affect_validation.jsonl" if split == "val" else "affect_training.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    print(f"\nLoading model from {MODEL_DIR}...")
    model, tokenizer, cfg = load_model()
    max_len = cfg.get("max_length", 128)

    records = load_jsonl(path)
    print(f"Evaluating on {len(records)} {split} examples\n")

    texts = [r["text"] for r in records]

    # Ground truth
    def eng_label(r):
        raw = r.get("engagement_level", "engaged")
        return ENGAGEMENT_MAP.get("engaged" if raw in ("active", "highly_engaged") else raw, 1)

    gt = {
        "confidence":  [CONFIDENCE_MAP.get(r.get("rep_confidence", "medium"), 1) for r in records],
        "engagement":  [eng_label(r) for r in records],
        "urgency":     [URGENCY_MAP.get(r.get("query_urgency", "routine"), 0) for r in records],
        "frustration": [int(r.get("frustration_signal", False)) for r in records],
        "stress":      [int(r.get("stress_signal", False)) for r in records],
    }

    # Inference in batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    input_ids, attention_mask = encode(texts, tokenizer, max_len)

    preds = defaultdict(list)
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            ids  = input_ids[i:i+batch_size].to(device)
            mask = attention_mask[i:i+batch_size].to(device)
            out  = model(ids, mask)
            for head in ("confidence", "engagement", "urgency", "frustration", "stress"):
                preds[head].extend(out[head].argmax(-1).cpu().tolist())

    # ── Per-dimension report ──────────────────────────────────────────────────
    HEADS = {
        "confidence":  (["low", "medium", "high"],       "rep_confidence"),
        "engagement":  (["passive", "engaged"],          "engagement_level"),
        "urgency":     (["routine", "elevated", "urgent"],"query_urgency"),
        "frustration": (["no", "yes"],                   "frustration_signal"),
        "stress":      (["no", "yes"],                   "stress_signal"),
    }

    for head, (labels, field) in HEADS.items():
        acc = sum(t == p for t, p in zip(gt[head], preds[head])) / len(gt[head])
        print(f"{'═'*60}")
        print(f"  {head.upper():<20} accuracy: {acc:.1%}")
        print(f"{'─'*60}")
        print(classification_report(gt[head], preds[head], labels))
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix_str(gt[head], preds[head], labels))
        print()

    # ── Per-language breakdown ────────────────────────────────────────────────
    print(f"{'═'*60}")
    print("  PER-LANGUAGE ACCURACY")
    print(f"{'─'*60}")
    for lang in ("en", "fr"):
        idx = [i for i, r in enumerate(records) if r.get("language") == lang]
        if not idx:
            continue
        print(f"\n  Language: {lang.upper()} ({len(idx)} examples)")
        for head in ("confidence", "engagement", "frustration", "stress", "urgency"):
            lang_gt   = [gt[head][i] for i in idx]
            lang_pred = [preds[head][i] for i in idx]
            acc = sum(t == p for t, p in zip(lang_gt, lang_pred)) / len(lang_gt)
            print(f"    {head:<14} {acc:.1%}")

    # ── Per-mode breakdown ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  PER-MODE ACCURACY")
    print(f"{'─'*60}")
    for mode in ("medrep_training", "physician_portal"):
        idx = [i for i, r in enumerate(records) if r.get("mode") == mode]
        if not idx:
            continue
        print(f"\n  Mode: {mode} ({len(idx)} examples)")
        for head in ("confidence", "engagement", "frustration", "stress", "urgency"):
            mode_gt   = [gt[head][i] for i in idx]
            mode_pred = [preds[head][i] for i in idx]
            acc = sum(t == p for t, p in zip(mode_gt, mode_pred)) / len(mode_gt)
            print(f"    {head:<14} {acc:.1%}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "train"], default="val",
                        help="Which split to evaluate (default: val)")
    args = parser.parse_args()
    evaluate(args.split)
