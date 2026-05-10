"""
Fine-tuned affect classifier — inference wrapper.

Lazy-loads the model on first call. Returns None if model directory
does not exist yet (pipeline falls back to LLM → rules).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import torch

from alia_nlp.src.layers.L7_affect.schema import AffectResult

logger = logging.getLogger(__name__)

_MODEL_DIR      = Path(__file__).resolve().parents[3] / "models" / "affect_classifier"
_lock           = threading.Lock()
_model          = None
_tokenizer      = None
_cfg            = None
_load_attempted = False


def _load_model() -> None:
    global _model, _tokenizer, _cfg, _load_attempted

    if _load_attempted:
        return
    _load_attempted = True

    if not (_MODEL_DIR / "model.pt").exists():
        logger.debug("Affect classifier not found at %s — using fallback", _MODEL_DIR)
        return

    try:
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        with open(_MODEL_DIR / "config.json", encoding="utf-8") as f:
            _cfg = json.load(f)

        base_name = _cfg["base_model_name"]
        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR / "tokenizer"))

        # Architecture must match train_affect_classifier.py exactly
        class _AffectClassifier(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                h = encoder.config.hidden_size
                self.dropout          = nn.Dropout(0.25)
                self.confidence_head  = nn.Linear(h, 3)
                self.engagement_head  = nn.Linear(h, 2)   # binary
                self.urgency_head     = nn.Linear(h, 3)
                self.frustration_head = nn.Linear(h, 2)
                self.stress_head      = nn.Linear(h, 2)   # new

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

        encoder = AutoModel.from_pretrained(base_name)
        net     = _AffectClassifier(encoder)
        state   = torch.load(_MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        _model = net
        logger.info("Affect classifier loaded (%s)", base_name)

    except Exception as exc:
        logger.warning("Failed to load affect classifier: %s — using fallback", exc)
        _model = None


def predict(text: str, mode: str, prev_turn: str = "") -> Optional[AffectResult]:
    with _lock:
        _load_model()

    if _model is None or _tokenizer is None or _cfg is None:
        return None

    try:
        max_len = _cfg.get("max_length", 128)
        # Prepend previous turn as context when available
        full_text = (prev_turn.strip() + f" {_tokenizer.sep_token} " + text) if prev_turn.strip() else text
        enc     = _tokenizer(
            full_text, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = _model(enc["input_ids"], enc["attention_mask"])

        lc = _cfg["label_config"]
        return AffectResult(
            rep_confidence   = lc["confidence"]["inv"][str(logits["confidence"].argmax(-1).item())],
            engagement_level = lc["engagement"]["inv"][str(logits["engagement"].argmax(-1).item())],
            query_urgency    = lc["urgency"]["inv"][str(logits["urgency"].argmax(-1).item())],
            frustration_signal = bool(logits["frustration"].argmax(-1).item()),
            stress_signal      = bool(logits["stress"].argmax(-1).item()),
            affect_source    = "model",
        )
    except Exception as exc:
        logger.warning("Affect classifier inference failed: %s", exc)
        return None
