"""
Hybrid adapter inference bridge — QLoRA Qwen2.5-1.5B-Instruct (v6).

Lazy-loads the fine-tuned LoRA adapter on first call using 4-bit quantization
(bitsandbytes NF4) so it fits inside 4 GB VRAM alongside the rest of the stack.

Enable:  ALIA_USE_HYBRID_ADAPTER=1  in backend/.env
Disable: ALIA_USE_HYBRID_ADAPTER=0  (default) — all calls return None immediately.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _ROOT / "alia_nlp" / "models" / "adapters" / "models" / "intent_qlora_v6_1p5b_multiepoch"
_PROMPTS_DIR = _ROOT / "alia_nlp" / "src" / "prompts"
_BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Module-level singletons ───────────────────────────────────────────────────
_lock           = threading.Lock()
_model          = None
_tokenizer      = None
_load_attempted = False


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_model() -> None:
    global _model, _tokenizer, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    if not (_ADAPTER_DIR / "adapter_model.safetensors").exists():
        logger.warning("QLoRA adapter weights not found at %s", _ADAPTER_DIR)
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        logger.info("Loading QLoRA base model %s (4-bit NF4)…", _BASE_MODEL)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(_BASE_MODEL, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        logger.info("Applying LoRA adapter from %s…", _ADAPTER_DIR)
        peft_model = PeftModel.from_pretrained(base, str(_ADAPTER_DIR))
        peft_model.eval()

        _model     = peft_model
        _tokenizer = tok
        logger.info("QLoRA adapter ready (%s + v6 LoRA)", _BASE_MODEL)

    except Exception as exc:
        logger.warning("QLoRA adapter load failed: %s — falling back to baseline NLP", exc)
        _model     = None
        _tokenizer = None


# ── Public API ────────────────────────────────────────────────────────────────

def get_hybrid_adapter_runtime() -> Optional[Any]:
    """Return (model, tokenizer) tuple, or None if unavailable/disabled."""
    if os.getenv("ALIA_USE_HYBRID_ADAPTER", "0").lower() not in {"1", "true", "yes"}:
        return None
    with _lock:
        _load_model()
    if _model is None or _tokenizer is None:
        return None
    return (_model, _tokenizer)


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def _build_system_prompt(mode: str) -> str:
    filename = (
        "extraction_physician.txt"
        if mode == "physician_portal"
        else "extraction_medrep.txt"
    )
    template = ""
    try:
        template = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    except Exception:
        return (
            "Extract intent from the message. "
            "Return JSON with intent, confidence, safety_flags, secondary_tags, entity_map."
        )

    mode_guidance = {
        "physician_portal": (
            "You are analyzing a query from a HEALTHCARE PROFESSIONAL. "
            "Clinical intents dominate. Training/simulation intents are not applicable."
        ),
        "medrep_training": (
            "You are analyzing a query from a MEDICAL REPRESENTATIVE practicing sales skills. "
            "Training, methodology, objection_handling, and visit intents are common."
        ),
    }.get(mode, "")

    try:
        from alia_nlp.data.taxonomy.loader import (
            SUPPORTED_INTENTS, SUPPORTED_SECONDARY_TAGS,
            ENTITY_TYPES, SUPPORTED_SAFETY_FLAGS,
        )
        return template.format(
            mode_guidance=mode_guidance,
            intents=", ".join(sorted(SUPPORTED_INTENTS)),
            secondary_tags=", ".join(sorted(SUPPORTED_SECONDARY_TAGS)) or "none",
            entity_types=", ".join(ENTITY_TYPES) or "none",
            safety_flags=", ".join(sorted(SUPPORTED_SAFETY_FLAGS)) or "none",
        )
    except Exception:
        return template


def _detect_mode(messages: list[dict[str, str]]) -> str:
    """Infer mode from the ALIA system prompt injected by chat_helpers."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "").lower()
            if "simulating a physician" in content or "medical representative" in content:
                return "medrep_training"
            return "physician_portal"
    return "physician_portal"


def _extract_user_text(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_hybrid_adapter(
    messages: list[dict[str, str]],
    runtime: Any,
    max_new_tokens: int = 220,
    hybrid_with_baseline: bool = True,
) -> Optional[dict[str, Any]]:
    """Run a single NLP extraction pass through the QLoRA adapter.

    Returns a dict with keys: intent, confidence, safety_flags,
    secondary_tags, entity_map, needs_clarification.
    Returns None on any failure so the caller falls back to baseline.
    """
    if runtime is None:
        return None

    model, tokenizer = runtime
    mode      = _detect_mode(messages)
    user_text = _extract_user_text(messages)
    if not user_text:
        return None

    system_prompt = _build_system_prompt(mode)
    user_prompt   = f"Mode: {mode}\nMessage: {user_text}"

    try:
        import torch

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # greedy — deterministic JSON
                temperature=1.0,           # ignored when do_sample=False
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract the first BALANCED JSON object (model may emit text after closing brace)
        start = raw.find("{")
        if start == -1:
            logger.warning("QLoRA output contained no JSON: %r", raw[:200])
            return None
        depth, end = 0, -1
        for idx, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end == -1:
            logger.warning("QLoRA output has unbalanced braces: %r", raw[:200])
            return None

        result = json.loads(raw[start:end])

        return {
            "intent": result.get("intent", "other") if isinstance(result.get("intent"), str) else "other",
            "confidence": float(result.get("confidence") or 0.0),
            "safety_flags": result.get("safety_flags", []) if isinstance(result.get("safety_flags"), list) else [],
            "secondary_tags": result.get("secondary_tags", []) if isinstance(result.get("secondary_tags"), list) else [],
            "entity_map": result.get("entity_map", {}) if isinstance(result.get("entity_map"), dict) else {},
            "needs_clarification": bool(result.get("needs_clarification", False)),
        }

    except json.JSONDecodeError as exc:
        logger.warning("QLoRA output is not valid JSON: %s | raw: %r", exc, raw[:300])
        return None
    except Exception as exc:
        logger.warning("QLoRA inference error: %s", exc)
        return None
