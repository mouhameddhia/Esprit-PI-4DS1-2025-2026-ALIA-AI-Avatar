"""
Hybrid adapter inference integration for backend.

Provides local inference using a fine-tuned 1.5B adapter with baseline fallback.
Use this from backend/routes/chat.py as an optional replacement for the in-repo NLP analyzer.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
HYBRID_CONFIG_PATH = REPO_ROOT / "NLP" / "fine_tuning" / "hybrid_adapter_config.json"

def load_hybrid_adapter_config() -> Dict[str, Any]:
    """Load hybrid adapter config from NLP/fine_tuning/hybrid_adapter_config.json."""
    if not HYBRID_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Hybrid adapter config not found: {HYBRID_CONFIG_PATH}")

    with HYBRID_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def init_hybrid_adapter_runtime() -> Optional[Dict[str, Any]]:
    """
    Initialize hybrid adapter runtime (model + tokenizer + PEFT).
    Returns runtime dict or None if init fails.
    
    Usage:
        runtime = init_hybrid_adapter_runtime()
        if runtime:
            prediction = infer_hybrid_adapter(
                messages=[...],
                runtime=runtime,
                hybrid_with_baseline=True
            )
    """
    try:
        config = load_hybrid_adapter_config()
        
        # Only init if adapter dir exists and we want to use it
        adapter_dir = Path(config["adapter_dir"])
        if not adapter_dir.exists():
            print(f"[HybridAdapter] Adapter dir not found: {adapter_dir}")
            return None
        
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        base_model_id = config["base_model"]
        print(f"[HybridAdapter] Loading base model: {base_model_id}")
        
        # Load base model (4-bit for memory efficiency)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"[HybridAdapter] Adapter loaded: {adapter_dir}")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
        }
    
    except Exception as e:
        print(f"[HybridAdapter] Init failed: {e}")
        return None


@lru_cache(maxsize=1)
def get_hybrid_adapter_runtime() -> Optional[Dict[str, Any]]:
    """Lazily initialize the hybrid adapter runtime when enabled by env var."""
    if os.getenv("ALIA_USE_HYBRID_ADAPTER", "0").lower() not in {"1", "true", "yes", "on"}:
        return None
    return init_hybrid_adapter_runtime()


def infer_hybrid_adapter(
    messages: list[dict],
    runtime: Dict[str, Any],
    max_new_tokens: int = 220,
    hybrid_with_baseline: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run inference with hybrid adapter (adapter + optional baseline merger).
    
    Returns prediction dict like:
    {
        "intent": "dosage_question",
        "needs_clarification": False,
        "safety_flags": [],
        "secondary_tags": [],
        "entity_map": {...}
    }
    """
    if runtime is None:
        return None
    
    try:
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]
        _config = runtime["config"]
        
        # Build chat template input
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(model.device)
        
        # Generate
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            top_k=None,
        )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON response
        import json
        try:
            prediction = json.loads(generated_text.strip())
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            prediction = {
                "intent": "other",
                "needs_clarification": True,
                "safety_flags": [],
                "secondary_tags": [],
                "entity_map": {},
                "error": "JSON parse failed",
            }
        
        return prediction
    
    except Exception as e:
        print(f"[HybridAdapter] Inference failed: {e}")
        return None
