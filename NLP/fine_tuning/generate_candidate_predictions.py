#!/usr/bin/env python3
"""Generate candidate predictions for fine-tuning evaluation.

Supports three modes:
- hosted: calls an OpenAI-compatible chat completions endpoint
- local_baseline: uses the current in-repo NLP analyzer as a smoke-test generator
- local_adapter: runs local inference against a PEFT/LoRA adapter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.nlp import analyze_message_nlp
from NLP.taxonomy.validator import allowed_secondary_tags, load_taxonomy


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = str(msg.get("content", ""))
            marker = "Message:"
            if marker in content:
                after = content.split(marker, 1)[1].strip()
                return after.splitlines()[0].strip()
            return content.strip()
    return ""


def _extract_mode(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = str(msg.get("content", ""))
            marker = "Mode:"
            if marker in content:
                after = content.split(marker, 1)[1].strip()
                return after.splitlines()[0].strip() or "physician_portal"
    return "physician_portal"


def _extract_system_prompt(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return str(msg.get("content", ""))
    return ""


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _is_domain_specific_text(text: str) -> bool:
    corpus = text.strip().lower()
    if not corpus:
        return False
    domain_markers = [
        "product",
        "dosage",
        "dose",
        "indication",
        "active ingredient",
        "safety",
        "adverse",
        "contra",
        "interaction",
        "pregnancy",
        "visit",
        "crm",
        "simulation",
        "objection",
        "competency",
        "qare",
    ]
    return any(marker in corpus for marker in domain_markers)


def _should_force_clarification(
    text: str,
    intent: str,
    entity_map: Dict[str, List[str]],
    current_flag: bool,
) -> bool:
    if current_flag:
        return True

    corpus = text.strip().lower()
    if not corpus:
        return True

    # Very short non-domain prompts are typically under-specified.
    if len(corpus.split()) <= 3 and not _is_domain_specific_text(corpus):
        return True

    ambiguity_markers = [
        "not sure",
        "not certain",
        "what do you mean",
        "could you be specific",
        "help",
        "guidance",
        "okay",
        "yes",
        "no",
    ]
    if any(marker in corpus for marker in ambiguity_markers):
        return True

    if intent == "other" and not entity_map:
        return True

    return False


def _guess_intent(intent: str, text: str, mode: str) -> str:
    normalized = intent.strip().lower()
    aliases = {
        "object_property_request": "product_information_request",
        "object_property_query": "product_information_request",
        "product_info": "product_information_request",
        "greeting": "general_greeting",
    }
    if normalized in aliases:
        return aliases[normalized]

    corpus = f"{normalized} {text.lower()} {mode.lower()}"
    if any(k in corpus for k in ["dosage", "dose", "how often", "taken"]):
        return "dosage_question"
    if any(k in corpus for k in ["safety", "adverse", "interaction", "contra", "black box", "pregnancy"]):
        return "safety_question"
    if any(k in corpus for k in ["objection", "budget", "not persuaded", "concern"]):
        return "objection_handling"
    if any(k in corpus for k in ["simulate", "simulation", "teach", "framework", "training"]):
        return "training_simulation"
    if any(k in corpus for k in ["crm", "follow up", "follow-up"]):
        return "crm_follow_up"
    if any(k in corpus for k in ["competency", "expert", "junior", "confirme", "debutant"]):
        return "competency_assessment"
    if any(k in corpus for k in ["flash visit", "deep visit", "standard visit", "visit format"]):
        return "visit_format_request"
    if any(k in corpus for k in ["methodology", "qare", "argumentation", "acr"]):
        return "sales_methodology_request"
    if any(k in corpus for k in ["hello", "hey", "hi", "good morning", "got a minute"]):
        return "general_greeting"
    if "product" in corpus or "indication" in corpus or "active ingredient" in corpus:
        return "product_information_request"
    return "other"


def _project_prediction_to_taxonomy(
    prediction: Dict[str, Any],
    taxonomy: Dict[str, Any],
    text: str,
    mode: str,
) -> Dict[str, Any]:
    intents = {item for item in taxonomy.get("intents", []) if isinstance(item, str)}
    safety = {item for item in taxonomy.get("safety_flags", []) if isinstance(item, str)}
    secondary_allowed: Set[str] = allowed_secondary_tags(taxonomy)
    entity_types = {item for item in taxonomy.get("entity_types", []) if isinstance(item, str)}

    raw_intent = str(prediction.get("intent", "")).strip()
    intent = raw_intent if raw_intent in intents else _guess_intent(raw_intent, text, mode)
    if intent not in intents:
        intent = "other"

    raw_safety = _normalize_str_list(prediction.get("safety_flags"))
    safety_flags = [flag for flag in raw_safety if flag in safety]

    raw_secondary = _normalize_str_list(prediction.get("secondary_tags"))
    secondary_tags = [tag for tag in raw_secondary if tag in secondary_allowed]

    raw_entity_map_any = prediction.get("entity_map")
    raw_entity_map: Dict[str, Any] = raw_entity_map_any if isinstance(raw_entity_map_any, dict) else {}
    entity_map: Dict[str, List[str]] = {}
    for key, value in raw_entity_map.items():
        if not isinstance(key, str) or key not in entity_types:
            continue
        if isinstance(value, str):
            cleaned_vals = [value.strip()] if value.strip() else []
        else:
            cleaned_vals = _normalize_str_list(value)
        if cleaned_vals:
            entity_map[key] = cleaned_vals

    needs_clarification = _should_force_clarification(
        text=text,
        intent=intent,
        entity_map=entity_map,
        current_flag=bool(prediction.get("needs_clarification", False)),
    )

    return {
        "intent": intent,
        "needs_clarification": needs_clarification,
        "safety_flags": safety_flags,
        "secondary_tags": secondary_tags,
        "entity_map": entity_map,
    }


def _merge_projected_with_baseline(
    projected: Dict[str, Any],
    baseline: Dict[str, Any],
    taxonomy: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_projected = _project_prediction_to_taxonomy(
        prediction=baseline,
        taxonomy=taxonomy,
        text="",
        mode="",
    )

    projected_intent = str(projected.get("intent", "other"))
    baseline_intent = str(baseline_projected.get("intent", "other"))

    # For hybrid mode, trust baseline intent when it is confident enough to avoid over-generic adapter intents.
    if baseline_intent != "other":
        merged_intent = baseline_intent
    else:
        merged_intent = projected_intent

    projected_safety = _normalize_str_list(projected.get("safety_flags"))
    baseline_safety = _normalize_str_list(baseline_projected.get("safety_flags"))
    merged_safety = sorted(set(projected_safety) | set(baseline_safety))

    projected_secondary = _normalize_str_list(projected.get("secondary_tags"))
    baseline_secondary = _normalize_str_list(baseline_projected.get("secondary_tags"))
    merged_secondary = sorted(set(projected_secondary) | set(baseline_secondary))

    projected_entity_map_any = projected.get("entity_map")
    projected_entity_map: Dict[str, Any] = projected_entity_map_any if isinstance(projected_entity_map_any, dict) else {}
    baseline_entity_map_any = baseline_projected.get("entity_map")
    baseline_entity_map: Dict[str, Any] = baseline_entity_map_any if isinstance(baseline_entity_map_any, dict) else {}

    merged_entity_map: Dict[str, List[str]] = {}
    all_keys = set(projected_entity_map.keys()) | set(baseline_entity_map.keys())
    for key in all_keys:
        projected_vals = _normalize_str_list(projected_entity_map.get(key))
        baseline_vals = _normalize_str_list(baseline_entity_map.get(key))
        merged_vals = sorted(set(projected_vals) | set(baseline_vals))
        if merged_vals:
            merged_entity_map[str(key)] = merged_vals

    merged_needs_clarification = bool(projected.get("needs_clarification", False)) or bool(
        baseline_projected.get("needs_clarification", False)
    )

    merged = {
        "intent": merged_intent,
        "needs_clarification": merged_needs_clarification,
        "safety_flags": merged_safety,
        "secondary_tags": merged_secondary,
        "entity_map": merged_entity_map,
    }
    return _project_prediction_to_taxonomy(merged, taxonomy, "", "")


def _build_prompt(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not role or not content or role == "assistant":
            continue
        lines.append(f"<{role}>\n{content}\n</{role}>")
    lines.append("<assistant>\n")
    return "\n".join(lines)


def _build_fallback_prompt(messages: List[Dict[str, Any]]) -> str:
    mode = _extract_mode(messages)
    user_text = _extract_user_text(messages)
    system_prompt = _extract_system_prompt(messages)
    parts: List[str] = [
        system_prompt or "You are an NLP analyzer.",
        "Return exactly one JSON object with keys: intent, needs_clarification, safety_flags, secondary_tags, entity_map.",
        f"Mode: {mode}",
        f"Message: {user_text}",
    ]
    return "\n".join(parts).strip()


def _load_local_adapter(adapter_dir: Path) -> Dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies for local adapter mode. Install with: "
            "pip install torch transformers peft"
        ) from exc

    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise RuntimeError(f"Missing adapter config: {adapter_cfg_path}")

    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    base_model = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
    if not base_model:
        raise RuntimeError("adapter_config.json missing base_model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
    }


def _local_adapter_prediction(
    messages: List[Dict[str, Any]],
    runtime: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]

    def _run_generate(input_ids: Any, attention_mask: Any) -> str:
        bad_words_ids = [
            token_ids
            for token_ids in [
                tokenizer.encode("```", add_special_tokens=False),
                tokenizer.encode("<assistant>", add_special_tokens=False),
                tokenizer.encode("<user>", add_special_tokens=False),
                tokenizer.encode("<system>", add_special_tokens=False),
            ]
            if token_ids
        ]
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                bad_words_ids=bad_words_ids,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Attempt 1: native chat template (best match for instruct models).
    chat_messages: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role in {"system", "user"} and content:
            chat_messages.append({"role": role, "content": content})

    if chat_messages and hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            output_text = _run_generate(input_ids, None)
            parsed = _parse_json_object(output_text)
            if parsed:
                return parsed
        except Exception:
            pass

    # Attempt 2: training-style tagged prompt.
    prompt = _build_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    output_text = _run_generate(input_ids, attention_mask)
    parsed = _parse_json_object(output_text)
    if parsed:
        return parsed

    # Attempt 3: explicit fallback instruction.
    fallback_prompt = _build_fallback_prompt(messages)
    fallback_inputs = tokenizer(fallback_prompt, return_tensors="pt")
    fallback_ids = fallback_inputs["input_ids"].to(model.device)
    fallback_mask = fallback_inputs.get("attention_mask")
    if fallback_mask is not None:
        fallback_mask = fallback_mask.to(model.device)
    fallback_text = _run_generate(fallback_ids, fallback_mask)
    return _parse_json_object(fallback_text)


def _local_baseline_prediction(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_text = _extract_user_text(messages)
    mode = _extract_mode(messages)
    result = analyze_message_nlp(user_text=user_text, mode=mode, history=[])
    return {
        "intent": result.get("intent", "other"),
        "needs_clarification": bool(result.get("action_items") and "needs_intent_clarification" in result.get("action_items", [])),
        "safety_flags": result.get("safety_flags", []),
        "secondary_tags": result.get("secondary_tags", []),
        "entity_map": result.get("entity_map", {}),
        "confidence": result.get("confidence", 0.0),
        "language": result.get("language", "unknown"),
    }


def _call_openai_compatible_endpoint(messages: List[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
    base_url = os.getenv("FT_ENDPOINT_URL", "").rstrip("/")
    api_key = os.getenv("FT_API_KEY", "")
    model = os.getenv("FT_MODEL_NAME", "")

    if not base_url or not api_key or not model:
        raise RuntimeError("FT_ENDPOINT_URL, FT_API_KEY, and FT_MODEL_NAME must be set for hosted mode")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("FT_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("FT_MAX_TOKENS", "512")),
    }
    if system_prompt:
        payload["messages"] = messages

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"Hosted inference HTTP error: {exc.code} {exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Hosted inference connection error: {exc.reason}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0] if isinstance(choices[0], dict) else {}
    message_any = first.get("message")
    message: Dict[str, Any] = message_any if isinstance(message_any, dict) else {}
    content = str(message.get("content", ""))
    return _parse_json_object(content)


def _load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate predictions for fine-tuning evaluation")
    parser.add_argument("--reference-openai-jsonl", required=True, help="OpenAI-messages JSONL input")
    parser.add_argument("--output-jsonl", required=True, help="Where to write candidate predictions")
    parser.add_argument("--mode", choices=["hosted", "local_baseline", "local_adapter"], default="hosted")
    parser.add_argument("--config-json", default="", help="Optional hosted endpoint config JSON")
    parser.add_argument("--adapter-dir", default="", help="Path to local LoRA/QLoRA adapter directory")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation tokens for local_adapter mode")
    parser.add_argument("--taxonomy-json", default="NLP/taxonomy/nlp_taxonomy.json", help="Taxonomy for constrained projection")
    parser.add_argument("--hybrid-with-baseline", action="store_true", help="Merge local adapter output with local baseline output")
    args = parser.parse_args()

    ref_path = Path(args.reference_openai_jsonl).expanduser().resolve()
    if not ref_path.exists():
        print(f"Reference dataset not found: {ref_path}")
        return 1

    rows = _read_jsonl(ref_path)
    if not rows:
        print("No rows found in reference dataset")
        return 1

    out_path = Path(args.output_jsonl).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    taxonomy_path = Path(args.taxonomy_json).expanduser().resolve()
    if not taxonomy_path.exists():
        print(f"Taxonomy not found: {taxonomy_path}")
        return 1
    taxonomy = load_taxonomy(taxonomy_path)

    adapter_runtime: Dict[str, Any] = {}
    if args.mode == "local_adapter":
        adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir else None
        if adapter_dir is None or not adapter_dir.exists():
            print("--adapter-dir is required and must exist for local_adapter mode")
            return 1
        try:
            adapter_runtime = _load_local_adapter(adapter_dir)
        except Exception as exc:
            print(f"Failed to initialize local adapter runtime: {exc}")
            return 1

    if args.config_json:
        cfg_path = Path(args.config_json).expanduser().resolve()
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}")
            return 1
        cfg = _load_config(cfg_path)
        if isinstance(cfg.get("endpoint_url"), str) and cfg.get("endpoint_url"):
            os.environ["FT_ENDPOINT_URL"] = str(cfg.get("endpoint_url"))
        if isinstance(cfg.get("api_key"), str) and cfg.get("api_key"):
            os.environ["FT_API_KEY"] = str(cfg.get("api_key"))
        if isinstance(cfg.get("model_name"), str) and cfg.get("model_name"):
            os.environ["FT_MODEL_NAME"] = str(cfg.get("model_name"))
        if isinstance(cfg.get("temperature"), (int, float)):
            os.environ["FT_TEMPERATURE"] = str(cfg.get("temperature"))
        max_tokens = cfg.get("max_tokens")
        if isinstance(max_tokens, (int, float)):
            os.environ["FT_MAX_TOKENS"] = str(int(max_tokens))

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        if not messages:
            continue

        mode = _extract_mode(messages)
        user_text = _extract_user_text(messages)
        system_prompt = _extract_system_prompt(messages)

        try:
            if args.mode == "local_baseline":
                prediction = _local_baseline_prediction(messages)
            elif args.mode == "local_adapter":
                prediction = _local_adapter_prediction(messages, adapter_runtime, args.max_new_tokens)
            else:
                prediction = _call_openai_compatible_endpoint(messages, system_prompt)
        except Exception as exc:
            failures.append({"row": idx, "mode": mode, "text": user_text, "error": str(exc)})
            continue

        if not isinstance(prediction, dict):
            prediction = {}

        prediction = _project_prediction_to_taxonomy(prediction, taxonomy, user_text, mode)

        if args.mode == "local_adapter" and args.hybrid_with_baseline:
            baseline_prediction = _local_baseline_prediction(messages)
            prediction = _merge_projected_with_baseline(prediction, baseline_prediction, taxonomy)
            prediction = _project_prediction_to_taxonomy(prediction, taxonomy, user_text, mode)

        results.append(
            {
                "mode": mode,
                "text": user_text,
                "prediction": prediction,
            }
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} predictions to: {out_path}")
    if failures:
        print(f"Encountered {len(failures)} failures")
        for failure in failures[:10]:
            print(f"- row {failure['row']}: {failure['error']}")
        return 2 if not results else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
