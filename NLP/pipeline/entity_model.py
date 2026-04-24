"""Lightweight trainable entity extraction model based on learned phrase lexicons."""

from __future__ import annotations

import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from NLP.taxonomy.entity_normalization import normalize_entity_value


_ENTITY_NORMALIZATION_PATH = Path(__file__).resolve().parents[1] / "taxonomy" / "entity_normalization.json"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _load_normalization_aliases() -> Dict[str, List[str]]:
    try:
        payload = json.loads(_ENTITY_NORMALIZATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    out: Dict[str, List[str]] = defaultdict(list)
    for entity_type, mapping in payload.items():
        if not isinstance(entity_type, str) or not isinstance(mapping, dict):
            continue
        for canonical, aliases in mapping.items():
            if not isinstance(canonical, str):
                continue
            out[entity_type].append(_normalize_text(canonical))
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip():
                        out[entity_type].append(_normalize_text(alias))
    return {k: sorted(set(v), key=lambda x: (-len(x), x)) for k, v in out.items()}


NORMALIZATION_ALIASES = _load_normalization_aliases()


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def train_entity_model(
    rows: Iterable[Dict[str, Any]],
    *,
    min_value_support: int = 1,
    max_values_per_entity: int = 200,
    strategy: str = "hybrid_lexicon",
) -> Dict[str, Any]:
    rows = list(rows)
    support: Dict[str, Counter[str]] = defaultdict(Counter)
    token_to_value: Dict[str, Counter[str]] = defaultdict(Counter)
    seen = 0
    used = 0

    for row in rows:
        seen += 1
        text = row.get("text")
        expected_map = row.get("expected_entity_map")
        if not isinstance(text, str) or not text.strip() or not isinstance(expected_map, dict):
            continue

        used += 1
        text_norm = _normalize_text(text)
        for entity_type, raw_values in expected_map.items():
            if not isinstance(entity_type, str) or not isinstance(raw_values, list):
                continue
            for raw in raw_values:
                if not isinstance(raw, str) or not raw.strip():
                    continue
                normalized = normalize_entity_value(entity_type, raw)
                if not normalized:
                    continue
                value_norm = _normalize_text(normalized)
                if not value_norm:
                    continue
                support[entity_type][value_norm] += 1

                for tok in re.findall(r"[a-z0-9']+", text_norm):
                    if len(tok) < 3:
                        continue
                    token_to_value[f"{entity_type}::{tok}"][value_norm] += 1

    lexicon: Dict[str, List[str]] = {}
    for entity_type, counts in support.items():
        kept = [
            value
            for value, count in counts.most_common(max_values_per_entity)
            if count >= max(1, int(min_value_support))
        ]
        if kept:
            aliases = NORMALIZATION_ALIASES.get(entity_type, [])
            merged = sorted(set(kept + aliases), key=lambda x: (-len(x), x))
            lexicon[entity_type] = merged[: max_values_per_entity]

    token_index: Dict[str, Dict[str, int]] = {}
    for key, counts in token_to_value.items():
        best = {value: count for value, count in counts.most_common(3) if count >= 2}
        if best:
            token_index[key] = best

    return {
        "model_type": "entity_phrase_lexicon",
        "version": "entity_lex_v1",
        "strategy": strategy if strategy in {"hybrid_lexicon", "lexicon"} else "hybrid_lexicon",
        "meta": {
            "rows_seen": seen,
            "rows_used": used,
            "entity_types": sorted(lexicon.keys()),
            "min_value_support": max(1, int(min_value_support)),
            "max_values_per_entity": max_values_per_entity,
        },
        "lexicon": lexicon,
        "token_index": token_index,
    }


def _rule_predict_entity_map(text: str) -> Dict[str, List[str]]:
    try:
        # Reuse deterministic production-safe rule extractor (no LLM/runtime proxy call).
        from NLP.pipeline.nlp import _rule_entity_map

        payload = _rule_entity_map(text)
        if isinstance(payload, dict):
            out: Dict[str, List[str]] = {}
            for key, values in payload.items():
                if not isinstance(key, str) or not isinstance(values, list):
                    continue
                deduped: List[str] = []
                for value in values:
                    if isinstance(value, str) and value.strip() and value not in deduped:
                        deduped.append(value)
                if deduped:
                    out[key] = deduped[:10]
            if out:
                return out
    except Exception:
        pass

    text_norm = _normalize_text(text)
    out: Dict[str, List[str]] = {}

    product_matches = re.findall(r"\bproduct\s+[a-z0-9-]+\b", text_norm)
    if product_matches:
        out["product_name"] = [m.title() for m in product_matches]

    if any(token in text_norm for token in ["flash", "30 seconds", "60 seconds", "one minute", "keep it short", "keep it brief"]):
        out.setdefault("visit_format", []).append("Flash")
    if "standard" in text_norm:
        out.setdefault("visit_format", []).append("Standard")
    if any(token in text_norm for token in ["deep", "approfondie"]):
        out.setdefault("visit_format", []).append("Approfondie")

    for level in ["Debutant", "Junior", "Confirme", "Expert"]:
        if level.lower() in text_norm:
            out.setdefault("competency_level", []).append(level)

    if any(token in text_norm for token in ["dose", "dosage", "route", "schedule", "dosing", "adjustment"]):
        out.setdefault("dosage", []).append("dosage")

    if any(token in text_norm for token in ["pregnancy", "pediatric", "children", "child", "elderly", "renal", "patient", "population"]):
        out.setdefault("patient_profile", []).append("patient")

    return out


def predict_entity_map(model: Dict[str, Any], text: str) -> Dict[str, List[str]]:
    text_norm = _normalize_text(text)
    lexicon_raw = model.get("lexicon")
    token_index_raw = model.get("token_index")
    lexicon: Dict[str, List[str]] = lexicon_raw if isinstance(lexicon_raw, dict) else {}
    token_index: Dict[str, Dict[str, int]] = token_index_raw if isinstance(token_index_raw, dict) else {}

    out: Dict[str, List[str]] = _rule_predict_entity_map(text)
    for entity_type, values in lexicon.items():
        if not isinstance(entity_type, str) or not isinstance(values, list):
            continue

        matches: List[Tuple[int, str]] = []
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            if _contains_phrase(text_norm, value):
                matches.append((len(value), value))

        matches.sort(reverse=True)
        deduped: List[str] = []
        for _, value in matches:
            normalized = normalize_entity_value(entity_type, value)
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        if deduped:
            out.setdefault(entity_type, [])
            for item in deduped:
                if item not in out[entity_type]:
                    out[entity_type].append(item)

    tokens = {tok for tok in re.findall(r"[a-z0-9']+", text_norm) if len(tok) >= 3}
    for token in tokens:
        for entity_type in list(lexicon.keys()):
            key = f"{entity_type}::{token}"
            suggestions = token_index.get(key)
            if not isinstance(suggestions, dict):
                continue
            candidate = next((value for value, _ in sorted(suggestions.items(), key=lambda x: x[1], reverse=True)), None)
            if not candidate:
                continue
            normalized = normalize_entity_value(entity_type, candidate)
            if not normalized:
                continue
            out.setdefault(entity_type, [])
            if normalized not in out[entity_type]:
                out[entity_type].append(normalized)

    for entity_type in list(out.keys()):
        unique: List[str] = []
        for value in out.get(entity_type, []):
            normalized = normalize_entity_value(entity_type, value)
            if normalized and normalized not in unique:
                unique.append(normalized)
        if unique:
            out[entity_type] = unique[:10]
        else:
            out.pop(entity_type, None)

    return out