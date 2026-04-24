"""NLP analysis utilities for intent/entity extraction and retrieval query rewriting."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from NLP.pipeline.language import detect_language
from NLP.pipeline.safety import detect_safety_flags
from NLP.taxonomy.entity_normalization import normalize_entity_value
from NLP.taxonomy.loader import load_taxonomy

try:
    from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2
except Exception:
    EntityExtractorV2 = None


logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "nlp_extraction_system_prompt.txt"
_DOMAIN_SYNONYMS_PATH = Path(__file__).resolve().parents[1] / "taxonomy" / "domain_synonyms.json"

TAXONOMY = load_taxonomy()
SUPPORTED_INTENTS = {
    intent for intent in (TAXONOMY.get("intents") or []) if isinstance(intent, str)
}
SUPPORTED_SAFETY_FLAGS = {
    flag for flag in (TAXONOMY.get("safety_flags") or []) if isinstance(flag, str)
}
ENTITY_TYPES = [
    entity_type
    for entity_type in (TAXONOMY.get("entity_types") or [])
    if isinstance(entity_type, str)
]
SUPPORTED_SECONDARY_TAGS = {
    *[
        tag
        for tag in (TAXONOMY.get("visit_phases") or [])
        if isinstance(tag, str)
    ],
    *[
        tag
        for tag in (TAXONOMY.get("visit_formats") or [])
        if isinstance(tag, str)
    ],
    *[
        tag
        for tag in (TAXONOMY.get("objection_types") or [])
        if isinstance(tag, str)
    ],
        *[
            tag
            for tag in (TAXONOMY.get("auxiliary_tags") or [])
            if isinstance(tag, str)
        ],
}

INTENT_ALIASES = {
    "training_objection": "objection_handling",
    "follow_up": "crm_follow_up",
}


def _load_domain_synonyms() -> Dict[str, List[str]]:
    try:
        with _DOMAIN_SYNONYMS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to load domain synonyms: {exc}")
        return {}

    if not isinstance(payload, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for canonical, aliases in payload.items():
        if not isinstance(canonical, str):
            continue
        if not isinstance(aliases, list):
            continue
        out[canonical.lower().strip()] = [
            alias.lower().strip()
            for alias in aliases
            if isinstance(alias, str) and alias.strip()
        ]
    return out


DOMAIN_SYNONYMS = _load_domain_synonyms()

_PHARMA_ENTITY_EXTRACTOR: Any = None

_EXPECTED_CONCEPTS_BY_INTENT: Dict[str, List[str]] = {
    "product_information_request": [
        "product_name",
        "indication",
        "active_ingredient",
        "benefit",
        "proof_reference",
    ],
    "dosage_question": [
        "product_name",
        "dosage",
        "patient_profile",
    ],
    "safety_question": [
        "product_name",
        "patient_profile",
        "contraindication",
        "adverse_event",
        "safety_flag",
    ],
    "objection_handling": [
        "objection_phrase",
        "proof_reference",
        "benefit",
    ],
    "training_simulation": [
        "scenario",
        "visit_format",
        "objection_phrase",
    ],
    "crm_follow_up": [
        "action_item",
        "next_visit",
        "crm",
    ],
    "competency_assessment": [
        "competency_level",
        "score",
        "feedback",
    ],
    "visit_format_request": [
        "visit_format",
        "time_constraint",
        "meeting_length",
    ],
    "sales_methodology_request": [
        "opening_permission",
        "discovery_sondage",
        "summary_reformulation",
        "argumentation",
        "closing_commitment",
    ],
    "general_greeting": [
        "greeting",
    ],
}


def _normalize_user_text(user_text: str) -> str:
    normalized = (user_text or "").strip()
    if not normalized:
        return ""

    for canonical, aliases in DOMAIN_SYNONYMS.items():
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            normalized = re.sub(pattern, canonical, normalized, flags=re.IGNORECASE)

    # Normalize whitespace artifacts after replacements.
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _use_entity_extractor_v2() -> bool:
    return os.getenv("ALIA_USE_ENTITY_EXTRACTOR_V2", "0").lower() in {"1", "true", "yes", "on"}


def _get_pharma_entity_extractor() -> Any:
    global _PHARMA_ENTITY_EXTRACTOR
    if _PHARMA_ENTITY_EXTRACTOR is not None:
        return _PHARMA_ENTITY_EXTRACTOR
    if not _use_entity_extractor_v2() or EntityExtractorV2 is None:
        return None
    try:
        _PHARMA_ENTITY_EXTRACTOR = EntityExtractorV2(use_spacy=False)
    except Exception as exc:
        logger.warning(f"Failed to initialize pharma entity extractor v2: {exc}")
        _PHARMA_ENTITY_EXTRACTOR = None
    return _PHARMA_ENTITY_EXTRACTOR


def _pharma_entity_map(user_text: str) -> Dict[str, List[str]]:
    extractor = _get_pharma_entity_extractor()
    if extractor is None:
        return {}

    try:
        extracted = extractor.extract_entities(user_text, use_fuzzy=True, use_ner=False)
    except Exception as exc:
        logger.warning(f"Pharma entity extraction failed: {exc}")
        return {}

    entity_map: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}

    mapping = {
        "product": "product_name",
        "molecule": "active_ingredient",
        "dosage": "dosage",
        "indication": "indication",
    }

    for source_type, target_type in mapping.items():
        values = extracted.get(source_type, []) if isinstance(extracted, dict) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if not isinstance(value, str) or not value.strip():
                continue
            normalized_value = normalize_entity_value(target_type, value)
            if normalized_value:
                entity_map.setdefault(target_type, []).append(normalized_value)

    for entity_type in entity_map:
        entity_map[entity_type] = _dedupe_preserve(entity_map[entity_type])[:10]

    return {key: value for key, value in entity_map.items() if value}


def _build_explainability(
    user_text: str,
    intent: str,
    entity_map: Dict[str, List[str]],
    secondary_tags: List[str],
    confidence: float,
) -> Dict[str, Any]:
    lower_text = user_text.lower()

    matched_keywords: List[str] = []
    for term in _INTENT_TOKEN_MAP.get(intent, []):
        normalized_term = term.lower().strip()
        if not normalized_term:
            continue
        if normalized_term in lower_text or _contains_token(lower_text, normalized_term):
            matched_keywords.append(normalized_term)

    matched_keywords.extend([value for values in entity_map.values() for value in values if value])
    matched_keywords.extend([tag for tag in secondary_tags if tag])

    influential_keywords = _dedupe_preserve([
        value for value in matched_keywords if value and len(value) > 1
    ])[:8]

    expected_concepts = _EXPECTED_CONCEPTS_BY_INTENT.get(intent, [])
    present_concepts = [entity_type for entity_type, values in entity_map.items() if values]
    missing_expected_concepts = [
        concept for concept in expected_concepts if concept not in present_concepts
    ]

    if intent == "general_greeting":
        why_class_was_chosen = (
            "Detected a greeting pattern and a short conversational opening, which matches the greeting class."
        )
    elif influential_keywords:
        why_class_was_chosen = (
            f"Classified as {intent} because the message contains evidence such as "
            f"{', '.join(influential_keywords[:4])}."
        )
    else:
        why_class_was_chosen = (
            f"Classified as {intent} based on the overall phrasing and routing heuristics, without strong lexical matches."
        )

    missing_text = (
        ", ".join(missing_expected_concepts)
        if missing_expected_concepts
        else "none"
    )
    reasoning = (
        f"Intent={intent}; confidence={confidence:.2f}. "
        f"Influential keywords: {', '.join(influential_keywords) if influential_keywords else 'none'}. "
        f"Present concepts: {', '.join(present_concepts) if present_concepts else 'none'}. "
        f"Missing expected concepts: {missing_text}."
    )

    return {
        "influential_keywords": influential_keywords,
        "why_class_was_chosen": why_class_was_chosen,
        "missing_expected_concepts": missing_expected_concepts,
        "reasoning": reasoning,
    }

_INTENT_TOKEN_MAP: Dict[str, List[str]] = {
    "general_greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "how are you",
        "pleasure to see",
        "nice to see",
        "great to see",
        "welcome",
        "salutations",
        "pleased",
        "delighted",
    ],
    "training_simulation": [
        "simulate",
        "role-play",
        "role play",
        "roleplay",
        "challenge me",
        "scenario",
        "train me",
        "training session",
        "simulated",
        "practice handling",
        "practice with",
        "lets practice",
        "let me practice",
        "practice a",
        "education program",
        "education programs",
        "reference materials",
        "simulate a",
        "demonstrate a difficult conversation",
        "difficult conversation",
        "when should reps start using this",
    ],
    "visit_format_request": [
        "flash visit",
        "standard visit",
        "deep visit",
        "approfondie",
        "visit flow",
        "one minute",
        "two minutes",
        "three minutes",
        "30 seconds",
        "60 seconds",
        "keep it short",
        "keep it brief",
    ],
    "dosage_question": [
        "dose",
        "dosage",
        "how much",
        "how often",
        "frequency",
        "mg",
        "route",
        "schedule",
        "posology",
        "hepatic dosing",
        "dosing for",
        "renal dose",
        "geriatric dosing",
        "dose adjustment",
        "washout period",
    ],
    "safety_question": [
        "side effect",
        "adverse",
        "contraindication",
        "contraindicated",
        "safe",
        "suitable for",
        "use in",
        "use during",
        "interaction",
        "my patient",
        "for this patient",
        "diagnose",
        "diagnosis",
        "tolerance",
        "elderly",
        "pregnan",
        "pregnancy",
        "breastfeeding",
        "breast feeding",
        "lactation",
        "off label",
        "off-label",
        "pediatric",
        "child",
        "high-risk interaction",
        "black box",
        "populations should avoid",
        "contraindication in",
        "avoid in",
        "pregnancy category",
        "can i use during",
        "qt prolongation",
        "photosensitivity",
        "hepatotoxicity",
        "nephrotoxicity",
        "teratogenic",
        "dialysis",
        "liver disease",
        "concurrently with other medications",
        "concurrently with",
    ],
    "objection_handling": [
        "not convinced",
        "not persuaded",
        "pas convaincu",
        "too expensive",
        "expensive",
        "cher",
        "habit",
        "do not have time",
        "don't have time",
        "no time",
        "before i believe",
        "need a source",
        "worry about safety",
        "worried",
        "tolerance concern",
        "i'm too busy",
        "i'm in a rush",
        "outside my budget",
        "price is too high",
        "i don't believe",
        "show me something",
        "published",
        "generic version",
        "want to see trial data first",
        "prefer the older brand",
    ],
    "sales_methodology_request": [
        "teach me",
        "teach opening",
        "teach discovery",
        "teach qare",
        "teach a-c-r-v",
        "teach closing",
        "teach argumentation",
        "opening with permission",
        "discovery questions",
        "reformulate",
        "reformulation",
        "synthesize",
        "synthese",
        "qare",
        "a-c-r-v",
        "argue with evidence",
        "closing commitment",
        "argumentation structure",
        "secure next steps",
        "instant zero",
        "sondage",
        "active listening",
        "benefit segmentation",
        "structure this visit",
        "walk me through",
        "objection handling techniques",
        "objection-handling method",
        "objection handling method",
        "engagement questions",
        "closing questions",
        "techniques for",
        "methodology for",
        "structure for",
        "approach for",
        "best practices for",
        "how should i",
        "how do i adjust",
        "how do i close and get engagement",
        "how do i prepare",
        "how do i listen",
        "listen actively",
        "help my sales team improve",
        "coaching approach",
        "how do physicians typically respond",
        "most common objection",
        "handle skeptical physicians",
    ],
    "crm_follow_up": [
        "crm",
        "follow-up",
        "follow up",
        "next visit",
        "relance",
        "plan",
        "checklist",
        "write in the crm",
        "what goes in the crm",
        "non-responder",
        "non responder",
        "reconnect next month",
        "discuss this with my team",
    ],
    "competency_assessment": [
        "competency",
        "competency level",
        "competency assessment",
        "debutant",
        "junior",
        "confirme",
        "expert",
        "expert-level",
        "expert level",
        "what level",
        "assess my level",
        "evaluate my competency",
        "my level",
        "advancement",
        "promotion",
        "how many simulations",
        "score my competency",
        "rate my",
        "am i ready",
        "ready to advance",
        "ready to progress",
        "feedback on my",
        "review my",
        "evaluate my",
        "assess my",
        "what's my weak point",
        "how do i get better",
        "visit left them uninterested",
    ],
    "product_information_request": [
        "product",
        "indication",
        "source",
        "guideline",
        "reference",
        "clinical trial",
        "trial data",
        "mechanism of action",
        "mechanism",
        "efficacy",
        "effectiveness",
        "real-world data",
        "real world",
        "how does it work",
        "pharmacology",
        "pharmacodynamics",
        "pharmacokinetics",
        "bioavailability",
        "formulation",
        "available",
        "approved",
        "published",
        "data",
        "show me clinical",
        "show me trial",
        "latest research",
        "case studies",
        "patient compliance",
    ],
}

_HARD_NEGATIVE_INTENT_OVERRIDES: List[tuple[str, str]] = [
    ("my patients prefer the older brand", "objection_handling"),
    ("they want to see trial data first", "objection_handling"),
    ("show me clinical evidence", "objection_handling"),
    ("i'm worried about adverse effects", "objection_handling"),
    ("what about adverse reactions", "objection_handling"),
    ("i use what i know", "objection_handling"),
    ("i need peer reviews first", "objection_handling"),
    ("what's your coaching approach", "sales_methodology_request"),
    ("how do physicians typically respond", "sales_methodology_request"),
    ("what's the most common objection", "sales_methodology_request"),
    ("how do i prioritize arguments", "sales_methodology_request"),
    ("teach me how to adapt to confirme doctors", "sales_methodology_request"),
    ("demonstrate a difficult conversation", "training_simulation"),
    ("when should reps start using this", "training_simulation"),
    ("document this visit interaction for crm", "crm_follow_up"),
    ("my visit left them uninterested", "competency_assessment"),
    ("geriatric considerations", "dosage_question"),
    ("what's the washout period", "dosage_question"),
    ("any contraindication for product", "safety_question"),
    ("list contraindications for", "safety_question"),
    ("can this be used concurrently with other medications", "safety_question"),
    ("any concerns with liver disease", "safety_question"),
]


def _intent_override(lower_text: str) -> str | None:
    for phrase, intent in _HARD_NEGATIVE_INTENT_OVERRIDES:
        if phrase in lower_text:
            return intent
    return None


def _match_count(lower_text: str, terms: List[str]) -> int:
    matches = 0
    for term in terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if " " not in normalized and normalized.isalnum() and len(normalized) <= 4:
            if _contains_token(lower_text, normalized):
                matches += 1
            continue
        if normalized in lower_text:
            matches += 1
    return matches


def _intent_scores(user_text: str) -> Dict[str, int]:
    lower = user_text.lower()
    scores = {
        intent: _match_count(lower, terms)
        for intent, terms in _INTENT_TOKEN_MAP.items()
    }

    if any(_contains_token(lower, token) for token in ["hello", "hi", "hey"]) and len(lower.split()) <= 6:
        scores["general_greeting"] += 2

    if "formulation available" in lower:
        scores["product_information_request"] += 2
        scores["safety_question"] = max(0, scores["safety_question"] - 1)

    if lower.startswith("tell me about") or lower.startswith("talk to me about"):
        scores["product_information_request"] += 2

    if "practice" in lower and any(term in lower for term in ["doctor", "physician", "scenario"]):
        scores["training_simulation"] += 2

    if "simulate" in lower and any(term in lower for term in ["doctor", "physician"]):
        scores["training_simulation"] += 2
        scores["competency_assessment"] = max(0, scores["competency_assessment"] - 1)

    if "close and get engagement" in lower:
        scores["sales_methodology_request"] += 2

    if any(term in lower for term in ["how do i adjust", "approach for", "techniques", "methodology"]) and any(
        level in lower for level in ["debutant", "junior", "confirme", "expert-level", "expert level"]
    ):
        scores["sales_methodology_request"] += 2
        scores["competency_assessment"] = max(0, scores["competency_assessment"] - 1)

    objection_override_terms = [
        "mixed reviews",
        "usual product",
        "no change needed",
        "i've always used",
        "not persuaded",
        "need more time",
        "cost per patient",
        "insurance may not cover",
        "review the data myself",
    ]
    if any(term in lower for term in objection_override_terms):
        scores["objection_handling"] += 2
        scores["product_information_request"] = max(0, scores["product_information_request"] - 1)
        scores["safety_question"] = max(0, scores["safety_question"] - 1)

    if "my concern is safety" in lower:
        scores["objection_handling"] += 2
        scores["safety_question"] = max(0, scores["safety_question"] - 1)

    return scores


def _coarse_route_from_scores(scores: Dict[str, int]) -> str:
    bucket_scores = {
        "greeting": scores.get("general_greeting", 0),
        "coaching": (
            scores.get("training_simulation", 0)
            + scores.get("visit_format_request", 0)
            + scores.get("sales_methodology_request", 0)
            + scores.get("competency_assessment", 0)
        ),
        "clinical": (
            scores.get("product_information_request", 0)
            + scores.get("dosage_question", 0)
            + scores.get("safety_question", 0)
        ),
        "objection": scores.get("objection_handling", 0),
        "crm": scores.get("crm_follow_up", 0),
    }

    ordered_buckets = ["coaching", "clinical", "objection", "crm", "greeting"]
    best_bucket = max(ordered_buckets, key=lambda bucket: (bucket_scores[bucket], -ordered_buckets.index(bucket)))
    return best_bucket if bucket_scores.get(best_bucket, 0) > 0 else "other"


def _fine_intent_from_route(route: str, scores: Dict[str, int]) -> str:
    if route == "greeting":
        return "general_greeting"

    if route == "crm":
        return "crm_follow_up"

    if route == "objection":
        return "objection_handling"

    if route == "clinical":
        ordered = ["safety_question", "dosage_question", "product_information_request"]
        return max(ordered, key=lambda intent: (scores.get(intent, 0), -ordered.index(intent)))

    if route == "coaching":
        ordered = [
            "training_simulation",
            "visit_format_request",
            "sales_methodology_request",
            "competency_assessment",
        ]
        return max(ordered, key=lambda intent: (scores.get(intent, 0), ordered.index(intent)))

    return "other"


def _rule_intent_router(user_text: str) -> tuple[str, float]:
    override_intent = _intent_override(user_text.lower())
    if override_intent is not None:
        return override_intent, 0.9

    scores = _intent_scores(user_text)
    route = _coarse_route_from_scores(scores)
    intent = _fine_intent_from_route(route, scores)
    best_score = max(scores.values()) if scores else 0

    if intent == "other" or best_score <= 0:
        return "other", 0.25
    if best_score >= 4:
        return intent, 0.85
    if best_score == 3:
        return intent, 0.72
    if best_score == 2:
        return intent, 0.58
    if best_score == 1:
        token_count = len(re.findall(r"\w+", user_text))
        domain_markers = [
            "product",
            "dosage",
            "contraindication",
            "show me",
            "flash visit",
            "price is too high",
            "teach",
            "simulate",
            "crm",
        ]
        if token_count >= 5 or any(marker in user_text.lower() for marker in domain_markers):
            return intent, 0.55
    return intent, 0.45


def _build_system_prompt() -> str:
    default_prompt = (
        "You are an NLP extraction engine for a pharmaceutical assistant. "
        "Return ONLY a valid JSON object with exactly these keys: "
        "intent, secondary_tags, entities, entity_map, topics, objections, action_items, "
        "safety_flags, rewritten_query, confidence. "
        "intent must be one of: {intents}. "
        "secondary_tags must be an array using these labels when applicable: {secondary_tags}. "
        "entities/topics/objections/action_items/safety_flags must be arrays of short strings. "
        "entity_map must be an object with keys from: {entity_types}; values are arrays of strings. "
        "safety_flags must use only: {safety_flags}. "
        "rewritten_query must be a concise semantic-search query. "
        "confidence must be a number between 0 and 1."
    )

    template = default_prompt
    try:
        if _PROMPT_PATH.exists():
            template = _PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Failed to read NLP prompt template: {exc}")

    return template.format(
        intents=", ".join(sorted(SUPPORTED_INTENTS)),
        secondary_tags=", ".join(sorted(SUPPORTED_SECONDARY_TAGS)) or "none",
        entity_types=", ".join(ENTITY_TYPES) or "none",
        safety_flags=", ".join(sorted(SUPPORTED_SAFETY_FLAGS)) or "none",
    )


def _groq_client():
    try:
        from groq import Groq
    except ImportError:
        return None

    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}

    text = raw_text.strip()

    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                text = cleaned
                break

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    snippet = text[start : end + 1]
    try:
        data = json.loads(snippet)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _safe_list(value: Any, limit: int = 10) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned)
    return out[:limit]


def _contains_token(text: str, token: str) -> bool:
    pattern = rf"\b{re.escape(token.lower())}\b"
    return re.search(pattern, text.lower()) is not None


def _normalize_entity_map(value: Any) -> Dict[str, List[str]]:
    entity_map: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}
    if not isinstance(value, dict):
        return entity_map

    for key, raw_values in value.items():
        if not isinstance(key, str):
            continue
        if key not in entity_map:
            continue
        entity_map[key] = [
            normalize_entity_value(key, item)
            for item in _safe_list(raw_values, limit=10)
            if normalize_entity_value(key, item)
        ]
    return entity_map


def _dedupe_preserve(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _merge_entity_maps(primary: Dict[str, List[str]], secondary: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}
    for entity_type in ENTITY_TYPES:
        merged_values = [
            normalize_entity_value(entity_type, item)
            for item in ((primary.get(entity_type) or []) + (secondary.get(entity_type) or []))
            if normalize_entity_value(entity_type, item)
        ]
        merged[entity_type] = _dedupe_preserve(merged_values)[:12]
    return merged


def _rule_entity_map(user_text: str) -> Dict[str, List[str]]:
    text = user_text.strip()
    lower = text.lower()
    out: Dict[str, List[str]] = {entity_type: [] for entity_type in ENTITY_TYPES}

    stop_tokens = {
        "Hello",
        "Doctor",
        "Can",
        "What",
        "How",
        "Give",
        "Any",
        "Provide",
        "Assess",
        "This",
    }

    explicit_product = re.findall(r"\bProduct\s+[A-Za-z0-9-]+\b", text)
    camel_case_product = re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9-]*\b", text)

    product_matches = explicit_product + camel_case_product
    for match in product_matches:
        first_token = match.split()[0]
        if first_token in stop_tokens:
            continue
        out.get("product_name", []).append(match)

    if any(token in lower for token in ["dosage", "dose", "mg", "twice daily", "once daily", "posology"]):
        out.get("dosage", []).append("dosage")

    patient_profile_terms = [
        "renal impairment",
        "renal",
        "adult",
        "adults",
        "pediatric",
        "child",
        "elderly",
        "pregnant",
        "pregnancy",
        "high-risk patient",
    ]
    for term in patient_profile_terms:
        if term in lower:
            out.get("patient_profile", []).append(term)

    if any(term in lower for term in ["flash", "standard", "deep", "approfondie", "one minute", "two minutes", "three minutes", "30 seconds", "60 seconds", "keep it short", "keep it brief"]):
        if "flash" in lower or "30 seconds" in lower or "60 seconds" in lower or "one minute" in lower or "two minutes" in lower or "three minutes" in lower or "keep it short" in lower or "keep it brief" in lower:
            out.get("visit_format", []).append("Flash")
        if "standard" in lower:
            out.get("visit_format", []).append("Standard")
        if "deep" in lower or "approfondie" in lower:
            out.get("visit_format", []).append("Approfondie")

    for term in ["indication", "hypertension", "diabetes", "asthma"]:
        if term in lower:
            out.get("indication", []).append(term)

    for term in ["active ingredient", "molecule", "ingredient"]:
        if term in lower:
            out.get("active_ingredient", []).append(term)

    for term in ["adherence", "convenience", "observance", "tolerance", "benefit"]:
        if term in lower:
            out.get("benefit", []).append(term)

    for term in ["contraindication", "interaction", "adverse", "side effect", "tolerance"]:
        if term in lower:
            out.get("adverse_event", []).append(term)

    if any(term in lower for term in ["how often", "frequency", "route", "schedule", "posology"]):
        out.get("dosage", []).append("dosage")

    if any(term in lower for term in ["proof", "study", "evidence", "guideline", "source"]):
        out.get("proof_reference", []).append("proof")

    for term in ["debutant", "junior", "confirme", "expert"]:
        if term in lower:
            out.get("competency_level", []).append(term)

    objection_markers = [
        "not convinced",
        "too expensive",
        "i have my habits",
        "pas convaincu",
        "cher",
        "do not have time",
        "no time",
        "need a source",
        "before i believe",
        "worry about safety",
        "tolerance concern",
        "don't have time",
        "too busy",
        "usual product",
        "no change",
        "always used this brand",
        "price is too high",
        "generics",
        "not persuaded",
        "safety concern",
        "worried about adverse effects",
    ]
    for marker in objection_markers:
        if marker in lower:
            out.get("objection_phrase", []).append(marker)

    if "clinical evidence" in lower:
        out.get("proof_reference", []).append("clinical evidence")
    if "evidence" in lower:
        out.get("proof_reference", []).append("evidence")
    if "data" in lower:
        out.get("proof_reference", []).append("data")

    if "frequency" in lower:
        out.get("dosage", []).append("frequency")
    if "route" in lower:
        out.get("dosage", []).append("route")
    if "schedule" in lower:
        out.get("dosage", []).append("schedule")

    if "indicated for" in lower:
        out.get("indication", []).append("indicated for")
    if "active molecule" in lower:
        out.get("active_ingredient", []).append("active molecule")
    if "benefits" in lower:
        out.get("benefit", []).append("benefits")
    if "advantage" in lower:
        out.get("benefit", []).append("advantage")

    if "contraindication" in lower:
        out.get("contraindication", []).append("contraindication")

    if "patient population" in lower:
        out.get("patient_profile", []).append("patient population")
    if "patient" in lower:
        out.get("patient_profile", []).append("patient")

    if "adverse effects" in lower:
        out.get("adverse_event", []).append("adverse effects")
    if "drug-drug interaction" in lower or "drug drug interaction" in lower:
        out.get("adverse_event", []).append("drug-drug interaction")

    for entity_type in ENTITY_TYPES:
        normalized_values = [
            normalize_entity_value(entity_type, item)
            for item in out.get(entity_type, [])
            if normalize_entity_value(entity_type, item)
        ]
        out[entity_type] = _dedupe_preserve(normalized_values)[:10]
    return out


def _flatten_entities(entity_map: Dict[str, List[str]], extracted_entities: List[str]) -> List[str]:
    flattened = list(extracted_entities)
    for values in entity_map.values():
        flattened.extend(values)
    return _dedupe_preserve([value for value in flattened if value])[:20]


def _infer_secondary_tags(user_text: str, intent: str) -> List[str]:
    text = user_text.lower()
    tags: List[str] = []

    # Visit format detection
    if any(token in text for token in ["flash", "30 seconds", "60 seconds", "one minute", "two minutes", "three minutes", "keep it short", "keep it brief", "quick", "1 min", "60 sec"]):
        tags.append("flash_visit")
    if any(token in text for token in ["standard", "2-4 min", "2 minutes", "standard visit", "2 to 4"]):
        tags.append("standard_visit")
    if any(token in text for token in ["approfondie", "deep", "5-8 min", "deep visit", "5 to 8"]):
        tags.append("deep_visit")

    # Objection and problem handling
    if any(token in text for token in ["objection", "not convinced", "pas convaincu", "too expensive", "cher", "do not have time", "no time", "habit", "worry", "concern", "expensive"]):
        tags.append("objection_handling")
    if any(token in text for token in ["follow-up", "follow up", "crm", "relance", "next visit", "checklist"]):
        tags.append("crm_followup")
    
    # Methodology steps
    if any(token in text for token in ["opening", "permission", "introduction", "instant zero", "prepare", "preparation", "before the visit", "adapt the message", "adapt by profile", "adapt "]):
        tags.append("opening_permission")
    if any(token in text for token in ["question", "discovery", "sondage", "listen", "listening", "active listening"]):
        tags.append("discovery_sondage")
    if any(token in text for token in ["summary", "synthese", "resume", "reformulate", "reformulation"]):
        tags.append("summary_reformulation")
    if any(token in text for token in ["argument", "benefit", "prove", "proof", "usage", "data", "evidence", "guideline", "counter", "address"]):
        tags.append("argumentation")
    if any(token in text for token in ["close", "closing", "closing commitment", "engagement", "commitment", "next step", "secure"]):
        tags.append("closing_commitment")

    # Proof/evidence needs
    if any(token in text for token in ["need a source", "before i believe", "proof", "source", "evidence", "guideline", "study", "peer"]):
        if intent == "objection_handling" or "before i believe" in text or "need a source" in text:
            tags.append("needs_proof")

    # Specific objection types
    if any(token in text for token in ["no time", "do not have time", "don't have time", "busy", "rush", "short on time", "not enough time"]):
        tags.append("no_time")
    if any(token in text for token in ["habit", "habits", "already use", "already have", "always", "usual"]):
        tags.append("habitual_use")
    if any(token in text for token in ["too expensive", "expensive", "cher", "price", "cost", "budget"]):
        tags.append("too_expensive")
    if any(token in text for token in ["not convinced", "not persuaded", "persuaded", "skeptical"]):
        tags.append("not_convinced")
    if any(token in text for token in ["safety concern", "safety", "worry about safety", "high-risk patient", "adverse", "tolerance"]):
        tags.append("safety_concern")
    if any(token in text for token in ["tolerance concern", "tolerance", "side effect", "adverse effect", "adverse reaction"]):
        tags.append("tolerance_concern")

    # Add intent-based tags
    if intent == "objection_handling" and "objection_handling" not in tags:
        tags.append("objection_handling")
    if intent == "training_simulation" and intent not in tags:
        tags.append("training_simulation")
    if intent == "visit_format_request" and "visit_format" not in tags:
        tags.append("visit_format_request")
    if "second_visit" in text or "relance" in text:
        tags.append("second_visit_cycle")

    return [tag for tag in tags if tag in SUPPORTED_SECONDARY_TAGS]


def _normalize_intent(intent: Any) -> str:
    if not isinstance(intent, str):
        return "other"
    normalized = INTENT_ALIASES.get(intent, intent)
    if normalized in SUPPORTED_INTENTS:
        return normalized
    return "other"


def _normalize_safety_flags(flags: Any, user_text: str) -> List[str]:
    normalized = [flag for flag in _safe_list(flags, limit=10) if flag in SUPPORTED_SAFETY_FLAGS]
    normalized.extend(detect_safety_flags(user_text))

    # Keep order stable while removing duplicates.
    deduped: List[str] = []
    for flag in normalized:
        if flag not in deduped:
            deduped.append(flag)
    return deduped[:8]


def _fallback_analysis(user_text: str) -> Dict[str, Any]:
    normalized_text = _normalize_user_text(user_text)
    detected_language = detect_language(user_text)
    intent, confidence = _rule_intent_router(normalized_text)
    needs_clarification = confidence < 0.5

    secondary_tags = _infer_secondary_tags(normalized_text, intent)
    if "hello" in normalized_text.lower() and intent != "general_greeting":
        if "general_greeting" in SUPPORTED_SECONDARY_TAGS:
            secondary_tags.append("general_greeting")
        if "mixed_intent" in SUPPORTED_SECONDARY_TAGS:
            secondary_tags.append("mixed_intent")

    secondary_tags = _dedupe_preserve([tag for tag in secondary_tags if tag in SUPPORTED_SECONDARY_TAGS])
    rule_entity_map = _rule_entity_map(normalized_text)
    pharma_entity_map = _pharma_entity_map(normalized_text)
    merged_entity_map = _merge_entity_maps(rule_entity_map, pharma_entity_map)
    merged_entities = _flatten_entities(merged_entity_map, [])
    explainability = _build_explainability(
        user_text=normalized_text,
        intent=intent,
        entity_map=merged_entity_map,
        secondary_tags=secondary_tags,
        confidence=confidence,
    )

    return {
        "intent": intent,
        "secondary_tags": secondary_tags,
        "entities": merged_entities,
        "entity_map": merged_entity_map,
        "explainability": explainability,
        "topics": [],
        "objections": [],
        "action_items": ["needs_intent_clarification"] if needs_clarification else [],
        "safety_flags": _normalize_safety_flags([], normalized_text),
        "rewritten_query": normalized_text,
        "confidence": confidence,
        "language": detected_language,
        "taxonomy_version": "v1",
    }


def analyze_message_nlp(
    user_text: str,
    history: List[Dict[str, Any]] | None = None,
    mode: str = "physician_portal",
) -> Dict[str, Any]:
    user_text = (user_text or "").strip()
    if not user_text:
        return _fallback_analysis("")
    normalized_text = _normalize_user_text(user_text)
    detected_language = detect_language(user_text)

    history = history or []
    history_tail = history[-6:]
    history_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history_tail
    )

    client = _groq_client()
    if client is None:
        return _fallback_analysis(user_text)

    system_prompt = _build_system_prompt()

    user_prompt = (
        f"Mode: {mode}\n"
        "Recent conversation:\n"
        f"{history_text if history_text else '(none)'}\n\n"
        f"Current user message:\n{user_text}"
    )

    try:
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = completion.choices[0].message.content if completion.choices else ""
        parsed = _extract_json_object(raw or "")

        llm_intent = _normalize_intent(parsed.get("intent", "other"))

        confidence = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        rule_intent, rule_confidence = _rule_intent_router(normalized_text)
        intent = llm_intent

        if (intent == "other" or confidence < 0.55) and rule_intent != "other":
            intent = rule_intent
            confidence = max(confidence, rule_confidence)
        elif intent != rule_intent and rule_intent != "other" and confidence < 0.7:
            confidence = max(0.45, min(confidence, 0.6))

        rewritten_query = parsed.get("rewritten_query")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            rewritten_query = normalized_text

        secondary_tags = [
            tag for tag in _safe_list(parsed.get("secondary_tags"), limit=12)
            if tag in SUPPORTED_SECONDARY_TAGS
        ]
        for inferred in _infer_secondary_tags(normalized_text, intent):
            if inferred not in secondary_tags:
                secondary_tags.append(inferred)

        if any(_contains_token(normalized_text, token) for token in ["hello", "hi", "hey"]) and intent != "general_greeting":
            if "general_greeting" in SUPPORTED_SECONDARY_TAGS and "general_greeting" not in secondary_tags:
                secondary_tags.append("general_greeting")
            if "mixed_intent" in SUPPORTED_SECONDARY_TAGS and "mixed_intent" not in secondary_tags:
                secondary_tags.append("mixed_intent")

        secondary_tags = _dedupe_preserve([tag for tag in secondary_tags if tag in SUPPORTED_SECONDARY_TAGS])

        parsed_entity_map = _normalize_entity_map(parsed.get("entity_map"))
        rule_entity_map = _rule_entity_map(normalized_text)
        pharma_entity_map = _pharma_entity_map(normalized_text)
        merged_entity_map = _merge_entity_maps(parsed_entity_map, rule_entity_map)
        merged_entity_map = _merge_entity_maps(merged_entity_map, pharma_entity_map)
        merged_entities = _flatten_entities(
            merged_entity_map,
            _safe_list(parsed.get("entities"), limit=12),
        )
        explainability = _build_explainability(
            user_text=normalized_text,
            intent=intent,
            entity_map=merged_entity_map,
            secondary_tags=secondary_tags,
            confidence=confidence,
        )

        action_items = [item for item in _safe_list(parsed.get("action_items"), limit=8) if item != "needs_intent_clarification"]
        if confidence < 0.5:
            action_items.append("needs_intent_clarification")

        return {
            "intent": intent,
            "secondary_tags": secondary_tags,
            "entities": merged_entities,
            "entity_map": merged_entity_map,
            "explainability": explainability,
            "topics": _safe_list(parsed.get("topics"), limit=10),
            "objections": _safe_list(parsed.get("objections"), limit=8),
            "action_items": action_items,
            "safety_flags": _normalize_safety_flags(parsed.get("safety_flags"), normalized_text),
            "rewritten_query": rewritten_query.strip(),
            "confidence": confidence,
            "language": detected_language,
            "taxonomy_version": "v1",
        }
    except Exception as exc:
        logger.warning(f"NLP analysis failed, using fallback: {exc}")
        return _fallback_analysis(user_text)
