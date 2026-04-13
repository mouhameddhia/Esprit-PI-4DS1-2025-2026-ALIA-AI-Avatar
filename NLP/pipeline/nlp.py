"""NLP analysis utilities for intent/entity extraction and retrieval query rewriting."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "taxonomy" / "nlp_taxonomy.json"
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "nlp_extraction_system_prompt.txt"


def _load_taxonomy() -> Dict[str, Any]:
    try:
        with _TAXONOMY_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to load NLP taxonomy: {exc}")
        return {
            "intents": [
                "product_information_request",
                "dosage_question",
                "safety_question",
                "objection_handling",
                "training_simulation",
                "crm_follow_up",
                "competency_assessment",
                "visit_format_request",
                "sales_methodology_request",
                "general_greeting",
                "other",
            ],
            "visit_phases": [],
            "visit_formats": [],
            "objection_types": [],
            "safety_flags": ["patient_specific_advice_request", "diagnosis_request"],
            "entity_types": [],
        }


TAXONOMY = _load_taxonomy()
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
        entity_map[key] = _safe_list(raw_values, limit=10)
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
        merged[entity_type] = _dedupe_preserve((primary.get(entity_type) or []) + (secondary.get(entity_type) or []))[:12]
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
    ]
    for marker in objection_markers:
        if marker in lower:
            out.get("objection_phrase", []).append(marker)

    for entity_type in ENTITY_TYPES:
        out[entity_type] = _dedupe_preserve(out.get(entity_type, []))[:10]
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
    text = user_text.lower()
    # Only flag patient-specific advice when explicitly requesting evaluation for a specific patient/scenario
    if any(term in text for term in ["my patient", "for this patient", "is product", "safe for", "suitable for", "use", "given"]) and any(term in text for term in ["pregnan", "renal impairment", "pediatric", "child", "elderly"]):
        if "patient_specific_advice_request" in SUPPORTED_SAFETY_FLAGS:
            normalized.append("patient_specific_advice_request")
    if "diagnose" in text or "diagnosis" in text:
        if "diagnosis_request" in SUPPORTED_SAFETY_FLAGS:
            normalized.append("diagnosis_request")
    if ("off label" in text or "off-label" in text) and "request" not in text.lower():
        if "off_label_request" in SUPPORTED_SAFETY_FLAGS:
            normalized.append("off_label_request")
    if "interaction" in text and not any(term in text for term in ["how often", "dose", "schedule", "posology"]):
        if any(term in text for term in ["current", "high-risk", "high risk", "meds", "treatment", "medicine", "drug"]):
            if "high_risk_interaction" in SUPPORTED_SAFETY_FLAGS:
                normalized.append("high_risk_interaction")
    # Keep order stable while removing duplicates.
    deduped: List[str] = []
    for flag in normalized:
        if flag not in deduped:
            deduped.append(flag)
    return deduped[:8]


def _fallback_analysis(user_text: str) -> Dict[str, Any]:
    lower = user_text.lower()
    safety_flags: List[str] = []
    if "my patient" in lower or "for this patient" in lower:
        safety_flags.append("patient_specific_advice_request")
    if "diagnose" in lower or "diagnosis" in lower:
        safety_flags.append("diagnosis_request")
    if any(term in lower for term in ["pregnancy", "pregnant", "breastfeed", "breastfeeding", "lactation", "category", "use during", "pregnant women", "nursing mother"]):
        if "patient_specific_advice_request" not in safety_flags:
            safety_flags.append("patient_specific_advice_request")
    if any(term in lower for term in ["renal disease", "hepatic disease", "cardiac disease"]) and any(term in lower for term in [" in ", " for ", "dosage", " condition"]):
        if "patient_specific_advice_request" not in safety_flags:
            safety_flags.append("patient_specific_advice_request")

    is_greeting = any(_contains_token(lower, token) for token in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "how are you", "pleasure", "nice to see", "great to see", "welcome", "salutations", "pleased", "delighted"])
    is_dosage = any(token in lower for token in ["dose", "dosage", "how much", "how often", "frequency", "mg", "route", "schedule", "posology", "hepatic dosing", "dosing for", "hepatic", "cirrhosis", "renal dose", "geriatric", "geriatric dosing"]) and not any(token in lower for token in ["relance", "follow-up", "crm"])
    is_safety = any(
        token in lower
        for token in [
            "side effect",
            "adverse",
            "contraindication",
            "contraindicated",
            "safe",
            "suitable for",
            "use in",
            "use during",
            "renal",
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
            "adverse reaction",
            "adverse effects",
            "safety concern",
            "high-risk interaction",
            "black box",
            "populations should avoid",
            "populations should",
            "contraindication in",
            "avoid in",
            "pregnancy category",
            "category for pregnancy",
            "can i use during",
            "use during pregnancy",
            "use during breastfeed",
            "qt prolongation",
            "photosensitivity",
            "hepatotoxicity",
            "nephrotoxicity",
        ]
    ) and "formulation available" not in lower
    is_objection = any(
        token in lower
        for token in [
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
            "need peer",
            "worry about safety",
            "worry about",
            "worried",
            "tolerance concern",
            "i'm too busy",
            "i'm in a rush",
            "in a rush",
            "use what i know",
            "always used",
            "i've always",
            "usual",
            "outside my budget",
            "budget",
            "price is too high",
            "adverse reaction",
            "concern",
            "show me clinical",
            "need a source",
            "how to handle",
            "dealing with",
            "overcoming",
            "don't believe",
            "i don't believe",
            "show me something",
            "show me",
            "publish",
            "published",
        ]
    )
    is_training = any(token in lower for token in ["simulate", "role-play", "role play", "challenge me", "scenario", "train me", "training session", "simulated", "roleplay", "practice handling", "practice with", "lets practice", "let me practice"]) and "how many" not in lower
    is_visit_format = any(token in lower for token in ["flash visit", "standard visit", "deep visit", "approfondie", "visit flow", "one minute", "two minutes", "three minutes", "30 seconds", "60 seconds", "keep it short", "keep it brief"])
    is_methodology = any(
        token in lower
        for token in [
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
            "closing and get engagement",
            "closing commitment",
            "argumentation structure",
            "secure next steps",
            "instant zero",
            "sondage",
            "active listening",
            "benefit segmentation",
            "prioritize argument",
            "structure this visit",
            "structure this",
            "6 steps",
            "how do i prepare",
            "how do i listen",
            "how do i reformulate",
            "how do i handle",
            "how do i argue",
            "how do i close",
            "walk me through",
            "explain",
            "objection-handling method",
            "objection handling method",
            "engagement questions",
            "closing questions",
            "techniques for",
            "methodology for",
            "structure for",
            "approach for",
            "best practices for",
            "objection handling",
            "how do you",
            "how should i",
        ]
    )
    is_crm = any(token in lower for token in ["crm", "follow-up", "follow up", "next visit", "relance", "plan", "checklist", "write in the crm", "what goes in the crm"])
    is_competency = any(token in lower for token in ["competency", "competency level", "competency assessment", "what level", "what competency", "debutant", "junior", "confirme", "expert", "assess my level", "evaluate my competency", "my level", "advancement", "promotion", "how many simulations", "requirements for", "score my competency", "rate my", "am i ready", "evaluate", "assess", "performance", "ready to advance", "ready to progress", "feedback on my", "review my", "evaluate my", "assess my"])
    is_product_info = any(token in lower for token in ["product", "indication", "evidence", "source", "guideline", "reference", "clinical trial", "trial data", "mechanism of action", "mechanism", "mechanism of", "efficacy", "efficacy data", "efficacy study", "effectiveness", "real-world data", "real world", "how does it work", "how it works", "what is the", "pharmacology", "pharmacodynamics", "pharmacokinetics", "bioavailability", "formulation", "available", "FDA approved", "approved"])


    intent = "other"
    if is_greeting:
        intent = "general_greeting"
    elif is_training:
        intent = "training_simulation"
    elif is_visit_format:
        intent = "visit_format_request"
    elif is_dosage:
        intent = "dosage_question"
    elif is_competency:
        intent = "competency_assessment"
    elif is_methodology:
        intent = "sales_methodology_request"
    elif is_crm:
        intent = "crm_follow_up"
    elif is_objection:
        intent = "objection_handling"
    elif is_safety:
        intent = "safety_question"
    elif is_product_info:
        intent = "product_information_request"

    secondary_tags = _infer_secondary_tags(user_text, intent)
    if is_greeting and intent != "general_greeting":
        if "general_greeting" in SUPPORTED_SECONDARY_TAGS:
            secondary_tags.append("general_greeting")
        if "mixed_intent" in SUPPORTED_SECONDARY_TAGS:
            secondary_tags.append("mixed_intent")

    secondary_tags = _dedupe_preserve([tag for tag in secondary_tags if tag in SUPPORTED_SECONDARY_TAGS])
    rule_entity_map = _rule_entity_map(user_text)
    merged_entities = _flatten_entities(rule_entity_map, [])

    return {
        "intent": intent,
        "secondary_tags": secondary_tags,
        "entities": merged_entities,
        "entity_map": rule_entity_map,
        "topics": [],
        "objections": [],
        "action_items": [],
        "safety_flags": _normalize_safety_flags(safety_flags, user_text),
        "rewritten_query": user_text.strip(),
        "confidence": 0.4,
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

        intent = _normalize_intent(parsed.get("intent", "other"))

        confidence = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        rewritten_query = parsed.get("rewritten_query")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            rewritten_query = user_text

        secondary_tags = [
            tag for tag in _safe_list(parsed.get("secondary_tags"), limit=12)
            if tag in SUPPORTED_SECONDARY_TAGS
        ]
        for inferred in _infer_secondary_tags(user_text, intent):
            if inferred not in secondary_tags:
                secondary_tags.append(inferred)

        if any(_contains_token(user_text, token) for token in ["hello", "hi", "hey"]) and intent != "general_greeting":
            if "general_greeting" in SUPPORTED_SECONDARY_TAGS and "general_greeting" not in secondary_tags:
                secondary_tags.append("general_greeting")
            if "mixed_intent" in SUPPORTED_SECONDARY_TAGS and "mixed_intent" not in secondary_tags:
                secondary_tags.append("mixed_intent")

        secondary_tags = _dedupe_preserve([tag for tag in secondary_tags if tag in SUPPORTED_SECONDARY_TAGS])

        parsed_entity_map = _normalize_entity_map(parsed.get("entity_map"))
        rule_entity_map = _rule_entity_map(user_text)
        merged_entity_map = _merge_entity_maps(parsed_entity_map, rule_entity_map)
        merged_entities = _flatten_entities(
            merged_entity_map,
            _safe_list(parsed.get("entities"), limit=12),
        )

        return {
            "intent": intent,
            "secondary_tags": secondary_tags,
            "entities": merged_entities,
            "entity_map": merged_entity_map,
            "topics": _safe_list(parsed.get("topics"), limit=10),
            "objections": _safe_list(parsed.get("objections"), limit=8),
            "action_items": _safe_list(parsed.get("action_items"), limit=8),
            "safety_flags": _normalize_safety_flags(parsed.get("safety_flags"), user_text),
            "rewritten_query": rewritten_query.strip(),
            "confidence": confidence,
            "taxonomy_version": "v1",
        }
    except Exception as exc:
        logger.warning(f"NLP analysis failed, using fallback: {exc}")
        return _fallback_analysis(user_text)
