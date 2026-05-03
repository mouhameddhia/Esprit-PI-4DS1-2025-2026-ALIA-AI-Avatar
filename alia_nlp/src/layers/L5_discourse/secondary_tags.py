"""Secondary tag inference from text and intent."""

from typing import Any, List

from alia_nlp.data.taxonomy.loader import SUPPORTED_SECONDARY_TAGS


def _has(text: str, tokens: list) -> bool:
    return any(t in text for t in tokens)


def infer(user_text: str, intent: str, llm_tags: Any) -> List[str]:
    """Merge LLM tags with rule-inferred tags; deduplicate."""
    text = user_text.lower()

    inferred: List[str] = []

    # Visit formats
    if _has(text, ["flash", "30 seconds", "60 seconds", "one minute", "keep it short", "keep it brief", "1 min"]):
        inferred.append("flash_visit")
    if _has(text, ["standard", "2 minutes", "standard visit", "2-4 min"]):
        inferred.append("standard_visit")
    if _has(text, ["approfondie", "deep", "5-8 min", "deep visit"]):
        inferred.append("deep_visit")

    # Visit phases
    if _has(text, ["opening", "permission", "introduction", "instant zero", "prepare", "before the visit"]):
        inferred.append("opening_permission")
    if _has(text, ["question", "discovery", "sondage", "listen", "active listening"]):
        inferred.append("discovery_sondage")
    if _has(text, ["summary", "synthese", "resume", "reformulate", "reformulation"]):
        inferred.append("summary_reformulation")
    if _has(text, ["argument", "prove", "proof", "evidence", "guideline", "counter"]):
        inferred.append("argumentation")
    if _has(text, ["close", "closing", "commitment", "engagement", "next step", "secure"]):
        inferred.append("closing_commitment")
    if _has(text, ["follow-up", "follow up", "crm", "relance", "next visit", "checklist"]):
        inferred.append("crm_followup")
    if "second_visit" in text or "relance" in text:
        inferred.append("second_visit_cycle")

    # Objection types
    if _has(text, ["no time", "do not have time", "don't have time", "busy", "rush"]):
        inferred.append("no_time")
    if _has(text, ["habit", "habits", "already use", "always", "usual"]):
        inferred.append("habitual_use")
    if _has(text, ["too expensive", "expensive", "cher", "price", "cost", "budget"]):
        inferred.append("too_expensive")
    if _has(text, ["not convinced", "not persuaded", "skeptical", "pas convaincu"]):
        inferred.append("not_convinced")
    if _has(text, ["safety concern", "worry about safety", "adverse", "tolerance"]):
        inferred.append("safety_concern")
    if _has(text, ["tolerance concern", "side effect", "adverse reaction"]):
        inferred.append("tolerance_concern")
    if _has(text, ["need a source", "before i believe", "need proof", "need peer"]):
        if intent == "objection_handling" or "need a source" in text:
            inferred.append("needs_proof")

    # Intent-driven tags
    if intent == "objection_handling" and "objection_handling" not in inferred:
        inferred.append("objection_handling")
    if intent == "training_simulation" and "training_simulation" not in inferred:
        inferred.append("training_simulation")

    # Greeting mixed intent
    if any(w in text for w in ["hello", "hi", "hey", "bonjour"]) and intent != "general_greeting":
        inferred.extend(["general_greeting", "mixed_intent"])

    # Merge with LLM tags
    if isinstance(llm_tags, list):
        for tag in llm_tags:
            if isinstance(tag, str) and tag.strip() and tag not in inferred:
                inferred.append(tag.strip())

    # Filter to taxonomy and deduplicate
    seen: list = []
    for tag in inferred:
        if tag in SUPPORTED_SECONDARY_TAGS and tag not in seen:
            seen.append(tag)
    return seen
