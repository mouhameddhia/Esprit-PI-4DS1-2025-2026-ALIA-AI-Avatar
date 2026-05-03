"""Deterministic rule-based entity extraction."""

import re
from typing import Dict, List

from alia_nlp.data.taxonomy.loader import ENTITY_TYPES

_STOP_TOKENS = {"Hello", "Doctor", "Can", "What", "How", "Give", "Any", "Provide", "This"}

_HIGH_RISK_POPULATIONS = [
    "pregnan", "renal impairment", "renal failure", "renal disease", "renal",
    "pediatric", "child", "elderly", "geriatric", "hepatic", "liver disease",
    "cardiac", "heart failure", "nursing", "breastfeed", "lactation", "dialysis",
    "neonatal", "infant", "trimester",
]


def extract_rules(user_text: str) -> Dict[str, List[str]]:
    text = user_text.strip()
    lower = text.lower()
    out: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}

    # Product names — "Product X" prefix or CamelCase brand
    for m in re.findall(r"\bProduct\s+[A-Za-z0-9-]+\b", text):
        if m.split()[0] not in _STOP_TOKENS:
            out["product_name"].append(m)
    for m in re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9-]*\b", text):
        if m.split()[0] not in _STOP_TOKENS:
            out["product_name"].append(m)

    # Dosage — numeric amounts (most reliable)
    out["dosage"].extend(re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|g|tablets?|capsules?|doses?)\b", lower
    ))
    if not out["dosage"] and any(t in lower for t in [
        "dosage", "how much", "how often", "frequency", "posology",
        "twice daily", "once daily", "route", "schedule",
    ]):
        out["dosage"].append("dosage")

    # Patient profile
    for term in _HIGH_RISK_POPULATIONS:
        if term in lower:
            out["patient_profile"].append(term)

    # Visit format
    if any(t in lower for t in ["flash", "30 seconds", "60 seconds", "one minute", "keep it short"]):
        out["visit_format"].append("Flash")
    if "standard" in lower:
        out["visit_format"].append("Standard")
    if "deep" in lower or "approfondie" in lower:
        out["visit_format"].append("Approfondie")

    # Indication
    for term in ["indication", "hypertension", "diabetes", "asthma", "dyslipidemia",
                 "atrial fibrillation", "heart failure", "hypercholesterolemia"]:
        if term in lower:
            out["indication"].append(term)

    # Active ingredient
    for term in ["active ingredient", "molecule", "active substance"]:
        if term in lower:
            out["active_ingredient"].append(term)

    # Benefit
    for term in ["adherence", "convenience", "observance", "efficacy", "effectiveness", "benefit"]:
        if term in lower:
            out["benefit"].append(term)

    # Adverse events (includes implicit toxicity)
    for term in [
        "contraindication", "interaction", "adverse", "side effect",
        "qt prolongation", "torsades", "hepatotox", "nephrotox", "cardiotox",
        "photosensitivity", "rhabdomyolysis", "black box", "boxed warning",
        "teratogen", "teratogenic", "embryotox", "birth defect", "serotonin syndrome",
    ]:
        if term in lower:
            out["adverse_event"].append(term)

    # Proof / reference
    for term in ["proof", "study", "evidence", "guideline", "source", "trial",
                 "meta-analysis", "published", "data"]:
        if term in lower:
            out["proof_reference"].append(term)

    # Competency level
    for term in ["debutant", "junior", "confirme", "expert"]:
        if term in lower:
            out["competency_level"].append(term)

    # Objection phrases
    for marker in [
        "not convinced", "too expensive", "no time", "do not have time",
        "need a source", "before i believe", "worry about safety",
        "tolerance concern", "pas convaincu",
    ]:
        if marker in lower:
            out["objection_phrase"].append(marker)

    # Deduplicate
    for t in ENTITY_TYPES:
        seen: list = []
        for v in out.get(t, []):
            if v not in seen:
                seen.append(v)
        out[t] = seen[:10]

    return out
