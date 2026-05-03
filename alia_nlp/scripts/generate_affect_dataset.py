"""
Synthetic affect dataset generator for L7 fine-tuning.

Generates labeled training examples via Groq LLM across all affect states,
both languages (en, fr), and both modes (medrep_training, physician_portal).

Output: alia_nlp/data/affect_training.jsonl
        alia_nlp/data/affect_validation.jsonl  (20% hold-out)

Usage:
    python -m alia_nlp.scripts.generate_affect_dataset
    python -m alia_nlp.scripts.generate_affect_dataset --n-per-scenario 15 --output-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from alia_nlp.utils.groq_client import get_groq_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Scenario definitions ─────────────────────────────────────────────────────
# Each entry: (language, mode, intent_context, affect_labels, human_description)
SCENARIOS: List[Dict[str, Any]] = [
    # ── MEDREP / English ─────────────────────────────────────────────────────
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — price objection",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "highly_engaged"},
        "description": "a confident, assertive medical rep handling a 'too expensive' objection with strong clinical arguments",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — price objection",
        "affect": {"rep_confidence": "low", "frustration_signal": True, "engagement_level": "passive"},
        "description": "a frustrated, uncertain rep struggling with a price objection, losing confidence",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — 'no time' objection",
        "affect": {"rep_confidence": "medium", "frustration_signal": False, "engagement_level": "active"},
        "description": "a rep with moderate confidence handling a physician's time objection",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — argumentation phase",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "highly_engaged"},
        "description": "a highly engaged rep asking for a harder role-play challenge after succeeding",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — opening permission",
        "affect": {"rep_confidence": "low", "frustration_signal": False, "engagement_level": "passive"},
        "description": "a passive, uncertain rep giving a minimal attempt at an opening statement",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "sales_methodology_request — closing technique",
        "affect": {"rep_confidence": "medium", "frustration_signal": False, "engagement_level": "active"},
        "description": "a rep actively asking for help with closing techniques mid-simulation",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "competency_assessment — asking for level feedback",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "active"},
        "description": "a confident rep requesting competency evaluation and promotion criteria",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — safety concern objection",
        "affect": {"rep_confidence": "low", "frustration_signal": True, "engagement_level": "passive"},
        "description": "a rep who does not know how to handle a safety objection, clearly lost",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "crm_follow_up — planning next visit",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "active"},
        "description": "a rep confidently summarizing CRM notes and planning the next visit",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "product_information_request — clinical benefits",
        "affect": {"rep_confidence": "medium", "frustration_signal": False, "engagement_level": "highly_engaged"},
        "description": "a rep actively studying product clinical benefits to strengthen their argumentation",
    },
    # ── MEDREP / French ──────────────────────────────────────────────────────
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection prix",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "highly_engaged"},
        "description": "un délégué médical très confiant qui gère une objection prix avec des arguments cliniques solides",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection prix",
        "affect": {"rep_confidence": "low", "frustration_signal": True, "engagement_level": "passive"},
        "description": "un délégué frustré et incertain face à une objection prix, qui perd confiance",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection 'pas le temps'",
        "affect": {"rep_confidence": "medium", "frustration_signal": False, "engagement_level": "active"},
        "description": "un délégué avec une confiance modérée gérant l'objection de manque de temps du médecin",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "training_simulation — phase d'argumentation",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "highly_engaged"},
        "description": "un délégué très engagé qui demande un scénario plus difficile après avoir réussi",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "training_simulation — prise de contact",
        "affect": {"rep_confidence": "low", "frustration_signal": False, "engagement_level": "passive"},
        "description": "un délégué passif et peu sûr de lui qui fait un effort minimal à l'ouverture",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "sales_methodology_request — technique de closing",
        "affect": {"rep_confidence": "medium", "frustration_signal": False, "engagement_level": "active"},
        "description": "un délégué actif qui demande de l'aide sur les techniques de closing",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection sécurité",
        "affect": {"rep_confidence": "low", "frustration_signal": True, "engagement_level": "passive"},
        "description": "un délégué perdu qui ne sait pas comment répondre à une objection de sécurité",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "crm_follow_up — planification visite suivante",
        "affect": {"rep_confidence": "high", "frustration_signal": False, "engagement_level": "active"},
        "description": "un délégué confiant qui résume ses notes CRM et planifie la prochaine visite",
    },
    # ── PHYSICIAN / English ───────────────────────────────────────────────────
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "safety_question — drug interaction",
        "affect": {"query_urgency": "urgent", "frustration_signal": True},
        "description": "a physician asking urgently about a dangerous drug interaction for an immediate patient situation",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "dosage_question — renal impairment dosing",
        "affect": {"query_urgency": "routine", "frustration_signal": False},
        "description": "a physician calmly asking about dosing adjustment for a renal patient",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "safety_question — teratogenicity",
        "affect": {"query_urgency": "elevated", "frustration_signal": False},
        "description": "a physician expressing concern about prescribing to a potentially pregnant patient",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "product_information_request — mechanism of action",
        "affect": {"query_urgency": "routine", "frustration_signal": False},
        "description": "a physician asking a straightforward clinical question about the mechanism of action",
    },
    # ── PHYSICIAN / French ─────────────────────────────────────────────────────
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "safety_question — interaction médicamenteuse",
        "affect": {"query_urgency": "urgent", "frustration_signal": True},
        "description": "un médecin posant une question urgente sur une interaction médicamenteuse dangereuse",
    },
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "dosage_question — insuffisance rénale",
        "affect": {"query_urgency": "routine", "frustration_signal": False},
        "description": "un médecin posant calmement une question sur l'ajustement de dose en cas d'insuffisance rénale",
    },
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "safety_question — tératogénicité",
        "affect": {"query_urgency": "elevated", "frustration_signal": False},
        "description": "un médecin exprimant une préoccupation sur la prescription à une patiente potentiellement enceinte",
    },
]

# ── Generation prompt ────────────────────────────────────────────────────────
_GEN_SYSTEM = (
    "You are a pharmaceutical training data generator. "
    "Generate realistic, diverse, natural-sounding messages. "
    "Return ONLY a valid JSON array of strings — no extra keys, no explanation."
)

_GEN_MEDREP_TEMPLATE = """\
Generate {n} realistic messages a medical representative might send during a \
pharmaceutical sales training simulation with an AI assistant.

Context:
- Language: {language} (write entirely in {language})
- Training scenario: {intent_context}
- Product context: Cardivex (amlodipine-based cardiovascular product, 5mg and 10mg)
- Affect state to portray: {description}

Requirements:
- Between 8 and 60 words per message
- Vary length, phrasing, and structure naturally across messages
- Messages can be mid-simulation, requests for feedback, or practice attempts
- Feel authentic — include realistic hesitations, domain vocabulary, or confidence as appropriate
- Do NOT label or explain the affect — just write the message as the rep would

Return ONLY a JSON array: ["message1", "message2", ...]"""

_GEN_PHYSICIAN_TEMPLATE = """\
Generate {n} realistic queries a healthcare professional might send to a \
pharmaceutical digital assistant.

Context:
- Language: {language} (write entirely in {language})
- Clinical scenario: {intent_context}
- Product context: Cardivex (amlodipine-based cardiovascular product)
- Tone to portray: {description}

Requirements:
- Between 5 and 40 words per message
- Vary phrasing and clinical vocabulary naturally
- Messages should feel like real clinical queries (terse, specific, professional)
- Do NOT label or explain the tone — write the query as the physician would

Return ONLY a JSON array: ["query1", "query2", ...]"""


def _build_prompt(scenario: Dict[str, Any], n: int) -> str:
    template = _GEN_PHYSICIAN_TEMPLATE if scenario["mode"] == "physician_portal" else _GEN_MEDREP_TEMPLATE
    return template.format(
        n=n,
        language="English" if scenario["language"] == "en" else "French",
        intent_context=scenario["intent_context"],
        description=scenario["description"],
    )


def _generate_batch(client, scenario: Dict[str, Any], n: int) -> List[str]:
    prompt = _build_prompt(scenario, n)
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _GEN_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=2048,
        )
        raw = completion.choices[0].message.content or "[]"
        parsed = json.loads(raw)
        # Handle both {"messages": [...]} and direct array (wrapped by json_object mode)
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, str) and m.strip()]
        for v in parsed.values():
            if isinstance(v, list):
                return [m for m in v if isinstance(m, str) and m.strip()]
        return []
    except Exception as exc:
        logger.warning("Generation failed for scenario '%s': %s", scenario["description"][:40], exc)
        return []


def generate(n_per_scenario: int = 12, output_dir: Path = None) -> None:
    output_dir = output_dir or Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY not set — cannot generate dataset.")

    all_records: List[Dict[str, Any]] = []

    for i, scenario in enumerate(SCENARIOS):
        logger.info(
            "[%d/%d] Generating %d examples — %s / %s / %s",
            i + 1, len(SCENARIOS), n_per_scenario,
            scenario["language"], scenario["mode"], scenario["description"][:50],
        )
        messages = _generate_batch(client, scenario, n_per_scenario)
        logger.info("  → %d messages generated", len(messages))

        for msg in messages:
            record: Dict[str, Any] = {
                "text": msg,
                "language": scenario["language"],
                "mode": scenario["mode"],
                **scenario["affect"],
            }
            # Ensure physician records have rep fields defaulted
            if scenario["mode"] == "physician_portal":
                record.setdefault("rep_confidence", "medium")
                record.setdefault("engagement_level", "active")
            else:
                record.setdefault("query_urgency", "routine")
            all_records.append(record)

        # Respect Groq rate limits
        if i < len(SCENARIOS) - 1:
            time.sleep(1.2)

    random.shuffle(all_records)
    split = int(len(all_records) * 0.8)
    train_records = all_records[:split]
    val_records = all_records[split:]

    train_path = output_dir / "affect_training.jsonl"
    val_path = output_dir / "affect_validation.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Done. %d train + %d validation examples → %s",
        len(train_records), len(val_records), output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic affect training data")
    parser.add_argument("--n-per-scenario", type=int, default=12,
                        help="Examples to generate per scenario (default 12 → ~288 total)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write JSONL files (default: alia_nlp/data/)")
    args = parser.parse_args()
    generate(n_per_scenario=args.n_per_scenario, output_dir=args.output_dir)
