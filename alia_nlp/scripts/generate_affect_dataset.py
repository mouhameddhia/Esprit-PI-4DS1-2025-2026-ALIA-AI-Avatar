"""
Synthetic affect dataset generator for L7 fine-tuning.

Generates labeled training examples via Groq LLM across all affect states,
both languages (en, fr), and both modes (medrep_training, physician_portal).
Includes adversarial and stress scenarios for better boundary learning.

Output: alia_nlp/data/affect_training.jsonl
        alia_nlp/data/affect_validation.jsonl  (20% hold-out)

Usage:
    python -m alia_nlp.scripts.generate_affect_dataset
    python -m alia_nlp.scripts.generate_affect_dataset --n-per-scenario 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

_ENV_PATH = Path(__file__).resolve().parents[2] / "backend" / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except ImportError:
        for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from alia_nlp.utils.groq_client import get_groq_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Scenario definitions ─────────────────────────────────────────────────────
# affect keys: rep_confidence, engagement_level (binary), frustration_signal,
#              stress_signal, query_urgency
SCENARIOS: List[Dict[str, Any]] = [

    # ═══════════════════════════════════════════════════════════════════
    # MEDREP / ENGLISH — Core scenarios
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — price objection",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a confident, assertive rep handling a 'too expensive' objection with strong clinical arguments",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — price objection",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": False},
        "description": "a frustrated, uncertain rep losing confidence against a price objection, giving short defeated replies",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — 'no time' objection",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a rep with moderate confidence calmly handling a physician's time objection",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — argumentation phase",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a highly engaged rep asking for harder role-play challenges after succeeding",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — opening permission",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": False, "stress_signal": False},
        "description": "a passive, uncertain rep giving a minimal attempt at an opening statement, disengaged",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "sales_methodology_request — closing technique",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a rep actively asking for structured help with closing techniques mid-simulation",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "competency_assessment — asking for level feedback",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a confident rep requesting detailed competency evaluation and promotion criteria",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — safety concern objection",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": False},
        "description": "a rep clearly lost when handling a safety objection, giving up and asking for help",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "crm_follow_up — planning next visit",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a rep confidently summarizing CRM notes and structuring the next visit plan",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "product_information_request — clinical benefits",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a rep actively studying Cardivex clinical benefits to prepare stronger arguments",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDREP / ENGLISH — Adversarial scenarios (boundary learning)
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — habitual use objection",
        "affect": {"rep_confidence": "low", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a rep who sounds unsure and hedges every statement but is still actively trying and engaged — NOT frustrated, just not confident",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — discovery phase",
        "affect": {"rep_confidence": "medium", "engagement_level": "passive", "frustration_signal": False, "stress_signal": False},
        "description": "a rep giving formulaic, go-through-the-motions responses with no real engagement — bored but not frustrated",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — multiple objections",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": True, "stress_signal": False},
        "description": "a rep who sounds confident but is clearly frustrated after repeated failures — still engaged, but exasperated",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — full visit",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "a rep making a solid effort with reasonable domain vocabulary — neither very confident nor uncertain",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDREP / ENGLISH — Stress scenarios
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "objection_handling — simultaneous price + safety + time objections",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "a rep under pressure trying to handle three different objections at once, asking multiple questions in a single message",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "visit_format_request — flash visit under time pressure",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "a rep rushing to compress a full visit into 30 seconds, urgency in phrasing, rapid-fire questions",
    },
    {
        "language": "en", "mode": "medrep_training",
        "intent_context": "training_simulation — overwhelmed by complexity",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": True},
        "description": "a rep completely overwhelmed — stressed AND frustrated, asking incoherently across multiple topics at once",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDREP / FRENCH — Core scenarios
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection prix",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué très confiant qui gère une objection prix avec des arguments cliniques solides",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection prix",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": False},
        "description": "un délégué frustré et incertain face à une objection prix, qui abandonne et donne des réponses minimales",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection 'pas le temps'",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué avec une confiance modérée gérant calmement l'objection de manque de temps",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "training_simulation — phase d'argumentation",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué très engagé qui demande un scénario plus difficile après avoir réussi",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "training_simulation — prise de contact",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué passif et peu sûr de lui qui fait un effort minimal, réponses courtes et formulaiques",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "sales_methodology_request — technique de closing",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué actif qui demande de l'aide structurée sur les techniques de closing",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection sécurité",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": False},
        "description": "un délégué perdu qui ne sait pas comment répondre à une objection de sécurité, découragé",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "crm_follow_up — planification visite suivante",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué confiant qui résume ses notes CRM et structure la prochaine visite",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDREP / FRENCH — Adversarial scenarios
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objection habitudes prescriptives",
        "affect": {"rep_confidence": "low", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué hésitant et peu confiant mais qui continue d'essayer sérieusement — pas frustré, juste incertain",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "training_simulation — phase de découverte",
        "affect": {"rep_confidence": "medium", "engagement_level": "passive", "frustration_signal": False, "stress_signal": False},
        "description": "un délégué qui donne des réponses formulaiques sans vrai engagement — ennuyé mais pas frustré",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objections multiples",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": True, "stress_signal": False},
        "description": "un délégué confiant mais clairement exaspéré après des échecs répétés — toujours engagé mais frustré",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDREP / FRENCH — Stress scenarios
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "objection_handling — objections prix + sécurité + temps simultanées",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "un délégué sous pression qui essaie de gérer trois objections différentes en même temps, questions multiples en rafale",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "visit_format_request — visite flash sous contrainte de temps",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "un délégué qui essaie de compresser une visite complète en 30 secondes, phrasing urgent, questions rapides",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "intent_context": "training_simulation — dépassé par la complexité",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": True},
        "description": "un délégué complètement dépassé — stressé ET frustré, messages incohérents couvrant trop de sujets",
    },

    # ═══════════════════════════════════════════════════════════════════
    # PHYSICIAN / ENGLISH — Core + Adversarial + Stress
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "safety_question — drug interaction",
        "affect": {"query_urgency": "urgent", "frustration_signal": True, "stress_signal": False},
        "description": "a physician asking urgently about a dangerous drug interaction, impatient and direct",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "dosage_question — renal impairment dosing",
        "affect": {"query_urgency": "routine", "frustration_signal": False, "stress_signal": False},
        "description": "a physician calmly asking about dosing adjustment for a renal patient, no urgency",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "safety_question — teratogenicity",
        "affect": {"query_urgency": "elevated", "frustration_signal": False, "stress_signal": False},
        "description": "a physician expressing measured concern about prescribing to a potentially pregnant patient",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "product_information_request — mechanism of action",
        "affect": {"query_urgency": "routine", "frustration_signal": False, "stress_signal": False},
        "description": "a physician asking a straightforward, academic clinical question about mechanism of action",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "safety_question — multiple concurrent drug concerns",
        "affect": {"query_urgency": "urgent", "frustration_signal": False, "stress_signal": True},
        "description": "a physician stacking multiple drug safety questions in one urgent message — patient is in front of them right now",
    },
    {
        "language": "en", "mode": "physician_portal",
        "intent_context": "safety_question — complex polypharmacy patient",
        "affect": {"query_urgency": "elevated", "frustration_signal": False, "stress_signal": True},
        "description": "a physician overwhelmed by multiple concurrent concerns about a complex patient on many drugs, asking fragmented questions",
    },

    # ═══════════════════════════════════════════════════════════════════
    # PHYSICIAN / FRENCH — Core + Stress
    # ═══════════════════════════════════════════════════════════════════
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "safety_question — interaction médicamenteuse",
        "affect": {"query_urgency": "urgent", "frustration_signal": True, "stress_signal": False},
        "description": "un médecin posant une question urgente et impatiente sur une interaction médicamenteuse",
    },
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "dosage_question — insuffisance rénale",
        "affect": {"query_urgency": "routine", "frustration_signal": False, "stress_signal": False},
        "description": "un médecin posant calmement une question sur l'ajustement de dose en insuffisance rénale",
    },
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "safety_question — tératogénicité",
        "affect": {"query_urgency": "elevated", "frustration_signal": False, "stress_signal": False},
        "description": "un médecin exprimant une préoccupation mesurée sur la prescription chez une patiente potentiellement enceinte",
    },
    {
        "language": "fr", "mode": "physician_portal",
        "intent_context": "safety_question — polymédication complexe avec urgence",
        "affect": {"query_urgency": "urgent", "frustration_signal": False, "stress_signal": True},
        "description": "un médecin débordé par plusieurs préoccupations simultanées, patient devant lui, questions en rafale",
    },
]

# ── Multi-turn scenarios (response labeled in context of a preceding message) ──
# Each entry has a fixed 'context' (the preceding turn) so the model learns
# that the SAME words carry different affect depending on what preceded them.
MULTI_TURN_SCENARIOS: List[Dict[str, Any]] = [
    # EN — rep responds confidently after a hard objection
    {
        "language": "en", "mode": "medrep_training",
        "context": "Physician: 'I've been prescribing generic amlodipine for years and my patients do fine. I see no reason to switch to Cardivex.'",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "rep responding confidently and assertively with a clear clinical argument",
    },
    {
        "language": "en", "mode": "medrep_training",
        "context": "Physician: 'I've been prescribing generic amlodipine for years and my patients do fine. I see no reason to switch to Cardivex.'",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": False},
        "description": "rep giving up, short defeated reply, clearly doesn't know how to handle this objection",
    },
    # EN — rep under time pressure (flash visit context)
    {
        "language": "en", "mode": "medrep_training",
        "context": "ALIA: 'You have 30 seconds with the physician. The physician is already heading to their next patient.'",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "rep rapidly trying to deliver key message under extreme time pressure, rushed phrasing",
    },
    {
        "language": "en", "mode": "medrep_training",
        "context": "ALIA: 'You have 30 seconds with the physician. The physician is already heading to their next patient.'",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": True},
        "description": "rep panicking, completely disorganized, asking multiple questions at once instead of acting",
    },
    # EN — rep after positive feedback (confidence boost context)
    {
        "language": "en", "mode": "medrep_training",
        "context": "ALIA: 'Good job handling the price objection! Now the physician raises a safety concern about QT prolongation.'",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "rep riding momentum, confident and engaged after positive feedback, ready for the next challenge",
    },
    {
        "language": "en", "mode": "medrep_training",
        "context": "ALIA: 'Good job handling the price objection! Now the physician raises a safety concern about QT prolongation.'",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "rep stressed by the sudden shift to a harder topic, scrambling to recall safety data",
    },
    # EN — physician follow-up context
    {
        "language": "en", "mode": "physician_portal",
        "context": "Previous answer: 'Cardivex (amlodipine 5mg) can be used in mild renal impairment without dose adjustment.'",
        "affect": {"query_urgency": "elevated", "frustration_signal": True, "stress_signal": False},
        "description": "physician frustrated with the previous answer, asking a sharper follow-up question",
    },
    {
        "language": "en", "mode": "physician_portal",
        "context": "Previous answer: 'Cardivex (amlodipine 5mg) can be used in mild renal impairment without dose adjustment.'",
        "affect": {"query_urgency": "routine", "frustration_signal": False, "stress_signal": False},
        "description": "physician satisfied, asking a calm natural follow-up question",
    },
    # FR — rep after failed attempt (frustration context)
    {
        "language": "fr", "mode": "medrep_training",
        "context": "Médecin: 'Votre argument ne me convainc pas. Le générique coûte trois fois moins cher.'",
        "affect": {"rep_confidence": "low", "engagement_level": "passive", "frustration_signal": True, "stress_signal": False},
        "description": "délégué découragé après avoir été rejeté, réponse courte et défaitiste",
    },
    {
        "language": "fr", "mode": "medrep_training",
        "context": "Médecin: 'Votre argument ne me convainc pas. Le générique coûte trois fois moins cher.'",
        "affect": {"rep_confidence": "high", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": False},
        "description": "délégué persistant et confiant qui reformule son argument avec de nouveaux bénéfices cliniques",
    },
    # FR — rep under time pressure
    {
        "language": "fr", "mode": "medrep_training",
        "context": "ALIA: 'Visite flash — vous avez 30 secondes. Le médecin est pressé et part dans un instant.'",
        "affect": {"rep_confidence": "medium", "engagement_level": "engaged", "frustration_signal": False, "stress_signal": True},
        "description": "délégué stressé qui essaie de condenser son message clé en quelques secondes, phrasing haché",
    },
    # FR — physician follow-up
    {
        "language": "fr", "mode": "physician_portal",
        "context": "Réponse précédente: 'Cardivex peut être administré sans ajustement de dose en cas d'insuffisance rénale légère.'",
        "affect": {"query_urgency": "elevated", "frustration_signal": True, "stress_signal": False},
        "description": "médecin insatisfait de la réponse précédente, posant une question de suivi plus précise et impatiente",
    },
]

# ── Generation prompts ────────────────────────────────────────────────────────
_GEN_SYSTEM = (
    "You are a pharmaceutical training data generator. "
    "Generate realistic, diverse, natural-sounding messages. "
    "Return ONLY a valid JSON object with a single key 'messages' containing an array of strings."
)

_GEN_MEDREP_TEMPLATE = """\
Generate {n} realistic messages a medical representative might send during a \
pharmaceutical sales training simulation with an AI assistant.

Context:
- Language: {language} (write ENTIRELY in {language} — no mixing)
- Training scenario: {intent_context}
- Product context: Cardivex (amlodipine-based cardiovascular product, 5mg and 10mg)
- Affect state to portray: {description}

Requirements:
- Between 8 and 70 words per message
- Vary length, phrasing, and structure significantly across the {n} messages
- Some can be mid-simulation practice attempts, some requests for feedback, some opening statements
- Authentically reflect the described affect — if stressed, use rapid questions or "how do I handle X AND Y?"; \
if frustrated, use short defeated phrases; if confident, use assertive complete sentences
- Do NOT label or explain the affect — write naturally as the rep would

Return JSON: {{"messages": ["msg1", "msg2", ...]}}"""

_GEN_PHYSICIAN_TEMPLATE = """\
Generate {n} realistic queries a healthcare professional might send to a \
pharmaceutical digital assistant.

Context:
- Language: {language} (write ENTIRELY in {language} — no mixing)
- Clinical scenario: {intent_context}
- Product context: Cardivex (amlodipine-based cardiovascular product)
- Tone to portray: {description}

Requirements:
- Between 5 and 50 words per message
- Vary phrasing, clinical vocabulary, and sentence structure naturally
- If stressed, stack multiple concerns or use urgent fragmented phrasing; \
if routine, write one clear focused question; if urgent, be terse and direct
- Do NOT label or explain the tone — write naturally as the physician would

Return JSON: {{"messages": ["q1", "q2", ...]}}"""


_GEN_MULTITURN_TEMPLATE = """\
PRECEDING MESSAGE:
"{context}"

Generate {n} realistic messages that a {speaker} might send IN DIRECT RESPONSE to the above.
Language: {language} (write ENTIRELY in {language})
Affect to portray: {description}

Requirements:
- Between 8 and 60 words per response
- Each response must clearly relate to and follow from the preceding message
- Authentically reflect the described affect in the response
- Vary phrasing and structure across the {n} responses
- Do NOT label or explain the affect

Return JSON: {{"messages": ["response1", "response2", ...]}}"""


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
            temperature=0.9,
            max_tokens=3000,
        )
        raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        # Accept {"messages": [...]} or any list value
        if isinstance(parsed, list):
            msgs = parsed
        else:
            msgs = next((v for v in parsed.values() if isinstance(v, list)), [])
        return [m for m in msgs if isinstance(m, str) and m.strip()]
    except Exception as exc:
        logger.warning("Generation failed for '%s': %s", scenario["description"][:50], exc)
        return []


def _generate_multi_turn_batch(client, scenario: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    speaker = "medical representative" if scenario["mode"] == "medrep_training" else "physician"
    lang    = "English" if scenario["language"] == "en" else "French"
    prompt  = _GEN_MULTITURN_TEMPLATE.format(
        context=scenario["context"], n=n, speaker=speaker,
        language=lang, description=scenario["description"],
    )
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _GEN_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=2500,
        )
        raw    = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        msgs   = parsed if isinstance(parsed, list) else next(
            (v for v in parsed.values() if isinstance(v, list)), []
        )
        return [
            {"text": m, "context": scenario["context"]}
            for m in msgs if isinstance(m, str) and m.strip()
        ]
    except Exception as exc:
        logger.warning("Multi-turn generation failed: %s", exc)
        return []


def generate(n_per_scenario: int = 40, output_dir: Path = None) -> None:
    output_dir = output_dir or Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY not set — cannot generate dataset.")

    all_records: List[Dict[str, Any]] = []

    total_scenarios = len(SCENARIOS) + len(MULTI_TURN_SCENARIOS)

    for i, scenario in enumerate(SCENARIOS):
        logger.info(
            "[%d/%d] Generating %d — %s / %s / %s",
            i + 1, total_scenarios, n_per_scenario,
            scenario["language"], scenario["mode"], scenario["description"][:55],
        )
        messages = _generate_batch(client, scenario, n_per_scenario)
        logger.info("  → %d messages", len(messages))

        for msg in messages:
            record: Dict[str, Any] = {
                "text": msg,
                "language": scenario["language"],
                "mode": scenario["mode"],
                **scenario["affect"],
            }
            record.setdefault("rep_confidence", "medium")
            record.setdefault("engagement_level", "engaged")
            record.setdefault("query_urgency", "routine")
            record.setdefault("frustration_signal", False)
            record.setdefault("stress_signal", False)
            all_records.append(record)

        time.sleep(1.2)

    n_multi = max(n_per_scenario // 2, 10)
    for i, scenario in enumerate(MULTI_TURN_SCENARIOS):
        logger.info(
            "[%d/%d] Multi-turn %d — %s / %s",
            len(SCENARIOS) + i + 1, total_scenarios, n_multi,
            scenario["language"], scenario["description"][:55],
        )
        pairs = _generate_multi_turn_batch(client, scenario, n_multi)
        logger.info("  → %d context-response pairs", len(pairs))

        for pair in pairs:
            record = {
                "text":     pair["text"],
                "context":  pair["context"],
                "language": scenario["language"],
                "mode":     scenario["mode"],
                **scenario["affect"],
            }
            record.setdefault("rep_confidence", "medium")
            record.setdefault("engagement_level", "engaged")
            record.setdefault("query_urgency", "routine")
            record.setdefault("frustration_signal", False)
            record.setdefault("stress_signal", False)
            all_records.append(record)

        if i < len(MULTI_TURN_SCENARIOS) - 1:
            time.sleep(1.2)

    random.shuffle(all_records)
    split = int(len(all_records) * 0.8)
    train_records = all_records[:split]
    val_records   = all_records[split:]

    train_path = output_dir / "affect_training.jsonl"
    val_path   = output_dir / "affect_validation.jsonl"

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
    parser.add_argument("--n-per-scenario", type=int, default=40,
                        help="Examples per scenario (default 40 → ~1760 total)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    generate(n_per_scenario=args.n_per_scenario, output_dir=args.output_dir)
