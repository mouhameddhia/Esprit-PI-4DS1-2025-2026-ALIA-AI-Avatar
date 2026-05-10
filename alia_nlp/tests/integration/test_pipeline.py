"""Integration tests — golden-set end-to-end through the online pipeline.

These tests do NOT call the LLM (GROQ_API_KEY absent in CI).
They exercise the full pipeline using the rule-based fast path only.
"""

import os
import pytest

# Ensure no LLM call in CI
os.environ.setdefault("GROQ_API_KEY", "")

from alia_nlp.src.pipeline.online import analyze


GOLDEN_SET = [
    # Greetings
    ("Hello doctor.", "medrep_training", "general_greeting", []),
    ("Bonjour.", "medrep_training", "general_greeting", []),
    # Flash visit
    ("I have 30 seconds for a flash visit.", "medrep_training", "visit_format_request", []),
    # Dosage — numeric
    ("Is 5 mg once daily the right dose?", "physician_portal", "dosage_question", []),
    # Safety — short clinical query
    ("QT prolongation risk?", "physician_portal", "safety_question", ["contraindication_query"]),
    ("Teratogenic effects documented?", "physician_portal", "safety_question", ["patient_specific_advice_request"]),
    # Mode filter — training intent blocked in physician portal
    ("Simulate a flash visit.", "physician_portal", None, []),  # None = any non-medrep intent
    # Objection
    ("I'm not convinced by the evidence.", "medrep_training", "objection_handling", []),
    # Product info
    ("What is the mechanism of action?", "physician_portal", "product_information_request", []),
]


@pytest.mark.parametrize("text,mode,expected_intent,expected_flags", GOLDEN_SET)
def test_golden(text, mode, expected_intent, expected_flags):
    result = analyze(text, mode=mode)

    if expected_intent is not None:
        assert result.intent == expected_intent, (
            f"'{text}' [{mode}] → got {result.intent!r}, expected {expected_intent!r}"
        )

    for flag in expected_flags:
        assert flag in result.safety_flags, (
            f"'{text}' missing safety flag {flag!r}, got {result.safety_flags}"
        )


def test_mode_filter_blocks_medrep_intents_in_physician_portal():
    from alia_nlp.src.layers.L2_intent.mode_filter import MEDREP_ONLY_INTENTS
    result = analyze("Simulate a flash visit with a skeptical doctor.", mode="physician_portal")
    assert result.intent not in MEDREP_ONLY_INTENTS


def test_result_is_typed():
    from alia_nlp.src.schema import NLPResult
    result = analyze("Hello.", mode="physician_portal")
    assert isinstance(result, NLPResult)
    assert isinstance(result.intent, str)
    assert isinstance(result.confidence, float)
    assert isinstance(result.safety_flags, list)


def test_to_dict_backward_compat():
    result = analyze("What is the dosage?", mode="physician_portal")
    d = result.to_dict()
    assert "intent" in d
    assert "entity_map" in d
    assert "needs_clarification" in d


def test_arabic_language_detected():
    result = analyze("ما هي الجرعة الموصى بها؟", mode="physician_portal")
    assert result.language == "ar"


def test_spanish_language_detected():
    result = analyze("Hola, ¿cuál es la dosis para pacientes mayores?", mode="physician_portal")
    assert result.language == "es"


def test_fallback_on_empty_input():
    result = analyze("", mode="physician_portal")
    assert result.intent == "other"
    assert result.needs_clarification is True
