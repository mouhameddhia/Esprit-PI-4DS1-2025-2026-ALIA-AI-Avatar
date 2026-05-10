"""Unit tests for L2 intent classification."""

import pytest
from alia_nlp.src.layers.L2_intent.rules import classify_with_rules
from alia_nlp.src.layers.L2_intent.mode_filter import apply_mode_filter, MEDREP_ONLY_INTENTS


class TestRuleClassifier:
    def test_greeting(self):
        intent, conf = classify_with_rules("Hello doctor.", "medrep_training")
        assert intent == "general_greeting"
        assert conf >= 0.90

    def test_dosage_numeric(self):
        intent, conf = classify_with_rules("Is 10 mg twice daily safe?", "physician_portal")
        assert intent == "dosage_question"
        assert conf >= 0.90

    def test_flash_visit(self):
        intent, conf = classify_with_rules("I have 30 seconds for a flash visit.", "medrep_training")
        assert intent == "visit_format_request"
        assert conf >= 0.90

    def test_safety_keywords(self):
        intent, conf = classify_with_rules("Any QT prolongation risk?", "physician_portal")
        assert intent == "safety_question"

    def test_low_confidence_on_vague(self):
        intent, conf = classify_with_rules("Yes.", "physician_portal")
        assert intent == "other"
        assert conf < 0.50

    def test_product_info(self):
        intent, conf = classify_with_rules("What is the mechanism of action?", "physician_portal")
        assert intent == "product_information_request"


class TestModeFilter:
    def test_blocks_training_in_physician_portal(self):
        result = apply_mode_filter("training_simulation", "physician_portal")
        assert result not in MEDREP_ONLY_INTENTS

    def test_allows_training_in_medrep(self):
        result = apply_mode_filter("training_simulation", "medrep_training")
        assert result == "training_simulation"

    def test_clinical_fallback_on_filter(self):
        result = apply_mode_filter("crm_follow_up", "physician_portal", is_safety=True)
        assert result == "safety_question"
