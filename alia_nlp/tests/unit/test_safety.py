"""Unit tests for L4 safety flag detection."""

import pytest
from alia_nlp.src.layers.L4_safety.detector import detect


class TestSafetyDetector:
    def test_teratogenicity_triggers_patient_specific(self):
        flags = detect([], "Teratogenic effects documented?")
        assert "patient_specific_advice_request" in flags

    def test_qt_prolongation_triggers_contraindication(self):
        flags = detect([], "Is there a QT prolongation risk?")
        assert "contraindication_query" in flags

    def test_nephrotoxicity_triggers_contraindication(self):
        flags = detect([], "Nephrotoxicity concerns with this drug?")
        assert "contraindication_query" in flags

    def test_breastfeeding_triggers_patient_specific(self):
        flags = detect([], "Can I use during breastfeeding?")
        assert "patient_specific_advice_request" in flags

    def test_drug_interaction_triggers_high_risk(self):
        flags = detect([], "Drug-drug interaction with warfarin?")
        assert "high_risk_interaction" in flags

    def test_off_label_triggers_flag(self):
        flags = detect([], "Is there any off-label use for this?")
        assert "off_label_request" in flags

    def test_no_false_positive_on_generic(self):
        flags = detect([], "Hello doctor, how are you today?")
        assert flags == []

    def test_llm_flags_accepted_if_valid(self):
        flags = detect(["diagnosis_request"], "Please diagnose this condition.")
        assert "diagnosis_request" in flags

    def test_invalid_llm_flags_discarded(self):
        flags = detect(["made_up_flag"], "Some text.")
        assert "made_up_flag" not in flags

    def test_max_8_flags(self):
        flags = detect([], "Teratogenic nephrotoxic hepatotoxic qt prolongation black box drug interaction off-label diagnose")
        assert len(flags) <= 8
