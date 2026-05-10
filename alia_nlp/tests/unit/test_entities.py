"""Unit tests for L3 entity extraction."""

import pytest
from alia_nlp.src.layers.L3_entities.rules import extract_rules
from alia_nlp.src.layers.L3_entities.normalizer import normalize_entity_map, merge_maps


class TestRuleExtraction:
    def test_numeric_dosage(self):
        out = extract_rules("Give 10 mg twice daily.")
        assert any("10" in d for d in out["dosage"])

    def test_patient_profile_elderly(self):
        out = extract_rules("Dosing for elderly patients?")
        assert "elderly" in out["patient_profile"]

    def test_product_name_prefix(self):
        out = extract_rules("What is the dosage of Product Alpha?")
        assert any("Product" in p for p in out["product_name"])

    def test_adverse_event_teratogen(self):
        out = extract_rules("Teratogenic effects documented?")
        assert any("teratogen" in e for e in out["adverse_event"])

    def test_visit_format_flash(self):
        out = extract_rules("Let's do a flash visit in 30 seconds.")
        assert "Flash" in out["visit_format"]

    def test_proof_reference(self):
        out = extract_rules("Show me a clinical study or guideline.")
        assert out["proof_reference"]


class TestNormalizer:
    def test_invalid_keys_discarded(self):
        result = normalize_entity_map({"priority": ["fast"], "dosage": ["10 mg"]})
        assert "priority" not in result
        assert result["dosage"] == ["10 mg"]

    def test_merge_deduplicates(self):
        a = {"dosage": ["10 mg"], "product_name": ["ProductX"]}
        b = {"dosage": ["10 mg"], "product_name": ["ProductY"]}
        merged = merge_maps(a, b)
        assert merged["dosage"].count("10 mg") == 1
        assert "ProductY" in merged["product_name"]
