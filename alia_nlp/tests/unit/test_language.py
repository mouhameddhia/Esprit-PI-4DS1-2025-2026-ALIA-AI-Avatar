"""Unit tests for L1 language detection."""

import pytest
from alia_nlp.src.layers.L1_language.detector import detect_language


def test_english():
    assert detect_language("What is the dosage for this product?") == "en"


def test_french():
    assert detect_language("Bonjour, est-ce que vous avez la posologie?") == "fr"


def test_spanish():
    assert detect_language("Hola, ¿cuál es la dosis recomendada para este medicamento?") == "es"


def test_arabic():
    assert detect_language("ما هي الجرعة الموصى بها لهذا الدواء؟") == "ar"


def test_empty():
    assert detect_language("") == "unknown"


def test_single_word_english():
    assert detect_language("hello") == "en"


def test_mixed_defaults_to_majority():
    # More English markers than French
    result = detect_language("the patient is elderly and the product is approved")
    assert result == "en"
