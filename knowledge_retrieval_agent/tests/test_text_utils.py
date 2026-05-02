"""Tests for scientific text tokenization."""

from app.utils.text import tokenize_scientific_text


def test_tokenize_scientific_text_preserves_domain_terms() -> None:
    """Ensure BM25 tokenization keeps scientific terms and strips noise."""

    tokens = tokenize_scientific_text("Dose was 5 mg/dL for TNF-α response.")

    assert "mg/dl" in tokens
    assert "tnf-α" in tokens
    assert "dose" in tokens
    assert "was" not in tokens