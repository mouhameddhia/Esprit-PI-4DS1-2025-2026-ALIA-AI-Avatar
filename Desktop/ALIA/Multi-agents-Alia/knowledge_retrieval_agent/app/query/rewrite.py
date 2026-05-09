"""Query rewriting and domain-specific expansion logic."""

from __future__ import annotations

from typing import Iterable


class QueryRewriter:
    """Expand user queries with pharma/scientific synonyms."""

    def __init__(self) -> None:
        """Initialize static domain lexicon used for expansion."""

        self.term_map: dict[str, list[str]] = {
            "adverse effects": ["side effects", "drug safety", "toxicity"],
            "efficacy": ["effectiveness", "clinical benefit", "outcome"],
            "dosage": ["dose", "dosing", "administration"],
            "contraindications": ["warnings", "precautions"],
            "pharmacokinetics": ["absorption", "distribution", "metabolism", "excretion"],
        }

    def rewrite(self, query: str) -> str:
        """Return an expanded query string for better recall."""

        lowered = query.lower()
        additions: list[str] = []
        for key, synonyms in self.term_map.items():
            if key in lowered:
                additions.extend(synonyms)

        if not additions:
            return query
        return f"{query} {' '.join(sorted(set(additions)))}"

    def build_fallback_queries(self, query: str) -> list[str]:
        """Return alternate fallback queries for CRAG corrective retrieval."""

        rewritten = self.rewrite(query)
        variants: list[str] = [query]
        if rewritten != query:
            variants.append(rewritten)

        # Add high-recall keyword-only variant for sparse retrieval recovery.
        keyword_variant = " ".join(token for token in rewritten.split() if len(token) > 3)
        if keyword_variant and keyword_variant not in variants:
            variants.append(keyword_variant)

        return variants

    @staticmethod
    def deduplicate(queries: Iterable[str]) -> list[str]:
        """Preserve order while removing duplicate query strings."""

        seen: set[str] = set()
        output: list[str] = []
        for item in queries:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output
