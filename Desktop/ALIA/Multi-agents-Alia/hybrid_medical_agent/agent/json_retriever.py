"""Fast JSON knowledge retrieval with source-first semantics."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


@dataclass
class JsonRetrievalResult:
    """Structured retrieval output used by the controller."""

    source: str
    content: str
    confidence: float
    latency_ms: float
    drug_name: str | None
    topic: str | None
    source_file: str | None


class JSONRetriever:
    """Load manual JSON payloads and answer topic-level lookups quickly.

    Note: knowledge-base JSON files were moved to
    `rag_knowledge_builder/scripts`. Use `rag_knowledge_builder.kb_loader`
    to discover KB JSONs so code no longer hardcodes the old
    `rag_knowledge_base/json` path.
    """

    def __init__(
        self,
        workspace_root: Path,
        json_dirs: tuple[str, ...] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        # Resolve default JSON dirs via centralized loader to keep a single source of truth
        if json_dirs is None:
            try:
                from rag_knowledge_builder.kb_loader import default_json_dirs

                json_dirs = tuple(default_json_dirs())
            except Exception:
                json_dirs = ("rag_knowledge_builder/scripts",)

        self.json_paths = [workspace_root / folder for folder in json_dirs]
        self.payloads: list[dict[str, Any]] = []
        self._drug_names: list[str] = []
        self._load_payloads()

    @property
    def drug_names(self) -> list[str]:
        return list(self._drug_names)

    @property
    def has_multiple_drugs(self) -> bool:
        return len(self._drug_names) > 1

    @property
    def sample_drugs(self) -> list[str]:
        return self._drug_names[:8]

    def _discover_files(self) -> list[Path]:
        files: list[Path] = []
        for base in self.json_paths:
            if not base.exists():
                continue
            for path in base.rglob("*.json"):
                name = path.name.lower()
                if "rag_knowledge_builder/scripts" in str(path).replace("\\", "/"):
                    if not name.startswith("incoming_payload_"):
                        continue
                files.append(path)
        return sorted(set(files))

    @staticmethod
    def _expand_aliases(raw_name: str) -> list[str]:
        aliases: set[str] = set()
        clean = raw_name.strip()
        if not clean:
            return []

        aliases.add(clean)
        aliases.add(clean.replace(".pptx", "").replace("Product Range", "").replace("Gamme", "").strip())

        outside_parentheses = re.sub(r"\(.*?\)", "", clean).strip(" ,")
        if outside_parentheses:
            aliases.add(outside_parentheses)

        for group in re.findall(r"\((.*?)\)", clean):
            for piece in re.split(r"[,;/]", group):
                token = piece.strip()
                if token:
                    aliases.add(token)

        for piece in re.split(r"[,;/]", clean):
            token = piece.strip()
            if token:
                aliases.add(token)

        return sorted(alias for alias in aliases if len(alias) >= 2)

    def _extract_drug_names(self, payload: dict[str, Any]) -> list[str]:
        names: set[str] = set()

        name = payload.get("drug_name")
        if isinstance(name, str) and name.strip():
            names.update(self._expand_aliases(name))

        doc_name = payload.get("document_name") or payload.get("document")
        if isinstance(doc_name, str) and doc_name.strip():
            names.update(self._expand_aliases(doc_name))

        products = payload.get("products")
        if isinstance(products, dict) and products:
            for product_name in products.keys():
                if isinstance(product_name, str) and product_name.strip():
                    names.update(self._expand_aliases(product_name))

        return sorted(names)

    def _load_payloads(self) -> None:
        payloads: list[dict[str, Any]] = []
        for json_file in self._discover_files():
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload["__source_file"] = str(json_file.relative_to(self.workspace_root)).replace("\\", "/")
            payload["__aliases"] = self._extract_drug_names(payload)
            payloads.append(payload)

        self.payloads = payloads
        self._drug_names = sorted(
            {
                drug
                for payload in payloads
                for drug in payload.get("__aliases", [])
                if isinstance(drug, str) and drug
            }
        )

    def refresh(self) -> None:
        """Reload JSON files to include any manual updates."""

        self._load_payloads()

    def get_payload(self, drug_name: str | None) -> dict[str, Any] | None:
        """Return the best matching payload for a drug/product name."""

        matches = self._match_payloads(drug_name)
        return matches[0] if matches else None

    @staticmethod
    def build_full_context(payload: dict[str, Any], drug_name: str | None = None) -> str:
        """Create a compact full-context view for open commercial questions."""

        ordered_keys = [
            "document_name",
            "drug_name",
            "type",
            "indications",
            "composition",
            "dosage",
            "administration",
            "warnings",
            "side_effects",
            "mechanism_of_action",
            "age",
        ]

        sections: list[str] = []
        for key in ordered_keys:
            value = payload.get(key)
            if value in (None, "", {}, []):
                continue
            if isinstance(value, dict):
                en = value.get("en") if isinstance(value.get("en"), str) else ""
                fr = value.get("fr") if isinstance(value.get("fr"), str) else ""
                if en.strip() or fr.strip():
                    parts = []
                    if en.strip():
                        parts.append(f"EN: {en.strip()}")
                    if fr.strip():
                        parts.append(f"FR: {fr.strip()}")
                    sections.append(f"{key.upper()}: " + " | ".join(parts))
                    continue
                sections.append(f"{key.upper()}: {json.dumps(value, ensure_ascii=False)}")
                continue
            sections.append(f"{key.upper()}: {value}")

        if drug_name and drug_name not in " ".join(sections):
            sections.insert(0, f"DRUG_QUERY: {drug_name}")

        return "\n".join(sections)

    def _match_payloads(self, drug_name: str | None) -> list[dict[str, Any]]:
        if not drug_name:
            return []
        q = _normalize(drug_name)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for payload in self.payloads:
            aliases = payload.get("__aliases", [])
            norm_aliases = [_normalize(alias) for alias in aliases]

            rank = 0
            if any(q == alias for alias in norm_aliases):
                rank = 3
            elif any(q in alias for alias in norm_aliases):
                rank = 2
            elif any(alias in q for alias in norm_aliases if len(alias) >= 4):
                rank = 1

            if rank > 0:
                ranked.append((rank, payload))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in ranked]

    @staticmethod
    def _extract_subproduct_qualifiers(question: str | None, drug_name: str) -> list[str]:
        """Extract qualifier tokens following the detected base product in the question."""

        if not question:
            return []

        normalized_question = _normalize(question)
        normalized_drug = _normalize(drug_name)
        if normalized_drug not in normalized_question:
            return []

        suffix = normalized_question.split(normalized_drug, maxsplit=1)[1]
        suffix_tokens = [token for token in suffix.split() if token]

        stop = {
            "what",
            "which",
            "how",
            "about",
            "indication",
            "indications",
            "composition",
            "dosage",
            "dose",
            "administration",
            "warnings",
            "warning",
            "side",
            "effects",
            "effect",
            "is",
            "it",
            "exactly",
            "the",
            "of",
            "and",
            "for",
            "quelle",
            "quelles",
            "quels",
            "quel",
            "sont",
            "est",
            "posologie",
            "de",
            "des",
            "du",
            "la",
            "le",
            "les",
            "et",
            "quoi",
            "exactement",
        }

        qualifiers: list[str] = []
        for token in suffix_tokens:
            token_clean = re.sub(r"[^a-z0-9]+", "", token)
            if not token_clean:
                continue
            if token_clean in stop or len(token_clean) <= 2:
                continue
            qualifiers.append(token_clean)
            if len(qualifiers) >= 3:
                break
        return qualifiers

    @staticmethod
    def _qualifier_variants(token: str) -> set[str]:
        mapping = {
            "professional": {"professional", "professionnel"},
            "professionnel": {"professional", "professionnel"},
            "soap": {"soap", "savon"},
            "savon": {"soap", "savon"},
            "disinfectant": {"disinfectant", "desinfectant", "désinfectant"},
            "desinfectant": {"disinfectant", "desinfectant", "désinfectant"},
            "spray": {"spray"},
            "gel": {"gel"},
            "capsule": {"capsule", "capsules", "gelule", "gelules", "gélule", "gélules"},
            "capsules": {"capsule", "capsules", "gelule", "gelules", "gélule", "gélules"},
            "gelule": {"capsule", "capsules", "gelule", "gelules", "gélule", "gélules"},
            "gelules": {"capsule", "capsules", "gelule", "gelules", "gélule", "gélules"},
            "lipo": {"lipo", "liposomal"},
        }
        return mapping.get(token, {token})

    @staticmethod
    def _label_matches_qualifiers(normalized_label: str, qualifier_tokens: list[str]) -> bool:
        if not qualifier_tokens:
            return True
        return all(any(variant in normalized_label for variant in JSONRetriever._qualifier_variants(token)) for token in qualifier_tokens)

    @staticmethod
    def _split_enumerated_entries(text: str) -> list[str]:
        """Split numbered content like '1) ... 2) ...' into individual entries."""

        pattern = re.compile(r"(?:^|\s)(\d+\)\s.*?)(?=(?:\s\d+\)\s)|$)", re.DOTALL)
        entries = [match.group(1).strip() for match in pattern.finditer(text)]
        return [entry for entry in entries if entry]

    @staticmethod
    def _entry_label(entry: str) -> str:
        head = entry.split(":", maxsplit=1)[0]
        return head.strip()

    @staticmethod
    def _contains_variant_for_base(normalized_text: str, base_name: str) -> bool:
        pattern = re.compile(rf"\b{re.escape(base_name)}\b\s+[a-z0-9]{{3,}}")
        return pattern.search(normalized_text) is not None

    @staticmethod
    def _is_base_only_label(label: str, base_name: str) -> bool:
        return re.search(rf"\b{re.escape(base_name)}\b(?=\s*(?:\(|:|$))", label) is not None

    @staticmethod
    def _narrow_enumerated_entries(entries: list[str], base_name: str, qualifier_tokens: list[str]) -> list[str]:
        if not entries:
            return []

        labeled = [(entry, _normalize(JSONRetriever._entry_label(entry))) for entry in entries]

        if qualifier_tokens:
            matched = [
                entry
                for entry, normalized_label in labeled
                if base_name in normalized_label and JSONRetriever._label_matches_qualifiers(normalized_label, qualifier_tokens)
            ]
            if matched:
                return matched

        base_only = [
            entry
            for entry, normalized_label in labeled
            if JSONRetriever._is_base_only_label(normalized_label, base_name)
        ]
        if base_only:
            return base_only

        return []

    @staticmethod
    def _extract_segment_for_subproduct(
        value: str,
        drug_name: str | None,
        question: str | None = None,
    ) -> str:
        if not drug_name:
            return value

        base_name = _normalize(drug_name)
        if not base_name:
            return value

        qualifier_tokens = JSONRetriever._extract_subproduct_qualifiers(question=question, drug_name=drug_name)

        entries = JSONRetriever._split_enumerated_entries(value)
        if entries:
            narrowed = JSONRetriever._narrow_enumerated_entries(
                entries=entries,
                base_name=base_name,
                qualifier_tokens=qualifier_tokens,
            )
            if narrowed:
                return " ".join(narrowed)

        sentences = re.split(r"(?<=[.;])\s+", value)
        matches: list[str] = []
        base_tokens = [token for token in base_name.split() if token not in {"n", "no", "n0"}]
        for sentence in sentences:
            normalized_sentence = _normalize(sentence)
            if not all(token in normalized_sentence for token in base_tokens[:2]):
                continue
            if qualifier_tokens and not all(token in normalized_sentence for token in qualifier_tokens):
                continue
            if (not qualifier_tokens) and JSONRetriever._contains_variant_for_base(normalized_sentence, base_name):
                continue
            matches.append(sentence.strip())

        if matches:
            return " ".join(matches)
        return value

    @staticmethod
    def _narrow_topic_value(value: Any, drug_name: str | None, question: str | None) -> Any:
        if isinstance(value, str):
            return JSONRetriever._extract_segment_for_subproduct(value, drug_name, question=question)

        if isinstance(value, dict):
            narrowed: dict[str, Any] = {}
            changed = False
            for key, nested in value.items():
                if isinstance(nested, str):
                    narrowed_nested = JSONRetriever._extract_segment_for_subproduct(nested, drug_name, question=question)
                    narrowed[key] = narrowed_nested
                    changed = changed or (narrowed_nested != nested)
                else:
                    narrowed[key] = nested
            return narrowed if changed else value

        return value

    @staticmethod
    def _extract_topic(
        payload: dict[str, Any],
        topic: str,
        drug_name: str | None = None,
        question: str | None = None,
    ) -> Any:
        value = payload.get(topic)
        if value not in (None, "", {}, []):
            return JSONRetriever._narrow_topic_value(value, drug_name=drug_name, question=question)

        products = payload.get("products")
        if isinstance(products, dict) and drug_name:
            q = _normalize(drug_name)
            for product_name, product_payload in products.items():
                if not isinstance(product_payload, dict):
                    continue
                if q not in _normalize(product_name):
                    continue
                nested = product_payload.get(topic)
                if isinstance(nested, dict):
                    nested = nested.get("content") or nested
                return JSONRetriever._narrow_topic_value(nested, drug_name=drug_name, question=question)

        topics = payload.get("topics")
        if isinstance(topics, dict):
            nested = topics.get(topic)
            if isinstance(nested, dict):
                nested = nested.get("content") or nested
            return JSONRetriever._narrow_topic_value(nested, drug_name=drug_name, question=question)
        return None

    @staticmethod
    def _stringify_content(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            en = value.get("en") if isinstance(value.get("en"), str) else ""
            fr = value.get("fr") if isinstance(value.get("fr"), str) else ""
            parts = []
            if en.strip():
                parts.append(f"EN: {en.strip()}")
            if fr.strip():
                parts.append(f"FR: {fr.strip()}")
            if parts:
                return "\n".join(parts)
        return json.dumps(value, ensure_ascii=False)

    def find(self, drug_name: str | None, topic: str | None, question: str | None = None) -> JsonRetrievalResult | None:
        """Return JSON-backed answer content when available."""

        if not topic:
            return None

        start = time.perf_counter()
        matches = self._match_payloads(drug_name)
        for payload in matches:
            value = self._extract_topic(payload, topic, drug_name=drug_name, question=question)
            if value in (None, "", {}, []):
                continue
            latency_ms = (time.perf_counter() - start) * 1000.0
            return JsonRetrievalResult(
                source="json",
                content=self._stringify_content(value),
                confidence=1.0,
                latency_ms=latency_ms,
                drug_name=drug_name or payload.get("drug_name"),
                topic=topic,
                source_file=payload.get("__source_file"),
            )

        return None
