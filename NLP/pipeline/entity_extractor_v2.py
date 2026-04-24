#!/usr/bin/env python3
"""Enhanced entity extraction with fuzzy matching, dictionary lookup, and NER fallback.

Combines:
1. Pharmaceutical domain dictionary with exact/fuzzy matching
2. Optional spaCy/transformers NER model fallback
3. Intelligent merging of dictionary + model predictions
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

fuzz = None
spacy = None

try:
    from fuzzywuzzy import fuzz as _fuzz

    fuzz = _fuzz
    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False

try:
    import spacy as _spacy

    spacy = _spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


REPO_ROOT = Path(__file__).resolve().parents[2]


class PharmaceuticalDictionary:
    """Pharmaceutical domain dictionary with fuzzy matching."""

    def __init__(self, dict_path: Optional[Path] = None):
        if dict_path is None:
            dict_path = REPO_ROOT / "NLP" / "taxonomy" / "pharmaceutical_dictionary.json"
        self.dict_path = dict_path
        self.data: Dict[str, Any] = {}
        self.name_to_entity: Dict[str, Tuple[str, str]] = {}  # Maps normalized name -> (entity_type, entity_id)
        self._load()

    def _load(self) -> None:
        if not self.dict_path.exists():
            return
        try:
            self.data = json.loads(self.dict_path.read_text(encoding="utf-8"))
            self._build_name_index()
        except Exception as e:
            print(f"Warning: failed to load pharmaceutical dictionary: {e}")

    def _build_name_index(self) -> None:
        """Build a reverse index: normalized name -> entity info."""
        for entity_type in ["products", "molecules"]:
            if entity_type not in self.data:
                continue
            entities = self.data.get(entity_type, {})
            if not isinstance(entities, dict):
                continue
            for entity_id, entity_data in entities.items():
                if not isinstance(entity_data, dict):
                    continue
                names = entity_data.get("names", [])
                if not isinstance(names, list):
                    continue
                for name in names:
                    if isinstance(name, str):
                        normalized = self._normalize(name)
                        self.name_to_entity[normalized] = (entity_type, entity_id)

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        return re.sub(r'\s+', ' ', text.strip().lower())

    def exact_match(self, text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Exact match against dictionary."""
        normalized = self._normalize(text)
        key = self.name_to_entity.get(normalized)
        if key:
            entity_type, entity_id = key
            entity_data = self.data[entity_type].get(entity_id, {})
            return (entity_type, entity_id, entity_data)
        return None

    def fuzzy_match(self, text: str, threshold: int = 80) -> Optional[Tuple[str, str, Dict[str, Any], int]]:
        """Fuzzy match against dictionary using sequence matching or fuzzywuzzy."""
        if not HAS_FUZZYWUZZY or fuzz is None:
            return self._fuzzy_match_fallback(text, threshold)

        normalized_text = self._normalize(text)
        if self._looks_like_noise(normalized_text):
            return None
        best_match = None
        best_score = 0

        for indexed_name, (entity_type, entity_id) in self.name_to_entity.items():
            assert fuzz is not None
            score = fuzz.token_set_ratio(normalized_text, indexed_name)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (entity_type, entity_id, score)

        if best_match:
            entity_type, entity_id, score = best_match
            entity_data = self.data[entity_type].get(entity_id, {})
            return (entity_type, entity_id, entity_data, score)
        return None

    def _fuzzy_match_fallback(self, text: str, threshold: int = 80) -> Optional[Tuple[str, str, Dict[str, Any], int]]:
        """Fallback fuzzy matching using difflib."""
        normalized_text = self._normalize(text)
        if self._looks_like_noise(normalized_text):
            return None
        best_match = None
        best_score = 0

        for indexed_name, (entity_type, entity_id) in self.name_to_entity.items():
            ratio = SequenceMatcher(None, normalized_text, indexed_name).ratio()
            score = int(ratio * 100)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (entity_type, entity_id, score)

        if best_match:
            entity_type, entity_id, score = best_match
            entity_data = self.data[entity_type].get(entity_id, {})
            return (entity_type, entity_id, entity_data, score)
        return None

    def _looks_like_noise(self, normalized_text: str) -> bool:
        """Reject stopword-heavy windows that commonly create fuzzy false positives."""
        stopwords = {
            "a", "an", "and", "about", "can", "do", "for", "from", "how",
            "i", "in", "is", "it", "me", "of", "ok", "on", "or", "please",
            "talk", "tell", "the", "this", "to", "we", "what", "you"
        }
        tokens = [token for token in normalized_text.split() if token]
        if not tokens:
            return True
        if len(tokens) == 1:
            return False
        if tokens[0] in stopwords or tokens[-1] in stopwords:
            return True
        if sum(token in stopwords for token in tokens) >= len(tokens) - 1:
            return True
        return False

    def extract_dosage_patterns(self, text: str) -> List[str]:
        """Extract dosage-like patterns from text."""
        normalized_text = text.lower()
        dosage_patterns: Set[str] = set()
        amount_with_unit: Set[str] = set()

        def normalize_unit(unit: str) -> str:
            unit = unit.strip().lower()
            if unit in {"g", "gram", "grams"}:
                return "g"
            if unit in {"mg"}:
                return "mg"
            if unit in {"ml"}:
                return "ml"
            if unit in {"capsule", "capsules"}:
                return "capsules"
            if unit in {"gélule", "gélules", "gelule", "gelules"}:
                return "gélules"
            if unit in {"comprimé", "comprimés", "comprime", "comprises", "tablet", "tablets", "tab"}:
                return "comprimés"
            return unit

        for match in re.finditer(
            r"\b(\d{1,4})\s*(mg|ml|g|gélule|gélules|gelule|gelules|capsule|capsules|comprimé|comprimés|comprime|tablet|tablets|tab|sirop|syrup|liquid)\b",
            normalized_text,
            re.IGNORECASE,
        ):
            amount = match.group(1)
            unit = normalize_unit(match.group(2))
            amount_with_unit.add(amount)
            dosage_patterns.add(f"{amount} {unit}".strip())

        for match in re.finditer(r"\bb[\/-]?(\d{1,4})\b", normalized_text, re.IGNORECASE):
            amount = match.group(1)
            if amount not in amount_with_unit:
                dosage_patterns.add(amount)

        return sorted(dosage_patterns)


class EntityExtractorV2:
    """Enhanced entity extractor combining dictionary + fuzzy matching + optional NER."""

    def __init__(self, dict_path: Optional[Path] = None, use_spacy: bool = False):
        self.dictionary = PharmaceuticalDictionary(dict_path)
        self.spacy_model = None
        if use_spacy and HAS_SPACY and spacy is not None:
            try:
                self.spacy_model = spacy.load("fr_core_news_sm")
            except OSError:
                print("Warning: spaCy French model not found; NER fallback disabled")

    def extract_entities(self, text: str, use_fuzzy: bool = True, use_ner: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using all available methods."""
        if not text or not isinstance(text, str):
            return {}

        extracted: Dict[str, List[Dict[str, Any]]] = {
            "product": [],
            "molecule": [],
            "dosage": [],
            "indication": [],
        }

        # Extract using dictionary + fuzzy matching
        dict_entities = self._extract_with_dictionary(text, use_fuzzy)
        for entity_type, entities in dict_entities.items():
            if entity_type in extracted and isinstance(entities, list):
                extracted[entity_type].extend(entities)

        # Extract dosage patterns
        dosages = self.dictionary.extract_dosage_patterns(text)
        for dosage in dosages:
            extracted["dosage"].append({"value": dosage, "source": "pattern"})

        # Optional: NER model fallback
        if use_ner and self.spacy_model:
            ner_entities = self._extract_with_ner(text)
            for entity_type, entities in ner_entities.items():
                if entity_type in extracted and isinstance(entities, list):
                    extracted[entity_type].extend(entities)

        # Deduplicate and merge
        for entity_type in extracted:
            extracted[entity_type] = self._deduplicate_entities(extracted[entity_type])

        return extracted

    def _extract_with_dictionary(self, text: str, use_fuzzy: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using dictionary matching."""
        extracted: Dict[str, List[Dict[str, Any]]] = {
            "product": [],
            "molecule": [],
            "indication": [],
        }

        # Extract products and molecules with longest-match-first strategy
        tokens = self._tokenize_for_matching(text)
        for i, token in enumerate(tokens):
            # Try longest match first
            for length in range(min(4, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i : i + length])
                match = self.dictionary.exact_match(phrase)
                if match:
                    entity_type, entity_id, entity_data = match
                    if entity_type == "products":
                        extracted["product"].append({
                            "value": phrase,
                            "entity_id": entity_id,
                            "confidence": 1.0,
                            "source": "exact_match",
                        })
                        inferred_indication = self._infer_indication_from_product(entity_id, entity_data)
                        if inferred_indication:
                            extracted["indication"].append({
                                "value": inferred_indication,
                                "type": inferred_indication,
                                "confidence": 0.85,
                                "source": "product_category",
                            })
                    elif entity_type == "molecules":
                        extracted["molecule"].append({
                            "value": phrase,
                            "entity_id": entity_id,
                            "confidence": 1.0,
                            "source": "exact_match",
                        })
                    break

        # Fuzzy matching for incomplete/misspelled matches
        if use_fuzzy:
            for i, token in enumerate(tokens):
                for length in range(min(3, len(tokens) - i), 0, -1):
                    phrase = " ".join(tokens[i : i + length])
                    match = self.dictionary.fuzzy_match(phrase, threshold=75)
                    if match and len(phrase) > 3:  # Avoid very short phrases
                        entity_type, entity_id, entity_data, score = match
                        if entity_type == "products":
                            extracted["product"].append({
                                "value": phrase,
                                "entity_id": entity_id,
                                "confidence": score / 100.0,
                                "source": "fuzzy_match",
                            })
                            inferred_indication = self._infer_indication_from_product(entity_id, entity_data)
                            if inferred_indication:
                                extracted["indication"].append({
                                    "value": inferred_indication,
                                    "type": inferred_indication,
                                    "confidence": 0.8,
                                    "source": "product_category",
                                })
                        elif entity_type == "molecules":
                            extracted["molecule"].append({
                                "value": phrase,
                                "entity_id": entity_id,
                                "confidence": score / 100.0,
                                "source": "fuzzy_match",
                            })
                        break

        # Extract indications using keywords
        disease_indications = self.dictionary.data.get("diseases_indications", {})
        for indication_type, indication_data in disease_indications.items():
            keywords = indication_data.get("keywords", [])
            names = indication_data.get("names", [])
            canonical_value = indication_type.split("_")[0]
            matched = False

            for candidate in list(names) + list(keywords):
                if isinstance(candidate, str) and candidate.lower() in text.lower():
                    matched = True

            if matched:
                extracted["indication"].append({
                    "value": canonical_value,
                    "type": indication_type,
                    "confidence": 0.9,
                    "source": "keyword_match",
                })

        return extracted

    def _infer_indication_from_product(self, entity_id: str, entity_data: Dict[str, Any]) -> Optional[str]:
        """Infer a canonical indication label from a matched product."""
        category = str(entity_data.get("category", "")).lower()
        entity_id = entity_id.lower()

        if any(token in category for token in ["cardio", "omega", "heart"]):
            return "cardiovascular"
        if any(token in category for token in ["pregnancy", "maternal", "preconception", "allaitement", "grossesse"]):
            return "pregnancy"
        if any(token in category for token in ["pediatric", "baby", "child", "enfant"]):
            return "pediatric"
        if any(token in category for token in ["cough", "respiratory", "toux", "bronch", "asthma"]):
            return "respiratory"
        if any(token in category for token in ["energy", "iron", "vitamin", "zinc", "selenium", "mineral", "antioxidant", "fatigue"]):
            return "energy"
        if any(token in category for token in ["stress", "sleep", "calm", "sleep"]):
            return "stress"

        if any(token in entity_id for token in ["omega_3", "omevie"]):
            return "cardiovascular"
        if any(token in entity_id for token in ["grossesse", "allaitement", "conception"]):
            return "pregnancy"
        if any(token in entity_id for token in ["pediakids"]):
            return "pediatric"
        if any(token in entity_id for token in ["pulmax", "apitou"]):
            return "respiratory"
        if any(token in entity_id for token in ["zinc", "vitamin", "fersang", "oligovit", "q10"]):
            return "energy"

        return None

    def _extract_with_ner(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using spaCy NER model."""
        extracted: Dict[str, List[Dict[str, Any]]] = {
            "product": [],
            "molecule": [],
            "indication": [],
        }

        if not self.spacy_model:
            return extracted

        doc = self.spacy_model(text)
        for ent in doc.ents:
            if ent.label_ in ("MISC", "PRODUCT"):
                extracted["product"].append({
                    "value": ent.text,
                    "confidence": 0.7,
                    "source": "spacy_ner",
                })
            elif ent.label_ in ("GPE", "WORK_OF_ART"):
                extracted["molecule"].append({
                    "value": ent.text,
                    "confidence": 0.6,
                    "source": "spacy_ner",
                })

        return extracted

    def _tokenize_for_matching(self, text: str) -> List[str]:
        """Tokenize text while preserving case-insensitive matching."""
        return re.split(r'[\s\-,;.!?]+', text.lower())

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and merge entities from multiple sources."""
        if not entities:
            return []

        seen: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            value_normalized = re.sub(r'\s+', ' ', entity.get("value", "").strip().lower())
            if value_normalized not in seen:
                seen[value_normalized] = entity
            else:
                # Merge: keep highest confidence or prefer exact match
                existing = seen[value_normalized]
                new_conf = entity.get("confidence", 0.5)
                existing_conf = existing.get("confidence", 0.5)
                existing_source = existing.get("source", "")
                new_source = entity.get("source", "")

                if new_source == "exact_match" or new_conf > existing_conf:
                    seen[value_normalized] = entity
                elif new_conf == existing_conf and new_source not in existing_source:
                    seen[value_normalized]["source"] = f"{existing_source}+{new_source}"

        return list(seen.values())


def extract_entities(text: str, use_fuzzy: bool = True, use_spacy: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function for entity extraction."""
    extractor = EntityExtractorV2(use_spacy=use_spacy)
    return extractor.extract_entities(text, use_fuzzy=use_fuzzy, use_ner=use_spacy)