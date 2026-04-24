# Entity Extraction v2: Enhanced Pharmaceutical Domain Recognition

## Overview

Entity Extraction v2 improves baseline recall from **72.73%** by combining:

1. **Pharmaceutical Domain Dictionary** – comprehensive product, molecule, dosage, and indication catalog
2. **Fuzzy Matching** – handles misspellings and partial matches
3. **NER Model Fallback** – optional spaCy/transformers integration for unknown entities
4. **Intelligent Merging** – deduplicates and prioritizes high-confidence predictions

## Key Improvements

### Dictionary-First Approach

- Exact matching against 15+ pharmaceutical products
- 7+ molecules (omega-3, vitamins, minerals)
- Dosage pattern recognition (150ml, 30 gélules, etc.)
- Disease/indication keyword matching

### Fuzzy Matching

- Tolerates misspellings (e.g., "Omevie" → "Omévie")
- Handles alternative forms (e.g., "omega 3" ↔ "omega-3")
- Configurable similarity threshold (default 75%)
- Dual implementation: fuzzywuzzy (primary) + difflib fallback

### NER Fallback (Optional)
- Integrates spaCy French model for out-of-dictionary entities
- Assigns lower confidence to NER predictions
- Merges with dictionary matches to catch edge cases

## Usage

### Basic Extract (Dictionary + Fuzzy)

```python
from NLP.pipeline.entity_extractor_v2 import extract_entities

text = "Patient asks about Omevie Oméga 3 dosage for cardiovascular support"
entities = extract_entities(text, use_fuzzy=True, use_spacy=False)

# Output:
# {
#   "product": [
#     {"value": "Omevie Oméga 3", "entity_id": "omevie_omega_3", "confidence": 1.0, "source": "fuzzy_match"}
#   ],
#   "molecule": [...],
#   "dosage": [...],
#   "indication": [...]
# }
```

### With NER Fallback

```python
# Requires: pip install spacy && python -m spacy download fr_core_news_sm
from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2

extractor = EntityExtractorV2(use_spacy=True)
entities = extractor.extract_entities(text, use_fuzzy=True, use_ner=True)
```

### Integration with Existing NLP Pipeline

```python
# In backend/utils/nlp.py or similar:
from NLP.pipeline.entity_extractor_v2 import extract_entities

def analyze_message_nlp(user_text, mode, history):
    # ... existing code ...
    # Enhanced entity extraction
    entities = extract_entities(user_text, use_fuzzy=True, use_spacy=False)
    entity_map = {
        "product": [e["value"] for e in entities.get("product", [])],
        "molecule": [e["value"] for e in entities.get("molecule", [])],
        "dosage": [e["value"] for e in entities.get("dosage", [])],
        "indication": [e["value"] for e in entities.get("indication", [])],
    }
    return {
        "intent": ...,
        "entity_map": entity_map,
        # ... other fields ...
    }
```

## Evaluation

Run evaluation on a test dataset:

```bash
python NLP/evaluation/eval_entity_extraction_v2.py \
  NLP/datasets/eval_intent_safety_v5.jsonl \
  --min-recall 0.80 \
  --output-json NLP/evaluation/results/eval_entity_extraction_v2.json
```

Expected improvements:
- **Baseline (exact match)**: ~72.73% recall
- **+Fuzzy matching**: +5-8% recall improvement
- **+NER fallback**: +2-3% additional recall

## Configuration

### Fuzzy Matching Threshold

```python
extractor.dictionary.fuzzy_match(text, threshold=75)  # Default 75%
```

- 80+: very similar
- 75-79: likely misspelling
- 65-74: partial/alternative form
- <65: too different, risky

### Disable NER for Production (faster)

```python
# Fast path (no spaCy overhead)
extractor = EntityExtractorV2(use_spacy=False)
entities = extractor.extract_entities(text, use_fuzzy=True, use_ner=False)
```

### Load Custom Dictionary

```python
from pathlib import Path
from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2

custom_dict = Path("path/to/custom_dict.json")
extractor = EntityExtractorV2(dict_path=custom_dict)
```

## Extending the Dictionary

Add new products, molecules, or indications to `NLP/taxonomy/pharmaceutical_dictionary.json`:

```json
{
  "products": {
    "new_product_id": {
      "names": ["Product Name", "Alternative Name"],
      "category": "category_name",
      "form": "capsule|liquid|syrup"
    }
  },
  "molecules": {
    "new_molecule_id": {
      "names": ["Molecule Name"],
      "synonyms": ["synonym1", "synonym2"],
      "category": "molecule_category"
    }
  }
}
```

Then re-run the index build (automatic on initialization).

## Performance Notes

- **Dictionary matching**: O(n) where n = number of tokens
- **Fuzzy matching**: O(n*m) where m = dictionary size (~80 entries)
- **spaCy NER**: ~50-100ms per inference
- **Total latency**: 10-20ms (dictionary+fuzzy) + 50-100ms (optional NER)

For production, recommend dictionary+fuzzy without NER unless recall target requires it.

## Testing Fuzzy Matching

```bash
python -c "\
from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2
extractor = EntityExtractorV2()

test_cases = [
    'omevie omega 3',  # Alternative form
    'Omevie Omega3',   # No hyphen
    'OMEVIE',          # All caps
    'omvie oméga',     # Misspelling
]

for text in test_cases:
    match = extractor.dictionary.fuzzy_match(text)
    if match:
        entity_type, entity_id, entity_data, score = match
        print(f'{text:20} -> {entity_id:25} (score: {score}%)')
    else:
        print(f'{text:20} -> NO MATCH')
"
```

## Dependencies

- `fuzzywuzzy` + `python-Levenshtein` (optional, for fuzzy matching)
- `spacy` (optional, for NER fallback)

```bash
pip install fuzzywuzzy python-Levenshtein
pip install spacy
python -m spacy download fr_core_news_sm
```

## Next Steps

1. Integrate into backend NLP pipeline
2. Run evaluation on production evaluation sets
3. Monitor entity recall metric in shadow monitoring
4. Expand pharmaceutical dictionary as new products are encountered
5. Consider training domain-specific NER model if recall gap persists# Entity Extraction v2: Enhanced Pharmaceutical Domain Recognition

## Overview

Entity Extraction v2 improves baseline recall from **72.73%** by combining:

1. **Pharmaceutical Domain Dictionary** – comprehensive product, molecule, dosage, and indication catalog
2. **Fuzzy Matching** – handles misspellings and partial matches
3. **NER Model Fallback** – optional spaCy/transformers integration for unknown entities
4. **Intelligent Merging** – deduplicates and prioritizes high-confidence predictions

## Key Improvements

### Dictionary-First Approach

- Exact matching against 15+ pharmaceutical products
- 7+ molecules (omega-3, vitamins, minerals)
- Dosage pattern recognition (150ml, 30 gélules, etc.)
- Disease/indication keyword matching

### Fuzzy Matching

- Tolerates misspellings (e.g., "Omevie" → "Omévie")
- Handles alternative forms (e.g., "omega 3" ↔ "omega-3")
- Configurable similarity threshold (default 75%)
- Dual implementation: fuzzywuzzy (primary) + difflib fallback

### NER Fallback (Optional)
- Integrates spaCy French model for out-of-dictionary entities
- Assigns lower confidence to NER predictions
- Merges with dictionary matches to catch edge cases

## Usage

### Basic Extract (Dictionary + Fuzzy)

```python
from NLP.pipeline.entity_extractor_v2 import extract_entities

text = "Patient asks about Omevie Oméga 3 dosage for cardiovascular support"
entities = extract_entities(text, use_fuzzy=True, use_spacy=False)

# Output:
# {
#   "product": [
#     {"value": "Omevie Oméga 3", "entity_id": "omevie_omega_3", "confidence": 1.0, "source": "fuzzy_match"}
#   ],
#   "molecule": [...],
#   "dosage": [...],
#   "indication": [...]
# }
```

### With NER Fallback

```python
# Requires: pip install spacy && python -m spacy download fr_core_news_sm
from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2

extractor = EntityExtractorV2(use_spacy=True)
entities = extractor.extract_entities(text, use_fuzzy=True, use_ner=True)
```

### Integration with Existing NLP Pipeline

```python
# In backend/utils/nlp.py or similar:
from NLP.pipeline.entity_extractor_v2 import extract_entities

def analyze_message_nlp(user_text, mode, history):
    # ... existing code ...
    # Enhanced entity extraction
    entities = extract_entities(user_text, use_fuzzy=True, use_spacy=False)
    entity_map = {
        "product": [e["value"] for e in entities.get("product", [])],
        "molecule": [e["value"] for e in entities.get("molecule", [])],
        "dosage": [e["value"] for e in entities.get("dosage", [])],
        "indication": [e["value"] for e in entities.get("indication", [])],
    }
    return {
        "intent": ...,
        "entity_map": entity_map,
        # ... other fields ...
    }
```

## Evaluation

Run evaluation on a test dataset:

```bash
python NLP/evaluation/eval_entity_extraction_v2.py \
  NLP/datasets/eval_intent_safety_v5.jsonl \
  --min-recall 0.80 \
  --output-json NLP/evaluation/results/eval_entity_extraction_v2.json
```

Expected improvements:
- **Baseline (exact match)**: ~72.73% recall
- **+Fuzzy matching**: +5-8% recall improvement
- **+NER fallback**: +2-3% additional recall

## Configuration

### Fuzzy Matching Threshold

```python
extractor.dictionary.fuzzy_match(text, threshold=75)  # Default 75%
```

- 80+: very similar
- 75-79: likely misspelling
- 65-74: partial/alternative form
- <65: too different, risky

### Disable NER for Production (faster)

```python
# Fast path (no spaCy overhead)
extractor = EntityExtractorV2(use_spacy=False)
entities = extractor.extract_entities(text, use_fuzzy=True, use_ner=False)
```

### Load Custom Dictionary

```python
from pathlib import Path
from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2

custom_dict = Path("path/to/custom_dict.json")
extractor = EntityExtractorV2(dict_path=custom_dict)
```

## Extending the Dictionary

Add new products, molecules, or indications to `NLP/taxonomy/pharmaceutical_dictionary.json`:

```json
{
  "products": {
    "new_product_id": {
      "names": ["Product Name", "Alternative Name"],
      "category": "category_name",
      "form": "capsule|liquid|syrup"
    }
  },
  "molecules": {
    "new_molecule_id": {
      "names": ["Molecule Name"],
      "synonyms": ["synonym1", "synonym2"],
      "category": "molecule_category"
    }
  }
}
```

Then re-run the index build (automatic on initialization).

## Performance Notes

- **Dictionary matching**: O(n) where n = number of tokens
- **Fuzzy matching**: O(n*m) where m = dictionary size (~80 entries)
- **spaCy NER**: ~50-100ms per inference
- **Total latency**: 10-20ms (dictionary+fuzzy) + 50-100ms (optional NER)

For production, recommend dictionary+fuzzy without NER unless recall target requires it.

## Testing Fuzzy Matching

```bash
python -c "\
from NLP.pipeline.entity_extractor_v2 import EntityExtractorV2
extractor = EntityExtractorV2()

test_cases = [
    'omevie omega 3',  # Alternative form
    'Omevie Omega3',   # No hyphen
    'OMEVIE',          # All caps
    'omvie oméga',     # Misspelling
]

for text in test_cases:
    match = extractor.dictionary.fuzzy_match(text)
    if match:
        entity_type, entity_id, entity_data, score = match
        print(f'{text:20} -> {entity_id:25} (score: {score}%)')
    else:
        print(f'{text:20} -> NO MATCH')
"
```

## Dependencies

- `fuzzywuzzy` + `python-Levenshtein` (optional, for fuzzy matching)
- `spacy` (optional, for NER fallback)

```bash
pip install fuzzywuzzy python-Levenshtein
pip install spacy
python -m spacy download fr_core_news_sm
```

## Next Steps

1. Integrate into backend NLP pipeline
2. Run evaluation on production evaluation sets
3. Monitor entity recall metric in shadow monitoring
4. Expand pharmaceutical dictionary as new products are encountered
5. Consider training domain-specific NER model if recall gap persists