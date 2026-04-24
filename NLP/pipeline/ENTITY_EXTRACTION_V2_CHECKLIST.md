# Entity Extraction v2 Implementation Checklist

## ✅ COMPLETED TASKS

### Core Implementation
- [x] **Pharmaceutical domain dictionary** created (`NLP/taxonomy/pharmaceutical_dictionary.json`)
  - 15+ products with synonyms
  - 7+ molecules with variants
  - Dosage patterns (ml, mg, gélules, etc.)
  - 6 disease indication categories

- [x] **Entity extractor module** created (`NLP/pipeline/entity_extractor_v2.py`)
  - PharmaceuticalDictionary class with exact matching
  - Fuzzy matching with fuzzywuzzy (+ difflib fallback)
  - Optional spaCy NER integration
  - Deduplication and merging logic
  - Public API: `extract_entities()` function

- [x] **Evaluation framework** created (`NLP/evaluation/eval_entity_extraction_v2.py`)
  - 3-method comparison (exact, fuzzy, fuzzy+NER)
  - Precision/recall/F1 metrics
  - Quality gate (min_recall default 0.80)
  - JSON output artifact format
  - Sample failure collection for debugging

### Documentation
- [x] **Integration guide** (`NLP/pipeline/ENTITY_EXTRACTION_V2_README.md`)
  - Basic usage examples
  - Configuration options
  - Dictionary extension instructions
  - Performance notes
  - Dependency list

- [x] **Implementation summary** (`NLP/pipeline/ENTITY_EXTRACTION_V2_SUMMARY.md`)
  - Components overview
  - Expected improvements table
  - Quick start guide
  - Design rationale
  - Performance characteristics
  - Known limitations
  - Future enhancements

- [x] **Demo script** (`NLP/pipeline/test_entity_extraction_v2.py`)
  - 8 pharmaceutical domain test cases
  - Fuzzy match quality demonstration
  - Standalone validation (no full eval needed)

- [x] **Dependency requirements** (`NLP/pipeline/requirements-entity-extraction-v2.txt`)
  - fuzzywuzzy, python-Levenshtein (primary)
  - spacy (optional)

- [x] **Phase 4 handoff update** (`NLP/docs/PHASE4_HANDOFF_SUMMARY.md`)
  - Added "Post-Promotion Improvements" section
  - Documented expected improvements
  - Noted integration path

### File Count
- **New files**: 7
- **Updated files**: 1
- **Total additions**: ~1400 lines of code + documentation
\n## 🔄 PENDING INTEGRATION TASKS

### Phase 1: Local Validation (Your Machine)
- [ ] **Install dependencies**
  ```bash
  pip install -r NLP/pipeline/requirements-entity-extraction-v2.txt
  python -m spacy download fr_core_news_sm  # Optional
  ```
  Expected time: 2-5 minutes

- [ ] **Run demo script**
  ```bash
  python NLP/pipeline/test_entity_extraction_v2.py
  ```
  Expected output: 8 test cases + fuzzy scoring demo\n  Expected time: 30 seconds

- [ ] **Validate dictionary loads correctly**\n  Check that no errors occur during extraction\n  Expected time: Included in demo run\n\n### Phase 2: Evaluation (Baseline Measurement)\n- [ ] **Run full evaluation on v5 dataset**
  ```bash
  python NLP/evaluation/eval_entity_extraction_v2.py \\\n    NLP/datasets/eval_intent_safety_v5.jsonl \\\n    --min-recall 0.80 \\\n    --output-json NLP/evaluation/results/eval_entity_extraction_v2.json\n  ```\n  Expected output: JSON with 3-method comparison\n  Expected time: 5-15 minutes\n  Expected recall: 78-83% (vs 72.73% baseline)\n\n- [ ] **Review evaluation results**\n  - Check if best method meets 80% recall gate\n  - Review sample_failures for missing dictionary entries\n  - Decide: use fuzzy only or add NER fallback\n\n### Phase 3: Integration (Backend Connection)
- [ ] **Integrate into NLP pipeline** (one of two approaches)\n+  \n  **Option A: Minimal** (Recommended)\n  - Add import in `backend/utils/nlp.py`\n  - Replace existing entity extraction with v2\n  - No config changes needed\n  - Risk: Low\n  \n  **Option B: Advanced** (Future)\n  - Create feature flag: `ALIA_USE_ENTITY_V2`\n  - Keep old extractor as fallback\n  - Canary test with 10% traffic first\n  - Risk: Medium (requires deployment flag logic)\n  \n- [ ] **Test integration locally**\n  - Run existing integration tests\n  - Verify no regressions on intent/clarification\n  - Check entity extraction on sample messages\n\n- [ ] **Update backend dependencies**\n  - Add fuzzywuzzy to `backend/requirements.txt`\n  - Consider python-Levenshtein (optional speedup)\n  - Consider spacy (optional, if NER needed)\n\n### Phase 4: Validation & Promotion\n- [ ] **Run Phase 4 exit gate re-evaluation**\n  - Ensure intent accuracy ≥ 91%\n  - Ensure clarification ≥ 100%\n  - Ensure entity recall now ≥ 80%\n  - Ensure coverage = 100%\n  - Update Phase 4 exit gate JSON with new metrics\n\n- [ ] **Deploy to staging/canary**\n  - Use feature flag if implemented\n  - Monitor entity recall metric\n  - Check for latency regression (<30ms target)\n  - Collect shadow monitoring data\n\n- [ ] **Promote to production**\n  - Update default config to use v2\n  - Deploy with monitoring alerts\n  - Monitor entity recall metric for 24-48 hours\n  - Roll back if needed\n\n### Phase 5: Expansion (Optional)\n- [ ] **Expand pharmaceutical dictionary**\n  - Add products from customer feedback\n  - Add new molecules/vitamins\n  - Add regional/country-specific products\n\n- [ ] **Train domain-specific NER model** (future)\n  - Collect labeled pharmaceutical corpus\n  - Fine-tune spaCy or transformers model\n  - Evaluate on pharmaceutical entities\n\n- [ ] **Add monitoring dashboards**\n  - Track entity recall over time\n  - Alert on recall drops\n  - Monitor average extraction latency\n\n## 📋 FILES CHECKLIST\n\n### Core Code\n- [x] `NLP/pipeline/entity_extractor_v2.py` (400+ lines)\n- [x] `NLP/taxonomy/pharmaceutical_dictionary.json` (150+ lines)\n- [x] `NLP/evaluation/eval_entity_extraction_v2.py` (250+ lines)\n\n### Documentation\n- [x] `NLP/pipeline/ENTITY_EXTRACTION_V2_README.md` (100+ lines)\n- [x] `NLP/pipeline/ENTITY_EXTRACTION_V2_SUMMARY.md` (250+ lines)\n- [x] `NLP/pipeline/ENTITY_EXTRACTION_V2_CHECKLIST.md` (This file)\n- [x] `NLP/docs/PHASE4_HANDOFF_SUMMARY.md` (Updated)\n\n### Tools & Configs\n- [x] `NLP/pipeline/test_entity_extraction_v2.py` (150+ lines)\n- [x] `NLP/pipeline/requirements-entity-extraction-v2.txt` (5 lines)\n\n### Total\n- **Total new files**: 7\n- **Total updated files**: 1\n- **Total lines of code**: ~1100\n- **Total lines of docs**: ~600\n\n## 🎯 SUCCESS CRITERIA\n\n### Immediate (Local Validation)\n- ✓ Demo script runs without errors\n- ✓ Fuzzy matching scores output correctly\n- ✓ Dictionary loads with ~80 indexed entities\n\n### Intermediate (Evaluation)\n- ✓ Evaluation script completes\n- ✓ Best method reaches ≥80% recall\n- ✓ Sample failures identified for dictionary expansion\n\n### Final (Production)\n- ✓ Entity extraction integrated into backend\n- ✓ Phase 4 exit gate passes with entity recall ≥80%\n- ✓ No regression on intent/clarification metrics\n- ✓ Latency impact <30ms per request\n- ✓ Monitoring alerts configured\n\n## ⏱️ ESTIMATED TIMELINE\n\n| Phase | Task | Duration | Blocker? |\n|-------|------|----------|----------|\n| 1 | Install dependencies | 2-5 min | No |\n| 1 | Run demo script | 30 sec | No |\n| 2 | Full evaluation run | 5-15 min | No |\n| 2 | Review results | 10-15 min | **YES** (decide fuzzy vs NER) |\n| 3 | Backend integration | 30-60 min | No |\n| 3 | Local testing | 15-30 min | No |\n| 4 | Phase 4 re-eval | 10-20 min | No |\n| 4 | Staging deployment | 15-30 min | No |\n| 4 | Production promotion | 15-30 min | No |\n| **Total** | **All phases** | **2-3 hours** | - |\n\n## 💡 DECISION POINTS\n\n### Decision 1: Fuzzy Matching Threshold\n**Current default**: 75%\n- 80+: Very similar (safe)\n- 75-79: Likely misspelling (default)\n- 65-74: Partial/alternative form (risky)\n- <65: Too different (skip)\n\n**Recommendation**: Keep default 75%. Adjust down to 70 only if recall below 78%.\n\n### Decision 2: Enable NER Fallback?\n**Baseline (dictionary+exact)**: 72.73% recall, 5ms latency\n**+Fuzzy matching**: ~78-80% recall, 10-20ms latency\n**+NER fallback**: ~80-83% recall, 60-120ms latency\n\n**Recommendation**: Start with dictionary+fuzzy (10-20ms). Enable NER only if:\n- Recall goal > 80%\n- Latency budget > 100ms\n- Evaluation shows NER adds ≥2% recall\n\n### Decision 3: Feature Flag or Direct Replacement?\n**Option A (Recommended)**: Replace existing extractor directly\n- Simpler deployment\n- No flag logic to maintain\n- Immediate benefit to all users\n- Risk: Need thorough evaluation first\n\n**Option B (Advanced)**: Feature flag + canary\n- Safer rollout\n- Easy rollback if issues\n- Allows A/B testing\n- Risk: Complex deployment logic\n\n**Recommendation**: Go with Option A if evaluation shows ≥80% recall. Use Option B only if <80% and need iterative improvement.\n\n## 🔗 RELATED DOCUMENTS\n\n- `NLP/docs/PHASE4_HANDOFF_SUMMARY.md` - Phase 4 context\n- `NLP/pipeline/ENTITY_EXTRACTION_V2_README.md` - Integration details\n- `NLP/pipeline/entity_extractor_v2.py` - Source code\n- `NLP/evaluation/eval_entity_extraction_v2.py` - Evaluation code\n- `NLP/taxonomy/pharmaceutical_dictionary.json` - Domain data\n\n## 📞 SUPPORT\n\n**Issue**: Dictionary doesn't have product X\n→ Add to `pharmaceutical_dictionary.json` and re-index\n\n**Issue**: Fuzzy match threshold too high/low\n→ Adjust threshold parameter in `extract_entities()` call\n\n**Issue**: Latency too high\n→ Disable NER fallback (use_ner=False)\n\n**Issue**: Recall still <80%\n→ Expand pharmaceutical_dictionary.json with missing entities\n→ Review sample_failures from evaluation JSON\n\n## ✋ SIGN-OFF\n\n**Implementation Status**: ✅ **COMPLETE**\n\nAll core components delivered and documented.\nReady for local validation → evaluation → integration → production promotion.\n\n**Next User Action**: Run demo script to validate local setup.\n***\n