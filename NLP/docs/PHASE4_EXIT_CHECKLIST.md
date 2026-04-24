# Phase 4 Exit Checklist

This checklist defines the objective criteria to declare **Phase 4 (Full-Fledged NLP)** complete.

## Scope

Phase 4 completion requires all of the following to be green in one CI run:

1. Trained intent holdout shadow gate passes.
2. Trained entity holdout shadow gate passes.
3. Trained reranker holdout gate passes.
4. Drift/retraining governance is healthy (`retraining_required=false`).
5. Version snapshot is complete (all tracked assets exist).
6. Intent and entity ratchet artifacts are generated and healthy.
7. Legacy production quality gates remain green (eval, shadow, clarification, language, retrieval, regression, review coverage).

## Required Artifacts

- `NLP/evaluation/results/ci_eval.json`
- `NLP/evaluation/results/ci_shadow_eval.json`
- `NLP/evaluation/results/ci_clarification_eval.json`
- `NLP/evaluation/results/ci_retrieval_rerank_eval.json`
- `NLP/evaluation/results/ci_language_eval.json`
- `NLP/evaluation/results/ci_retriever_generation_regression.json`
- `NLP/evaluation/results/ci_retrieval_review_gate.json`
- `NLP/evaluation/results/ci_retraining_trigger.json`
- `NLP/evaluation/results/ci_version_snapshot.json`
- `NLP/evaluation/results/ci_trained_intent_shadow_eval.json`
- `NLP/evaluation/results/ci_trained_entity_shadow_eval.json`
- `NLP/evaluation/results/ci_trained_rerank_shadow_eval.json`
- `NLP/evaluation/results/ci_intent_shadow_thresholds_next.json`
- `NLP/evaluation/results/ci_entity_shadow_thresholds_next.json`
- `NLP/evaluation/results/ci_phase4_exit_gate.json` (summary output)

## CI Gate Script

Use:

```bash
python NLP/evaluation/phase4_exit_gate.py \
  --output-json NLP/evaluation/results/ci_phase4_exit_gate.json
```

Gate policy:

- **PASS**: all Phase 4 checks pass.
- **FAIL**: any required check/artifact is missing or unhealthy.

## Operational Sign-off

Phase 4 can be marked complete when:

1. `ci_phase4_exit_gate.json` reports `quality_gate: pass`.
2. Ratchet workflow is active for both intent and entity thresholds.
3. Team accepts thresholds and rollback posture.
