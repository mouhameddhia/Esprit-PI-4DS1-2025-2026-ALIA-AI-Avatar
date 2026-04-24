# NLP Evaluation Drift Monitoring

This directory archives evaluation results over time to detect performance regressions and track quality trends.

## Archive Structure

Each evaluation run creates a timestamped JSON artifact with:
- Dataset version evaluated  
- Timestamp of evaluation
- All quality metrics (intent accuracy, safety precision/recall, etc.)
- Pass/fail status

## Drift Detection Thresholds

Quality gates trigger warnings if metrics drop beyond these tolerances from the current baseline:

- Intent accuracy: ≥ 90% (baseline: v4 = 95.92%)
  - Warning threshold: < 93% (3% margin)
  - Critical: < 90%
  
- Safety recall: ≥ 95% (baseline: v4 = 100%)
  - Warning threshold: < 97% (2% margin)
  - Critical: < 95%
  
- Secondary tags recall: ≥ 60% (baseline: v4 = 74.75%)
  - Warning threshold: < 68% (6% margin from baseline)
  - Critical: < 60%
  
- Entity map recall: ≥ 50% (baseline: v4 = 56.25%)
  - Warning threshold: < 53% (3% margin from baseline)
  - Critical: < 50%

## Phase 3 Shadow Monitoring (v4 vs v5)

Use shadow mode to compare two production classifiers without changing user-facing behavior.

- Primary model: current production routing
- Shadow model: candidate classifier (e.g., v5 keyword expansion path)
- Alert threshold: divergence rate > 15%

### Required Log Schema (JSONL)

Each row should include:
- `text`
- `primary_intent`
- `shadow_intent`

Optional quality fields:
- `primary_confidence`
- `shadow_confidence`
- `ground_truth_intent` (for delayed human labeling)

### Run Shadow Analysis

```bash
python NLP/evaluation/shadow_monitoring.py path/to/shadow_logs.jsonl --max-divergence 0.15
```

Artifact output (default): `NLP/evaluation/results/shadow_latest.json`

### Build Production Shadow Logs

Export real traffic events from MongoDB and recompute shadow predictions:

```bash
python backend/scripts/export_shadow_logs.py --days 7 --limit 2000 --output-jsonl NLP/evaluation/results/shadow_logs_latest.jsonl
```

Then analyze:

```bash
python NLP/evaluation/shadow_monitoring.py NLP/evaluation/results/shadow_logs_latest.jsonl --max-divergence 0.15
```

Build a weekly hard-negative labeling queue from disagreement logs:

```bash
python NLP/evaluation/build_hard_negative_queue.py NLP/evaluation/results/shadow_logs_latest.jsonl --min-pair-count 2 --max-rows 200
```

Artifacts:
- `NLP/evaluation/results/hard_negative_queue_latest.jsonl`
- `NLP/evaluation/results/hard_negative_queue_latest.json`

### CI Proxy Shadow Gate

Because CI cannot query production traffic, it uses labeled datasets as a stable proxy:

```bash
python NLP/evaluation/build_shadow_fixture.py NLP/datasets/eval_intent_safety_v4.jsonl --output-jsonl NLP/evaluation/results/ci_shadow_fixture.jsonl
python NLP/evaluation/shadow_monitoring.py NLP/evaluation/results/ci_shadow_fixture.jsonl --max-divergence 0.15 --output-json NLP/evaluation/results/ci_shadow_eval.json
```

### Daily Backend Automation

Backend startup now schedules a daily snapshot job that writes:
- `NLP/evaluation/results/shadow_logs_latest.jsonl`
- `NLP/evaluation/results/shadow_latest.json`
- Timestamped daily copies (`shadow_logs_YYYYMMDD.jsonl`, `shadow_YYYYMMDD.json`)

Default schedule: **02:00 UTC** (APScheduler cron)

Environment variables:
- `SHADOW_JOB_HOUR_UTC` (default `2`)
- `SHADOW_JOB_MINUTE_UTC` (default `0`)
- `SHADOW_LOG_LOOKBACK_DAYS` (default `7`)
- `SHADOW_LOG_MAX_ROWS` (default `2000`)
- `SHADOW_MAX_DIVERGENCE` (default `0.15`)

### How to Interpret

- Divergence ≤ 15%: acceptable for continued shadow testing
- Divergence > 15%: investigate disagreement clusters before rollout
- If `ground_truth_intent` is present, compare:
  - `primary_intent_accuracy`
  - `shadow_intent_accuracy`
  - `shadow_accuracy_delta`

## Historical Runs

### April 13, 2026 - v4 Baseline Established
- Dataset: eval_intent_safety_v4.jsonl (153 samples)
- Intent accuracy: 95.92% ✓
- Safety precision: 88.89%, recall: 100% ✓
- Secondary tags recall: 74.75% ✓
- Entity map recall: 56.25% ✓
- Status: PASS (all gates)

### April 13, 2026 - Public Supplement Validation
- Dataset: eval_intent_safety_public_supplement.jsonl (30 samples)
- (To be evaluated)
- Status: (Pending)

## How to Use

1. **Monitor new runs**: CI evaluations are automatically logged to `/results/` with timestamp
2. **Check for drift**: Compare latest against v4 baseline metrics
3. **Alert on regression**: If any metric falls below warning threshold, investigate the code change
4. **Review quarterly**: Pull full archive and plot trends to identify gradual drift

## Separate Gate Profiles

Run production baseline with strict thresholds:

```bash
python NLP/evaluation/run_eval.py NLP/datasets/eval_intent_safety_v5.jsonl --threshold-profile production --output-json NLP/evaluation/results/eval_v5_latest.json
```

Run public-domain hardening with a separate gate profile:

```bash
python NLP/evaluation/run_eval.py NLP/datasets/eval_intent_safety_public_supplement.jsonl --threshold-profile public_hardening --output-json NLP/evaluation/results/eval_public_supplement_latest.json
```

Available profiles in `run_eval.py`:
- `production` (strict release gate)
- `public_hardening` (iterative expansion gate)

## Entity Quality Reporting

`run_eval.py` now reports both:

- exact entity metrics (`entity_map_precision`, `entity_map_recall`)
- partial entity metrics (`partial_entity_map_precision`, `partial_entity_map_recall`)

Use exact metrics for strict release gates and partial metrics to track progress on
generalization where phrasing differs but semantic extraction is close.

## Promotion Gate

Use the candidate promotion gate to automate canary/rollback discipline:

```bash
python NLP/evaluation/promote_candidate.py \
  --baseline-eval NLP/evaluation/results/eval_v4_latest.json \
  --candidate-eval NLP/evaluation/results/eval_v5_latest.json \
  --shadow-eval NLP/evaluation/results/shadow_latest.json \
  --min-intent-delta 0.0 \
  --min-safety-delta 0.0 \
  --min-entity-delta 0.0 \
  --max-shadow-divergence 0.15 \
  --output-json NLP/evaluation/results/promotion_decision_latest.json
```

Gate behavior:

- hard-stop on any safety recall regression (`min_safety_delta >= 0.0`)
- requires candidate eval gate pass and shadow gate pass
- blocks promotion if shadow divergence exceeds threshold

Recommended pragmatic rollout policy:

- `min_intent_delta = -0.10` (allow bounded intent drop while other dimensions improve)
- `min_safety_delta = 0.0` (never allow safety recall regression)
- `min_entity_delta = 0.0` (require non-regression on entity recall)
- `max_shadow_divergence = 0.15`

## Clarification Quality Gate

Run the ambiguity/clarification benchmark to enforce low-confidence behavior:

```bash
python NLP/evaluation/run_clarification_eval.py NLP/datasets/eval_intent_clarification_v1.jsonl \
  --min-clarification-recall 0.80 \
  --min-clarification-precision 0.70 \
  --min-clear-intent-accuracy 0.85 \
  --output-json NLP/evaluation/results/eval_clarification_latest.json
```

This gate ensures:

- low-confidence requests are caught and routed to clarification (`recall`)
- clearly classifiable requests are not over-clarified (`precision`)
- intent quality stays high on clear messages (`clear_intent_accuracy`)

## Clarification KPI Tracking

Generate production KPI snapshots from stored `nlp_events`:

```bash
python backend/scripts/report_clarification_kpis.py --days 7 \
  --output-json NLP/evaluation/results/clarification_kpi_latest.json
```

Reported KPIs include:

- `clarification_rate`
- `follow_up_resolution_rate`
- unresolved follow-up count
- mode-level breakdown (`physician_portal`, `medrep_training`)

## Retrieval Rerank Quality Gate

Run reranker quality checks on a labeled candidate benchmark:

```bash
python NLP/evaluation/run_retrieval_rerank_eval.py NLP/datasets/eval_retrieval_rerank_v3.jsonl \
  --min-rerank-hit-at-1 0.70 \
  --min-rerank-mrr 0.85 \
  --max-hit-at-1-regression 0.01 \
  --output-json NLP/evaluation/results/eval_retrieval_rerank_latest.json
```

This gate ensures reranking does not regress retrieval hit quality while improving ordering of relevant chunks.

## Language Detection Quality Gate

Run multilingual detection checks on a labeled runtime benchmark:

```bash
python NLP/evaluation/run_language_detection_eval.py NLP/datasets/eval_language_detection_v1.jsonl \
  --min-accuracy 0.90 \
  --output-json NLP/evaluation/results/eval_language_detection_latest.json
```

This gate ensures runtime language tagging remains stable for analytics and multilingual policy routing.

## Retriever + Generation Regression Checks

Run static regression checks that validate critical prompt and retriever wiring:

```bash
python NLP/evaluation/run_retriever_generation_regression.py \
  --output-json NLP/evaluation/results/eval_retriever_generation_regression_latest.json
```

Current checks cover:

- reranker integration path in retrieval pipeline
- retrieval anti-verbatim guardrail text
- physician safety guardrail in generation prompt
- clarification prompt scaffolding and mode switch behavior
- language logging in chat NLP events

## Retrieval Label Review Workflow

Build a reviewer queue from real-label rows that are not yet verified:

```bash
python NLP/evaluation/build_retrieval_review_queue.py \
  --dataset-jsonl NLP/datasets/eval_retrieval_rerank_real_v1.jsonl \
  --output-jsonl NLP/evaluation/results/retrieval_review_queue_latest.jsonl \
  --output-json NLP/evaluation/results/retrieval_review_coverage_latest.json
```

Check coverage of human-verified labels:

```bash
python NLP/evaluation/check_retrieval_review_coverage.py \
  --dataset-jsonl NLP/datasets/eval_retrieval_rerank_real_v1.jsonl \
  --min-verified-ratio 0.20 \
  --output-json NLP/evaluation/results/retrieval_review_gate_latest.json
```

Recommended policy:

- treat coverage gate as informational until the team reaches >=20% verified rows
- raise threshold to 35% then 50% over subsequent releases

## Weekly Clarification Review Loop

Run this weekly cadence to keep clarification behavior stable while reducing noise:

1. Refresh offline clarification benchmark:

```bash
python NLP/evaluation/run_clarification_eval.py NLP/datasets/eval_intent_clarification_v1.jsonl \
  --min-clarification-recall 0.80 \
  --min-clarification-precision 0.70 \
  --min-clear-intent-accuracy 0.85 \
  --output-json NLP/evaluation/results/eval_clarification_latest.json
```

2. Refresh production KPI snapshot:

```bash
python backend/scripts/report_clarification_kpis.py --days 7 \
  --output-json NLP/evaluation/results/clarification_kpi_latest.json
```

3. Triage false positives from benchmark failures:

- prioritize rows where `type = clarification` and `expected = false`
- add recurring patterns into intent heuristics or hard-negative override lists
- add 5-10 new labeled rows for newly discovered ambiguity patterns

4. Promotion discipline:

- keep recall floor (`>= 0.80`) fixed
- only tighten precision threshold after two consecutive weekly passes

## Next Steps

- [ ] Correlate evaluation regressions with code changes (git blame integration)
- [ ] Set up automated alerts for critical threshold breaches
- [ ] Build visualization dashboard for stakeholder updates
- [ ] Archive results from each PR/branch for comparative analysis
- [ ] Feed top shadow disagreement pairs into weekly relabel/retraining queue
