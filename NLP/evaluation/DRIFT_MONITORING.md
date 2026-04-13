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

## Next Steps

- [ ] Correlate evaluation regressions with code changes (git blame integration)
- [ ] Set up automated alerts for critical threshold breaches
- [ ] Build visualization dashboard for stakeholder updates
- [ ] Archive results from each PR/branch for comparative analysis
