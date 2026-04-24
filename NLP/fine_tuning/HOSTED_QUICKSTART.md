# Hosted Fine-Tuning Quickstart

## Dataset Ready ✓

Your v2 dataset is prepared and ready for upload:
- **Train**: `NLP/fine_tuning/data/nlp_sft_v2_train_openai_messages.jsonl` (424 KB, 278 examples)
- **Validation** (optional): `NLP/fine_tuning/data/nlp_sft_v2_val_openai_messages.jsonl` (55 KB, 36 examples)
- **Test** (for eval): `NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl` (52 KB, 34 examples)

Each row is OpenAI messages format with system/user/assistant roles.

---

## Step 1: Choose Your Provider

### Option A: OpenAI
1. Sign in at https://platform.openai.com
2. Go to **Files** (left sidebar)
3. Upload: `NLP/fine_tuning/data/nlp_sft_v2_train_openai_messages.jsonl`
   - Note the **File ID** (e.g., `file-abc123...`)
4. Go to **Fine-tuning** (left sidebar)
5. Click **Create**
   - Model: Choose base model (e.g., `gpt-4o-mini-2024-07-18`)
   - Training file: Select your uploaded file
   - Validation (optional): Upload val split if desired
   - Hyperparameters: Leave defaults or set `n_epochs: 3` for more iterations
6. Click **Create fine-tuning job**
7. Wait for completion status email (typically 30 mins–2 hours)
8. Copy the resulting **Fine-tuned model ID** (e.g., `ft:gpt-4o-mini-2024-07-18:your-org::xxxxx`)

### Option B: Azure OpenAI
1. Sign in at https://portal.azure.com
2. Create/select an OpenAI resource
3. Go to **Model deployments** → **Create new deployment**
4. Upload training file, select model, submit fine-tune job
5. Wait for job completion
6. Copy the deployed endpoint URL and API key

---

## Step 2: Update Config

Copy the template and fill in your provider credentials:

```bash
cp NLP/fine_tuning/hosted_endpoint_config.example.json NLP/fine_tuning/hosted_endpoint_config.json
```

Edit `NLP/fine_tuning/hosted_endpoint_config.json`:

```json
{
  "endpoint_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model_name": "ft:gpt-4o-mini-2024-07-18:your-org::xxxxx",
  "temperature": 0.0,
  "max_tokens": 512
}
```

**Keep your API key secret.** Git will ignore this file (`.gitignore` entry exists).

---

## Step 3: Generate Candidate Predictions

Once your fine-tuned model is ready, generate predictions on the test split:

```bash
python NLP/fine_tuning/generate_candidate_predictions.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --output-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --mode hosted \
  --config-json NLP/fine_tuning/hosted_endpoint_config.json
```

This calls your fine-tuned endpoint for each test example and saves predictions.

Output: `NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl` (34 rows)

---

## Step 4: Evaluate

Compare predictions against test labels:

```bash
python NLP/fine_tuning/evaluate_candidate_outputs.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --candidate-predictions-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --output-json NLP/evaluation/results/ci_finetune_candidate_eval_hosted_v2.json
```

This outputs metrics and a gate status:

```
Reference rows: 34
Matched predictions: 34
Intent accuracy: XX%
Clarification recall: XX%
Entity recall: XX%
QUALITY GATE: PASS/FAIL
```

**Quality gates:**
- Intent accuracy ≥ 0.80
- Clarification recall ≥ 0.80
- Entity recall ≥ 0.45 (most challenging; baseline v2 = 69.70%)
- Coverage ≥ 0.98

---

## Step 5: Package Results

Package the evaluation into CI-ready artifacts with a PR comment:

```bash
python NLP/fine_tuning/package_candidate_eval.py \
  --predictions-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --evaluation-json NLP/evaluation/results/ci_finetune_candidate_eval_hosted_v2.json \
  --output-dir NLP/evaluation/results \
  --prefix ci_finetune_candidate_hosted_v2
```

Output files:
- `ci_finetune_candidate_hosted_v2_predictions.jsonl` (raw predictions archive)
- `ci_finetune_candidate_hosted_v2_eval.json` (metrics)
- `ci_finetune_candidate_hosted_v2_pr_comment.md` (ready for PR)
- `ci_finetune_candidate_hosted_v2_summary.json` (metadata)

---

## Expected Outcome (Baseline Reference)

Current local baseline on v2 dataset:
- Intent accuracy: **94.12%** ✓ (exceeds 80% gate)
- Clarification recall: **100%** ✓ (exceeds 80% gate)
- Entity recall: **69.70%** ✓ (exceeds 45% gate)
- **Gate: PASS** ✓

Your fine-tuned model should meet or exceed these metrics. If it underperforms locally, consider:
- Experimenting with more training epochs on the provider
- Increasing entity-focused examples (regenerate with `--entity-focus-copies 3`)
- Refining the taxonomy for better entity labels

---

## Next Steps After Evaluation

### If gate PASSES:
- ✅ Production candidate ready
- Update CI/CD to use fine-tuned endpoint in production
- Monitor shadow metrics for divergence

### If gate FAILS:
- Review metric breakdown in evaluation JSON
- Iterate: modify dataset → retrain → evaluate
- Common issue: entity recall below threshold → add more entity examples

---

## Troubleshooting

**Q: Provider job is still running after 2 hours**
- A: Check job status in provider dashboard. Some models take 3–4 hours. You can wait or start a new job with different hyperparameters.

**Q: "Connection refused" when generating candidates**
- A: Verify endpoint URL and API key in `hosted_endpoint_config.json`. Check provider dashboard that model is deployed and active.

**Q: Entity recall is very low (< 0.45)**
- A: Entity extraction is hard. Try:
  1. Regenerate dataset with `--entity-focus-copies 3` (more entity examples)
  2. Fine-tune again with more epochs
  3. Review taxonomy for missing entity types

**Q: How do I use the fine-tuned model in production?**
- A: Update `backend/utils/rag_pipeline.py` to call your hosted endpoint instead of local NLP analyzer. Use the same `hosted_endpoint_config.json` mechanism.

---

## Files Reference

- Dataset builder: `NLP/fine_tuning/build_finetune_dataset.py`
- Predictor: `NLP/fine_tuning/generate_candidate_predictions.py`
- Evaluator: `NLP/fine_tuning/evaluate_candidate_outputs.py`
- Packager: `NLP/fine_tuning/package_candidate_eval.py`
- Config template: `NLP/fine_tuning/hosted_endpoint_config.example.json`
- Full README: `NLP/fine_tuning/README.md`

---

## Timeline

1. **Provider job submission**: ~5 mins
2. **Provider training**: 30 mins–2 hours (depending on model size)
3. **Generate candidates**: ~2–5 mins (34 predictions)
4. **Evaluate**: ~1 min
5. **Package**: ~30 secs

**Total time from ready to validated candidate: ~1–2.5 hours** (mostly provider wait time)
