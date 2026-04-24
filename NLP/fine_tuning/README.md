# Fine-Tuning Starter

This folder contains utilities to prepare provider-agnostic fine-tuning datasets
from existing ALIA NLP labeled benchmarks.

## Hardware & Environment

- **Hosted fine-tuning (recommended for production)**: No GPU required. Works in any environment.
- **Local LoRA training**: Requires NVIDIA GPU with CUDA. CPU-only machines are not practical for real training; use hosted endpoints instead.
- **Current local environment**: CPU-only (PyTorch 2.11.0+cpu, torch.cuda.is_available()=False). Use hosted routes for actual training.

## Dataset Builder

Script: `NLP/fine_tuning/build_finetune_dataset.py`

Builds train/val/test splits and exports one or both formats:

- OpenAI messages format (`*_openai_messages.jsonl`)
- Alpaca instruction format (`*_alpaca.jsonl`)

## Example

```bash
python NLP/fine_tuning/build_finetune_dataset.py \
  --intent-dataset-jsonl NLP/datasets/eval_intent_safety_v5.jsonl \
  --clarification-dataset-jsonl NLP/datasets/eval_intent_clarification_v1.jsonl \
  --taxonomy-json NLP/taxonomy/nlp_taxonomy.json \
  --output-dir NLP/fine_tuning/data \
  --prefix nlp_sft_v1 \
  --format both
```

## Outputs

- `NLP/fine_tuning/data/nlp_sft_v1_train_openai_messages.jsonl`
- `NLP/fine_tuning/data/nlp_sft_v1_val_openai_messages.jsonl`
- `NLP/fine_tuning/data/nlp_sft_v1_test_openai_messages.jsonl`
- `NLP/fine_tuning/data/nlp_sft_v1_train_alpaca.jsonl`
- `NLP/fine_tuning/data/nlp_sft_v1_val_alpaca.jsonl`
- `NLP/fine_tuning/data/nlp_sft_v1_test_alpaca.jsonl`
- `NLP/fine_tuning/data/nlp_sft_v1_metadata.json`

## Notes

- This is data-prep only; no model training happens here.
- Use the generated files with hosted fine-tuning APIs or local HF/LoRA pipelines.
- Keep existing CI gates as promotion criteria for any fine-tuned candidate.

## Next Steps

### 1) Local LoRA starter (GPU-only, optional)

⚠️ **Requires GPU with CUDA support. Skip this section if you don't have a GPU; use hosted fine-tuning instead.**

Install local training dependencies:

```bash
pip install -r NLP/fine_tuning/requirements-local-lora.txt
```

Run LoRA SFT starter:

```bash
python NLP/fine_tuning/train_lora_intent_sft.py \
  --train-jsonl NLP/fine_tuning/data/nlp_sft_v2_train_openai_messages.jsonl \
  --val-jsonl NLP/fine_tuning/data/nlp_sft_v2_val_openai_messages.jsonl \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --output-dir NLP/fine_tuning/models/intent_lora_v2
```

Note: Local training can take 1–6 hours depending on model size, batch size, and number of epochs. GPU with 8GB+ VRAM recommended.

For 4GB GPUs (for example GTX 1650), use QLoRA with a smaller base model:

```bash
python NLP/fine_tuning/train_lora_intent_sft.py \
  --train-jsonl NLP/fine_tuning/data/nlp_sft_v2_train_openai_messages.jsonl \
  --val-jsonl NLP/fine_tuning/data/nlp_sft_v2_val_openai_messages.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir NLP/fine_tuning/models/intent_qlora_v2 \
  --use-qlora \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-seq-len 384
```

### 2) Hosted fine-tuning route (OpenAI, Azure, or compatible)

This is the recommended path for production fine-tuning. No local GPU required.

#### Step 2a: Export the data

Use the generated OpenAI-messages format for upload to your fine-tuning provider:

```bash
python NLP/fine_tuning/build_finetune_dataset.py \
  --intent-dataset-jsonl NLP/datasets/eval_intent_safety_v5.jsonl \
  --clarification-dataset-jsonl NLP/datasets/eval_intent_clarification_v1.jsonl \
  --taxonomy-json NLP/taxonomy/nlp_taxonomy.json \
  --output-dir NLP/fine_tuning/data \
  --prefix nlp_sft_v2 \
  --format both
```

The exported files:
- `NLP/fine_tuning/data/nlp_sft_v2_train_openai_messages.jsonl` (278 rows, use this for training)
- `NLP/fine_tuning/data/nlp_sft_v2_val_openai_messages.jsonl` (36 rows, optional validation)
- `NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl` (34 rows, use for candidate evaluation after training)

Each row follows the OpenAI messages format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an NLP analyzer."},
    {"role": "user", "content": "Mode: physician_portal\nMessage: What is the dosage for Product K?"},
    {"role": "assistant", "content": "{\"intent\": \"dosage_question\", \"needs_clarification\": false, ...}"}
  ]
}
```

#### Step 2b: Set up hosted endpoint config

Copy the config template and fill in your provider details:

```bash
copy NLP\fine_tuning\hosted_endpoint_config.example.json NLP\fine_tuning\hosted_endpoint_config.json
```

Edit `NLP/fine_tuning/hosted_endpoint_config.json`:

```json
{
  "endpoint_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model_name": "ft:gpt-4o-mini-2024-07-18:your-org::...",
  "temperature": 0.0,
  "max_tokens": 512
}
```

For OpenAI fine-tuning:
- Use the Files API to upload `nlp_sft_v2_train_openai_messages.jsonl`
- Submit a fine-tuning job with that file ID
- Wait for job completion (typically 30 mins to 2 hours)
- Copy the resulting fine-tuned model ID and update the config above

#### Step 2c: Generate candidate predictions

Once your hosted model is ready, generate predictions on the test split:

```bash
python NLP/fine_tuning/generate_candidate_predictions.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --output-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --mode hosted \
  --config-json NLP/fine_tuning/hosted_endpoint_config.json
```

This calls your fine-tuned endpoint for each test example and collects predictions.

#### Step 2d: Evaluate candidate predictions

Evaluate the hosted candidate against the test labels:

```bash
python NLP/fine_tuning/evaluate_candidate_outputs.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --candidate-predictions-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --output-json NLP/evaluation/results/ci_finetune_candidate_eval_hosted_v2.json
```

Expected thresholds (gated):
- `min-intent-accuracy 0.80`
- `min-clarification-recall 0.80`
- `min-entity-recall 0.45` (entity recall is the most challenging metric; v2 entity-focused dataset baseline achieves ~70%)
- `min-coverage 0.98`

#### Step 2e: Package the results

After evaluation passes, package outputs for PR/CI artifact reporting:

```bash
python NLP/fine_tuning/package_candidate_eval.py \
  --predictions-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --evaluation-json NLP/evaluation/results/ci_finetune_candidate_eval_hosted_v2.json \
  --output-dir NLP/evaluation/results \
  --prefix ci_finetune_candidate_hosted_v2
```

This generates:
- `ci_finetune_candidate_hosted_v2_predictions.jsonl` (for artifact archive)
- `ci_finetune_candidate_hosted_v2_eval.json` (evaluation metrics)
- `ci_finetune_candidate_hosted_v2_pr_comment.md` (formatted summary for PR)
- `ci_finetune_candidate_hosted_v2_summary.json` (metadata)

### 3) Candidate evaluation bridge

Expected candidate prediction row schema (returned by hosted endpoint or local baseline):

```json
{
  "mode": "physician_portal",
  "text": "What is the dosage for Product K?",
  "prediction": {
    "intent": "dosage_question",
    "needs_clarification": false,
    "safety_flags": [],
    "secondary_tags": [],
    "entity_map": {"product_name": ["Product K"], "dosage": ["dosage"]}
  }
}
```

Evaluation gates all candidates against the same thresholds used in Phase 4 NLP evaluation, ensuring fine-tuned models meet the same rigor as the baseline pipeline.

### 4) Local baseline smoke test

For quick validation without waiting for hosted training:

```bash
python NLP/fine_tuning/generate_candidate_predictions.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --output-jsonl NLP/fine_tuning/data/candidate_predictions_v2_local_baseline.jsonl \
  --mode local_baseline
```

Then evaluate:

```bash
python NLP/fine_tuning/evaluate_candidate_outputs.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --candidate-predictions-jsonl NLP/fine_tuning/data/candidate_predictions_v2_local_baseline.jsonl \
  --output-json NLP/evaluation/results/ci_finetune_candidate_eval_v2_local_baseline.json
```

Current local baseline (v2 entity-focused dataset) passes all gates: intent_accuracy=0.9412, clarification_recall=1.0, entity_recall=0.697.

### 5) Package evaluation for CI/PRs

After running the evaluator (hosted or local), package outputs into CI-ready artifacts and a PR comment:

```bash
python NLP/fine_tuning/package_candidate_eval.py \
  --predictions-jsonl NLP/fine_tuning/data/candidate_predictions_v2_local_baseline.jsonl \
  --evaluation-json NLP/evaluation/results/ci_finetune_candidate_eval_v2_local_baseline.json \
  --output-dir NLP/evaluation/results \
  --prefix ci_finetune_candidate_v2
```

The main NLP CI workflow (`.github/workflows/nlp-eval.yml`) now runs the packaging step against the checked-in v2 local baseline smoke test and uploads the packaged artifacts on every PR/push.

### 6) Entity-heavy tuning pass

To bias the fine-tuning set toward entity extraction, regenerate with entity-focused examples:

```bash
python NLP/fine_tuning/build_finetune_dataset.py \
  --intent-dataset-jsonl NLP/datasets/eval_intent_safety_v5.jsonl \
  --clarification-dataset-jsonl NLP/datasets/eval_intent_clarification_v1.jsonl \
  --taxonomy-json NLP/taxonomy/nlp_taxonomy.json \
  --output-dir NLP/fine_tuning/data \
  --prefix nlp_sft_v2 \
  --format both \
  --include-entity-focus \
  --entity-focus-copies 2
```

## Full Pipeline Example (Hosted Route)

For a complete end-to-end hosted fine-tuning workflow:

```bash
# 1. Build dataset (v2 with entity focus)
python NLP/fine_tuning/build_finetune_dataset.py \
  --intent-dataset-jsonl NLP/datasets/eval_intent_safety_v5.jsonl \
  --clarification-dataset-jsonl NLP/datasets/eval_intent_clarification_v1.jsonl \
  --taxonomy-json NLP/taxonomy/nlp_taxonomy.json \
  --output-dir NLP/fine_tuning/data \
  --prefix nlp_sft_v2 \
  --format both \
  --include-entity-focus \
  --entity-focus-copies 2

# 2. [Manual] Upload train split to your provider and submit fine-tuning job
#    Provider: OpenAI, Azure, or compatible endpoint
#    File: NLP/fine_tuning/data/nlp_sft_v2_train_openai_messages.jsonl
#    Wait for job completion and copy the fine-tuned model ID

# 3. Update config with your fine-tuned model endpoint
cp NLP/fine_tuning/hosted_endpoint_config.example.json NLP/fine_tuning/hosted_endpoint_config.json
# Edit NLP/fine_tuning/hosted_endpoint_config.json with your credentials

# 4. Generate predictions from the fine-tuned model
python NLP/fine_tuning/generate_candidate_predictions.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --output-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --mode hosted \
  --config-json NLP/fine_tuning/hosted_endpoint_config.json

# 5. Evaluate predictions against test labels
python NLP/fine_tuning/evaluate_candidate_outputs.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --candidate-predictions-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --output-json NLP/evaluation/results/ci_finetune_candidate_eval_hosted_v2.json

# 6. Package results into CI artifacts
python NLP/fine_tuning/package_candidate_eval.py \
  --predictions-jsonl NLP/fine_tuning/data/candidate_predictions_hosted_v2.jsonl \
  --evaluation-json NLP/evaluation/results/ci_finetune_candidate_eval_hosted_v2.json \
  --output-dir NLP/evaluation/results \
  --prefix ci_finetune_candidate_hosted_v2

# Outputs available in NLP/evaluation/results/
```

## 7) Local Adapter Hybrid Inference (Recommended for fine-tuned models)

For locally trained adapters, use hybrid inference mode to combine adapter predictions with a robust baseline fallback. This significantly improves intent and entity accuracy:

```bash
# Generate predictions using local adapter + baseline hybrid mode
python NLP/fine_tuning/generate_candidate_predictions.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --output-jsonl NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_hybrid.jsonl \
  --mode local_adapter \
  --adapter-dir NLP/fine_tuning/models/intent_qlora_v6_1p5b_multiepoch \
  --max-new-tokens 220 \
  --taxonomy-json NLP/taxonomy/nlp_taxonomy.json \
  --hybrid-with-baseline
```

The `--hybrid-with-baseline` flag merges adapter predictions with local baseline signals:
- **Intent**: Prefers baseline intent when it is not "other" (generic fallback)
- **Entity**: Combines adapter + baseline entity maps for improved recall
- **Clarification/Safety**: Uses adapter output with taxonomy projection constraints

Evaluate and package:

```bash
python NLP/fine_tuning/evaluate_candidate_outputs.py \
  --reference-openai-jsonl NLP/fine_tuning/data/nlp_sft_v2_test_openai_messages.jsonl \
  --candidate-predictions-jsonl NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_hybrid.jsonl \
  --output-json NLP/evaluation/results/ci_finetune_candidate_eval_hybrid.json

python NLP/fine_tuning/package_candidate_eval.py \
  --predictions-jsonl NLP/fine_tuning/data/candidate_predictions_qlora_v6_1p5b_hybrid.jsonl \
  --evaluation-json NLP/evaluation/results/ci_finetune_candidate_eval_hybrid.json \
  --output-dir NLP/evaluation/results \
  --prefix ci_finetune_candidate_hybrid
```

**Latest validated results** (v2 and v3 test sets, 1.5B adapter + hybrid inference):
- Intent accuracy: 91.18% ✓
- Clarification recall: 100.00% ✓
- Entity recall: 72.73% ✓
- Gate status: **PASS**

## Troubleshooting

**Q: My hosted endpoint times out during candidate generation.**
- A: Check your endpoint URL and API credentials in `hosted_endpoint_config.json`. Verify network access to the endpoint.

**Q: Candidate evaluation shows low entity recall (below 0.45 gate).**
- A: Entity extraction is challenging. Try regenerating with `--entity-focus-copies 3` or higher to bias the training set more toward entity examples.

**Q: Local LoRA training is very slow or not starting.**
- A: You likely have a CPU-only environment (torch.cuda.is_available()=False). Use hosted fine-tuning instead; it's faster and more practical.

**Q: Where are the CI/PR artifacts uploaded?**
- A: The `.github/workflows/nlp-eval.yml` workflow packages results and creates artifacts. Check your GitHub Actions run logs and the artifacts tab for `nlp-eval-results.zip`.

## References

- Dataset builder: `NLP/fine_tuning/build_finetune_dataset.py`
- Local LoRA trainer: `NLP/fine_tuning/train_lora_intent_sft.py`
- Candidate predictor: `NLP/fine_tuning/generate_candidate_predictions.py`
- Evaluator: `NLP/fine_tuning/evaluate_candidate_outputs.py`
- Packager: `NLP/fine_tuning/package_candidate_eval.py`
- Config template: `NLP/fine_tuning/hosted_endpoint_config.example.json`
- Requirements: `NLP/fine_tuning/requirements-local-lora.txt`
- CI workflow: `.github/workflows/nlp-eval.yml`
