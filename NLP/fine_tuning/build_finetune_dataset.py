#!/usr/bin/env python3
"""Build instruction-tuning datasets from existing NLP labeled benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _safe_entity_map(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for key, vals in value.items():
        if not isinstance(key, str):
            continue
        cleaned = _safe_list(vals)
        if cleaned:
            out[key] = cleaned
    return out


def _build_system_prompt(taxonomy: Dict[str, Any]) -> str:
    intents = taxonomy.get("intents") if isinstance(taxonomy.get("intents"), list) else []
    safety_flags = taxonomy.get("safety_flags") if isinstance(taxonomy.get("safety_flags"), list) else []
    entity_types = taxonomy.get("entity_types") if isinstance(taxonomy.get("entity_types"), list) else []

    return (
        "You are an NLP labeler. Return ONLY valid JSON. "
        "Classify intent, safety flags, secondary tags, and entity map, and decide if clarification is needed. "
        "Use only taxonomy values.\n"
        f"Allowed intents: {intents}\n"
        f"Allowed safety_flags: {safety_flags}\n"
        f"Allowed entity types: {entity_types}\n"
        "Output schema: "
        '{"intent": str, "safety_flags": [str], "secondary_tags": [str], '
        '"entity_map": {str:[str]}, "needs_clarification": bool}'
    )


def _build_intent_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        text = row.get("text")
        mode = row.get("mode", "physician_portal")
        intent = row.get("expected_intent", "other")
        if not isinstance(text, str) or not text.strip() or not isinstance(intent, str):
            continue

        example = {
            "task": "intent_safety_entities",
            "mode": str(mode),
            "text": text,
            "target": {
                "intent": intent,
                "safety_flags": _safe_list(row.get("expected_safety_flags")),
                "secondary_tags": _safe_list(row.get("expected_secondary_tags")),
                "entity_map": _safe_entity_map(row.get("expected_entity_map")),
                "needs_clarification": False,
            },
            "stratify_label": f"intent::{intent}",
        }
        out.append(example)
    return out


def _build_clarification_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        text = row.get("text")
        mode = row.get("mode", "physician_portal")
        expected_clarification = row.get("expected_clarification")
        if not isinstance(text, str) or not text.strip() or not isinstance(expected_clarification, bool):
            continue

        intent = row.get("expected_intent", "other")
        intent_clean = intent if isinstance(intent, str) and intent.strip() else "other"

        example = {
            "task": "clarification_decision",
            "mode": str(mode),
            "text": text,
            "target": {
                "intent": intent_clean if not expected_clarification else "other",
                "safety_flags": [],
                "secondary_tags": [],
                "entity_map": {},
                "needs_clarification": expected_clarification,
            },
            "stratify_label": f"clarification::{str(expected_clarification).lower()}",
        }
        out.append(example)
    return out


def _build_entity_focus_examples(rows: List[Dict[str, Any]], copies: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if copies <= 0:
        return out

    prompt_variants = [
        "Focus on the entity map and return the exact structured JSON.",
        "Extract the entity map precisely; keep the full JSON schema intact.",
        "Prioritize entity extraction and preserve the same output schema.",
    ]

    for row in rows:
        text = row.get("text")
        mode = row.get("mode", "physician_portal")
        intent = row.get("expected_intent", "other")
        entity_map = _safe_entity_map(row.get("expected_entity_map"))
        if not isinstance(text, str) or not text.strip() or not isinstance(intent, str) or not entity_map:
            continue

        for copy_idx in range(copies):
            variant = prompt_variants[copy_idx % len(prompt_variants)]
            example = {
                "task": "entity_focus_json",
                "mode": str(mode),
                "text": text,
                "target": {
                    "intent": intent,
                    "safety_flags": _safe_list(row.get("expected_safety_flags")),
                    "secondary_tags": _safe_list(row.get("expected_secondary_tags")),
                    "entity_map": entity_map,
                    "needs_clarification": False,
                },
                "prompt_hint": variant,
                "stratify_label": f"entity_focus::{intent}",
            }
            out.append(example)
    return out


def _stratified_split(
    rows: List[Dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get("stratify_label", "unknown"))
        buckets[label].append(row)

    rnd = random.Random(seed)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for label in sorted(buckets.keys()):
        bucket = list(buckets[label])
        rnd.shuffle(bucket)
        n = len(bucket)

        if n == 1:
            train_n, val_n = 1, 0
        elif n == 2:
            train_n, val_n = 1, 1
        else:
            train_n = max(1, int(round(n * train_ratio)))
            val_n = max(1, int(round(n * val_ratio)))
            if train_n + val_n >= n:
                val_n = max(1, min(val_n, n - train_n - 1))

        train.extend(bucket[:train_n])
        val.extend(bucket[train_n : train_n + val_n])
        test.extend(bucket[train_n + val_n :])

    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(test)
    return train, val, test


def _to_openai_messages(row: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    user_prompt = (
        f"Mode: {row['mode']}\n"
        f"Message: {row['text']}\n"
        "Return only JSON using the required schema."
    )
    assistant = json.dumps(row["target"], ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "task": row["task"],
            "mode": row["mode"],
            "stratify_label": row["stratify_label"],
        },
    }


def _to_alpaca(row: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    instruction = system_prompt
    input_text = f"Mode: {row['mode']}\nMessage: {row['text']}"
    output_text = json.dumps(row["target"], ensure_ascii=False)
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "meta": {
            "task": row["task"],
            "mode": row["mode"],
            "stratify_label": row["stratify_label"],
        },
    }


def _distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get("stratify_label", "unknown"))] += 1
    return dict(sorted(counter.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fine-tuning datasets from NLP benchmark JSONL files")
    parser.add_argument("--intent-dataset-jsonl", default="NLP/datasets/eval_intent_safety_v5.jsonl")
    parser.add_argument("--clarification-dataset-jsonl", default="NLP/datasets/eval_intent_clarification_v1.jsonl")
    parser.add_argument("--taxonomy-json", default="NLP/taxonomy/nlp_taxonomy.json")
    parser.add_argument("--output-dir", default="NLP/fine_tuning/data")
    parser.add_argument("--prefix", default="nlp_sft_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-entity-focus", action="store_true", help="Add extra entity-focused examples")
    parser.add_argument("--entity-focus-copies", type=int, default=1, help="Number of entity-focused variants per entity-heavy row")
    parser.add_argument("--format", choices=["openai_messages", "alpaca", "both"], default="both")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        print("Invalid ratios: require train_ratio > 0, val_ratio > 0, train+val < 1")
        return 1

    intent_path = Path(args.intent_dataset_jsonl).expanduser().resolve()
    clarification_path = Path(args.clarification_dataset_jsonl).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy_json).expanduser().resolve()

    if not intent_path.exists():
        print(f"Intent dataset not found: {intent_path}")
        return 1
    if not clarification_path.exists():
        print(f"Clarification dataset not found: {clarification_path}")
        return 1
    if not taxonomy_path.exists():
        print(f"Taxonomy not found: {taxonomy_path}")
        return 1

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    system_prompt = _build_system_prompt(taxonomy if isinstance(taxonomy, dict) else {})

    intent_rows = _read_jsonl(intent_path)
    clarification_rows = _read_jsonl(clarification_path)

    examples = _build_intent_examples(intent_rows) + _build_clarification_examples(clarification_rows)
    if args.include_entity_focus:
        examples.extend(_build_entity_focus_examples(intent_rows, copies=max(1, args.entity_focus_copies)))
    if not examples:
        print("No examples were generated")
        return 1

    train_rows, val_rows, test_rows = _stratified_split(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_base = out_dir / f"{args.prefix}_train"
    val_base = out_dir / f"{args.prefix}_val"
    test_base = out_dir / f"{args.prefix}_test"
    meta_path = out_dir / f"{args.prefix}_metadata.json"

    if args.format in {"openai_messages", "both"}:
        _write_jsonl(Path(str(train_base) + "_openai_messages.jsonl"), [_to_openai_messages(row, system_prompt) for row in train_rows])
        _write_jsonl(Path(str(val_base) + "_openai_messages.jsonl"), [_to_openai_messages(row, system_prompt) for row in val_rows])
        _write_jsonl(Path(str(test_base) + "_openai_messages.jsonl"), [_to_openai_messages(row, system_prompt) for row in test_rows])

    if args.format in {"alpaca", "both"}:
        _write_jsonl(Path(str(train_base) + "_alpaca.jsonl"), [_to_alpaca(row, system_prompt) for row in train_rows])
        _write_jsonl(Path(str(val_base) + "_alpaca.jsonl"), [_to_alpaca(row, system_prompt) for row in val_rows])
        _write_jsonl(Path(str(test_base) + "_alpaca.jsonl"), [_to_alpaca(row, system_prompt) for row in test_rows])

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intent_dataset": str(intent_path),
        "clarification_dataset": str(clarification_path),
        "taxonomy": str(taxonomy_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "counts": {
            "total_examples": len(examples),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "entity_focus_examples": len([row for row in examples if str(row.get("task", "")).startswith("entity_focus")]),
        },
        "distribution": {
            "all": _distribution(examples),
            "train": _distribution(train_rows),
            "val": _distribution(val_rows),
            "test": _distribution(test_rows),
        },
        "format": args.format,
        "files": {
            "output_dir": str(out_dir),
            "prefix": args.prefix,
        },
    }

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total examples: {len(examples)}")
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
