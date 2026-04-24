#!/usr/bin/env python3
"""Starter script for local LoRA SFT on generated OpenAI-messages datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role and content:
            lines.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(lines).strip()


def _messages_to_chat_template(tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
    # Prefer the model's native chat template for train/infer consistency.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        except Exception:
            pass
    return _messages_to_text(messages)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local LoRA SFT model on OpenAI-messages JSONL")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="NLP/fine_tuning/models/intent_lora_v1")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--use-qlora", action="store_true", help="Load base model in 4-bit for QLoRA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl.trainer.sft_trainer import SFTTrainer
    except Exception as exc:
        print("Missing dependencies for local LoRA training.")
        print("Install with: pip install transformers datasets peft trl accelerate bitsandbytes")
        print(f"Import error: {exc}")
        return 1

    try:
        import torch
    except Exception as exc:
        print(f"Missing torch dependency: {exc}")
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available in this Python environment.")
        print("Install CUDA-enabled torch first, then retry.")
        return 1

    train_path = Path(args.train_jsonl).expanduser().resolve()
    val_path = Path(args.val_jsonl).expanduser().resolve()
    if not train_path.exists() or not val_path.exists():
        print(f"Dataset path missing: train={train_path.exists()} val={val_path.exists()}")
        return 1

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)

    train_examples = [
        {"messages": row.get("messages", [])}
        for row in train_rows
        if isinstance(row.get("messages"), list) and row.get("messages")
    ]
    val_examples = [
        {"messages": row.get("messages", [])}
        for row in val_rows
        if isinstance(row.get("messages"), list) and row.get("messages")
    ]

    if not train_examples or not val_examples:
        print("No valid training/validation message rows found in messages JSONL")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(args.max_seq_len)

    load_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "dtype": torch.float16,
    }

    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            print("QLoRA requested, but BitsAndBytesConfig is unavailable.")
            print("Install/upgrade with: pip install bitsandbytes")
            print(f"Import error: {exc}")
            return 1

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        print("No valid target modules provided for LoRA")
        return 1

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples),
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=lambda example: _messages_to_chat_template(tokenizer, example.get("messages", [])),
    )

    trainer.train()
    trainer.save_model(str(Path(args.output_dir).expanduser().resolve()))
    tokenizer.save_pretrained(str(Path(args.output_dir).expanduser().resolve()))

    print(f"Saved LoRA model to: {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
