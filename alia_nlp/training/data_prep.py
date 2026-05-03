"""Convert labeled JSONL to chat-format for QLoRA fine-tuning."""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYSTEM_PROMPT = (
    "You are an NLP extraction engine for a pharmaceutical assistant. "
    "Return ONLY valid JSON with keys: intent, needs_clarification, secondary_tags, "
    "entities, entity_map, topics, objections, action_items, safety_flags, "
    "rewritten_query, confidence."
)


def convert(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    rows = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for row in rows:
        label = {
            "intent":          row.get("expected_intent", "other"),
            "needs_clarification": False,
            "secondary_tags":  row.get("expected_secondary_tags", []),
            "entities":        [],
            "entity_map":      row.get("expected_entity_map", {}),
            "topics":          [],
            "objections":      [],
            "action_items":    [],
            "safety_flags":    row.get("expected_safety_flags", []),
            "rewritten_query": row.get("text", ""),
            "confidence":      0.9,
        }
        out.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"Mode: {row.get('mode','physician_portal')}\nMessage: {row.get('text','')}"},
                {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
            ]
        })
    output_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    print(f"Converted {len(out)} rows → {output_path}")


if __name__ == "__main__":
    base = REPO_ROOT / "alia_nlp" / "data"
    convert(base / "raw" / "eval_intent_safety_v6.jsonl",
            base / "processed" / "train_chat.jsonl")
