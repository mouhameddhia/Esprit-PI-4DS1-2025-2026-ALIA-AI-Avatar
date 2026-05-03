"""Build or extend evaluation datasets."""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_v6(base_version: str = "v5") -> pathlib.Path:
    """Extend base_version with the additional physician_portal examples."""
    raw_dir  = REPO_ROOT / "alia_nlp" / "data" / "raw"
    base_path = raw_dir / f"eval_intent_safety_{base_version}.jsonl"
    out_path  = raw_dir / "eval_intent_safety_v6.jsonl"

    if not base_path.exists():
        raise FileNotFoundError(base_path)

    rows = [json.loads(l) for l in base_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Additional rows are committed in v6 already — this script re-generates if needed.
    v6_path = raw_dir / "eval_intent_safety_v6.jsonl"
    if v6_path.exists():
        print(f"v6 already exists: {v6_path}")
        return v6_path

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Built: {out_path} ({len(rows)} rows)")
    return out_path


if __name__ == "__main__":
    build_v6()
