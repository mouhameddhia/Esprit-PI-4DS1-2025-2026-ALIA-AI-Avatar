#!/usr/bin/env python3
"""Build a version snapshot artifact for prompts, models, and taxonomy assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_assets(repo_root: Path, assets: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, rel_path in assets.items():
        path = (repo_root / rel_path).resolve()
        exists = path.exists()
        item: Dict[str, Any] = {
            "path": rel_path,
            "exists": exists,
        }
        if exists and path.is_file():
            stat = path.stat()
            item.update(
                {
                    "sha256": _hash_file(path),
                    "size_bytes": stat.st_size,
                    "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        out[key] = item
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NLP version snapshot artifact")
    parser.add_argument(
        "--manifest-json",
        default="NLP/version_manifest.json",
        help="Path to version manifest JSON",
    )
    parser.add_argument(
        "--output-json",
        default="NLP/evaluation/results/ci_version_snapshot.json",
        help="Path for output snapshot artifact",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest_json).resolve()
    output_path = (repo_root / args.output_json).resolve()

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    manifest = _load_manifest(manifest_path)
    versions = manifest.get("versions") if isinstance(manifest.get("versions"), dict) else {}
    assets = manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {}

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.relative_to(repo_root).as_posix()),
        "manifest_version": manifest.get("manifest_version", "unknown"),
        "release_channel": manifest.get("release_channel", "unknown"),
        "versions": versions,
        "assets": _resolve_assets(repo_root=repo_root, assets=assets),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_assets = [k for k, v in artifact["assets"].items() if not v.get("exists")]
    print(f"Artifact saved to: {output_path}")
    print(f"Assets tracked: {len(artifact['assets'])}")
    print(f"Missing assets: {len(missing_assets)}")
    if missing_assets:
        for key in missing_assets:
            print(f"- {key}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
