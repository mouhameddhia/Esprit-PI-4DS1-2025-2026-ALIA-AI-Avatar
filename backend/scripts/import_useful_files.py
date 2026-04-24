"""Compatibility entrypoint for useful-files ingestion.

The source of truth now lives in NLP/pipeline/ingestion.py.
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NLP.pipeline.ingestion import main


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
