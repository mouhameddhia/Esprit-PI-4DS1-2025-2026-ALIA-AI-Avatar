"""Backend package bootstrap."""

from pathlib import Path
import sys


# Ensure top-level packages (for example NLP/) are importable when running from backend/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))
