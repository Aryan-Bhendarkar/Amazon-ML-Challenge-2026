"""Import this first in every script: makes `import ber` work without installing the package."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
try:  # Windows consoles default to cp1252 -> printing Devanagari/Tamil crashes without this
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
