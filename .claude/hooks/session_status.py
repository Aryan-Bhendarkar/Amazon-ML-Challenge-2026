"""SessionStart hook: inject competition status + key reminders into Claude's context."""
import io
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

from _common import ROOT

buf = io.StringIO()
try:
    with redirect_stdout(buf):
        sys.argv = ["status.py"]
        runpy.run_path(str(ROOT / "scripts" / "status.py"), run_name="__main__")
except Exception as e:  # never block a session
    buf.write(f"(status unavailable: {e})\n")
print(buf.getvalue().rstrip())
missing = []
if not (ROOT / "data" / "parquet" / "test_s3.parquet").exists():
    missing.append("parquet cache (python scripts/prepare_data.py)")
if not list((ROOT / "data" / "features").glob("norm_v*_test_s3.parquet")):
    missing.append("normalized cache (python scripts/build_norm_cache.py)")
if missing:
    print("SETUP MISSING: " + "; ".join(missing))
venv = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
print(f"Python to use: {venv.relative_to(ROOT).as_posix() if venv.exists() else 'python (no .venv found)'}")
print("Reminders: MIT/Apache<=8B models only | no external data | country is open-set (France in test) | "
      "outputs via ber.submission (LF) | every run via ber.tracking.Run | 15 LB submissions total.")
