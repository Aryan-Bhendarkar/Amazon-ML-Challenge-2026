"""Status line: model | hours left | submissions | best val."""
import io
import json
import runpy
import sys
from contextlib import redirect_stdout

from _common import ROOT

try:
    model = (json.load(sys.stdin).get("model") or {}).get("display_name", "")
except Exception:
    model = ""
buf = io.StringIO()
try:
    with redirect_stdout(buf):
        sys.argv = ["status.py", "--short"]
        runpy.run_path(str(ROOT / "scripts" / "status.py"), run_name="__main__")
except Exception:
    buf.write("AMLC")
print((f"[{model}] " if model else "") + buf.getvalue().strip())
