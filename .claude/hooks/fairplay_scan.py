"""PostToolUse scan of edited Python/notebook files for rule violations and known footguns.
Exit code 2 -> the message is fed back to Claude so it fixes the file."""
import re
import sys
from pathlib import Path

from _common import FORBIDDEN_HOSTS, FORBIDDEN_PKGS, ROOT, read_input

ev = read_input()
ti = ev.get("tool_input", {}) or {}
fp = ti.get("file_path") or ti.get("notebook_path") or ""
if not fp or not fp.endswith((".py", ".ipynb")):
    sys.exit(0)
p = Path(fp)
if not p.exists() or ".claude" in p.parts:
    sys.exit(0)
try:
    text = p.read_text(encoding="utf-8", errors="ignore")
except Exception:
    sys.exit(0)
low = text.lower()
issues = []

for h in FORBIDDEN_HOSTS:
    if h in low:
        issues.append(f"references '{h}' — external lookup services are PROHIBITED (disqualification).")
for pkg in FORBIDDEN_PKGS:
    if re.search(rf"^\s*(import|from)\s+{pkg}\b", text, re.M):
        issues.append(f"imports '{pkg}' — geocoding/postal libraries rely on external data; not allowed.")
for line_no, line in enumerate(text.splitlines(), 1):
    l = line.replace(" ", "")
    if ".to_csv(" in l and ("sep='\\t'" in l or 'sep="\\t"' in l) and "lineterminator" not in l:
        issues.append(f"line {line_no}: TSV written with to_csv without lineterminator='\\n' -> CRLF on "
                      "Windows corrupts IDs. Use ber.submission.write_id_lists for outputs, or add lineterminator='\\n'.")
    if re.search(r"get_dummies\(.*country", line):
        issues.append(f"line {line_no}: one-hot of country — forbidden (test has unseen France).")
    if re.search(r"""country['"]?\]?\s*(==|!=)\s*['"](us|india)['"]""", line, re.I) and "assert" not in line:
        issues.append(f"line {line_no}: hard-coded country comparison — make sure it has a generic "
                      "fallback and France is never dropped.")
    if re.search(r"""read_csv\(.*\.tsv""", line) and "keep_default_na" not in line and "ber" not in line:
        issues.append(f"line {line_no}: raw read_csv on a TSV — use ber.io loaders (handles 'NA' names, quoting, dtype=str).")
for bad in ("meta-llama", "llama-", "/gemma", "gemma-", "qwen2.5-3b", "qwen2.5-72b"):
    if bad in low:
        issues.append(f"mentions model '{bad}' — license is not MIT/Apache-2.0; not allowed as a model.")

if issues:
    rel = p.resolve().relative_to(ROOT) if ROOT in p.resolve().parents else p
    sys.stderr.write(f"[fair-play/footgun scan] {rel}:\n- " + "\n- ".join(dict.fromkeys(issues)) + "\n")
    sys.exit(2)
