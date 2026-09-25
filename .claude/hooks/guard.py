"""PreToolUse guard: protect raw competition files, key folders, and the fair-play rules."""
import re
from pathlib import Path

from _common import FORBIDDEN_HOSTS, ROOT, ask, deny, read_input

PROTECTED = (ROOT / "student_resource").resolve()
KEY_DIRS = r"(data|artifacts|experiments|submissions|student_resource|src|pipelines|\.git)"

ev = read_input()
tool = ev.get("tool_name", "")
ti = ev.get("tool_input", {}) or {}

if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
    fp = ti.get("file_path") or ti.get("notebook_path") or ""
    if fp:
        try:
            p = Path(fp).resolve()
            if p == PROTECTED or PROTECTED in p.parents:
                deny("student_resource/ holds the ORIGINAL competition files and is read-only. "
                     "Write derived data to data/ or artifacts/, docs to docs/.")
        except Exception:
            pass

elif tool in ("Bash", "PowerShell"):
    cmd = ti.get("command", "") or ""
    low = cmd.lower()
    for h in FORBIDDEN_HOSTS:
        if h in low:
            deny(f"Blocked: '{h}' looks like an external lookup service. Using external databases/APIs/"
                 "geocoding is PROHIBITED (disqualification). Use only the provided data.")
    if "student_resource" in low and re.search(r"(^|[;&|\s])(rm|mv|del|remove-item|move-item)\s|sed\s+-i|>\s*\S*student_resource|out-file|set-content", low):
        deny("Blocked: commands must not modify or delete student_resource/ (original competition files).")
    if re.search(r"\brm\s+-[a-z]*r[a-z]*\b.*\b" + KEY_DIRS + r"\b", low) or \
       re.search(r"remove-item\b.*-recurse.*\b" + KEY_DIRS + r"\b", low) or \
       re.search(r"remove-item\b.*\b" + KEY_DIRS + r"\b.*-recurse", low):
        ask("Recursive delete touching a key project folder (data/artifacts/experiments/submissions/src). "
            "Confirm this is intended — derived data takes a long time to rebuild.")
    if re.search(r"git\s+push\b.*(--force|-f\b)", low) or re.search(r"git\s+reset\s+--hard", low):
        ask("Force-push / hard reset can destroy teammates' work. Confirm.")
