"""Generate per-account G3 scoring kernels from kaggle/xenc_g3.py (line-level CONFIG replacement + compile check)."""
import json
import pathlib

src = pathlib.Path("kaggle/xenc_g3.py").read_text(encoding="utf-8")
lines = src.split("\n")
i = next(n for n, l in enumerate(lines) if l.startswith("CONFIG = {") and "@CONFIG@" in l)
jobs = {"xenc_g3_xlmr_h0.py": {"model": "FacebookAI/xlm-roberta-base", "halves": [0]},
        "xenc_g3_xlmr_h1.py": {"model": "FacebookAI/xlm-roberta-base", "halves": [1]},
        "xenc_g3_minilm.py": {"model": "microsoft/Multilingual-MiniLM-L12-H384", "halves": [0, 1]}}
for f, c in jobs.items():
    out = "\n".join(lines[:i] + [f"CONFIG = {json.dumps(c)}  # @CONFIG@ (generated from xenc_g3.py)"] + lines[i + 1:])
    compile(out, f, "exec")
    assert len(out.splitlines()) == len(src.splitlines())
    pathlib.Path("kaggle", f).write_text(out, encoding="utf-8")
    print(f, len(out.splitlines()), "lines OK")
