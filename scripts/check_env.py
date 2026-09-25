"""Environment sanity check. Usage: python scripts/check_env.py"""
import _bootstrap  # noqa: F401

import importlib
import platform
import shutil
import sys

from ber import paths

# torch FIRST: on Windows, loading torch after some native libs can fail with a DLL OSError
PKGS = ["torch", "sentence_transformers", "numpy", "pandas", "pyarrow", "sklearn", "scipy", "rapidfuzz",
        "anyascii", "lightgbm", "polars", "faiss", "psutil", "pytest"]
print(f"python {sys.version.split()[0]} on {platform.platform()}")
for p in PKGS:
    try:
        m = importlib.import_module(p)
        print(f"  ok   {p:22s} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  MISS {p:22s} ({type(e).__name__}: {str(e)[:150]})")
try:
    import torch
    print("  cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:
    pass
print("raw data:", paths.RAW_DIR, "exists" if paths.raw_tsv("train", 1).exists() else "MISSING")
print("parquet cache:", "ready" if paths.parquet_path("test", 3).exists() else "missing -> python scripts/prepare_data.py")
print("folds:", "ready" if paths.folds_path().exists() else "missing -> python scripts/prepare_data.py")
print("git:", shutil.which("git") or "MISSING")
