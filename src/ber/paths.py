"""Central path configuration. Every script imports paths from here — never hard-code.

Override locations with environment variables (useful on SageMaker / teammates' machines):
  AMLC_ROOT      project root (default: two levels above this file)
  AMLC_RAW_DIR   folder holding train/ and test/ TSVs (default: <root>/student_resource/dataset)
  AMLC_DATA_DIR  derived data: parquet caches, folds, samples (default: <root>/data)
  AMLC_ART_DIR   large run artifacts: candidates, predictions, models (default: <root>/artifacts)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AMLC_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = Path(os.environ.get("AMLC_RAW_DIR", ROOT / "student_resource" / "dataset"))
DATA_DIR = Path(os.environ.get("AMLC_DATA_DIR", ROOT / "data"))
ART_DIR = Path(os.environ.get("AMLC_ART_DIR", ROOT / "artifacts"))

PARQUET_DIR = DATA_DIR / "parquet"      # raw TSVs converted 1:1 (fast loading)
SPLIT_DIR = DATA_DIR / "splits"         # fold assignment for train S1 entities
SAMPLE_DIR = DATA_DIR / "samples"       # small human-readable samples for eyeballing
FEATURE_DIR = DATA_DIR / "features"     # cached normalized columns / embeddings

EXP_DIR = ROOT / "experiments" / "runs"         # small, committed: meta/metrics/notes per run
SUB_DIR = ROOT / "submissions"                  # submission records (committed) + files (ignored)
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def raw_tsv(split: str, source: int) -> Path:
    return RAW_DIR / split / f"{split}_source{source}.tsv"


def raw_gt() -> Path:
    return RAW_DIR / "train" / "train_ground_truth.tsv"


def parquet_path(split: str, source: int) -> Path:
    return PARQUET_DIR / f"{split}_s{source}.parquet"


def gt_pairs_path() -> Path:
    return PARQUET_DIR / "train_gt_pairs.parquet"


def folds_path() -> Path:
    return SPLIT_DIR / "folds.parquet"


def ensure_dirs() -> None:
    for d in (PARQUET_DIR, SPLIT_DIR, SAMPLE_DIR, FEATURE_DIR, ART_DIR, EXP_DIR, SUB_DIR / "records", SUB_DIR / "files"):
        d.mkdir(parents=True, exist_ok=True)
