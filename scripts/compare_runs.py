"""Paired bootstrap comparison of two runs on identical eval entities.
Needs artifacts/<run_id>/val_entity_scores.parquet (save metric.per_entity_scores(...) in every run).
Usage: python scripts/compare_runs.py <run_a (baseline)> <run_b (candidate)>
"""
import _bootstrap  # noqa: F401

import json
import sys

import pandas as pd

from ber import paths
from ber.metric import paired_bootstrap

a, b = sys.argv[1], sys.argv[2]
fa = pd.read_parquet(paths.ART_DIR / a / "val_entity_scores.parquet")
fb = pd.read_parquet(paths.ART_DIR / b / "val_entity_scores.parquet")
print(json.dumps(paired_bootstrap(fa, fb), indent=2))
