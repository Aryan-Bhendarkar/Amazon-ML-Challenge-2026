"""Blocking / candidate-generation quality. Run on EVERY blocking change.

Targets (docs/strategy.md): pair recall >= 0.99 on the eval subset with a mean of ~20-40
candidates per S1. Recall lost here can never be recovered by the matcher.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def blocking_report(cands: pd.DataFrame, truth: Mapping[str, set], n_pool: int | None = None,
                    country: Mapping[str, str] | None = None,
                    s1_col: str = "s1_id", id_col: str = "cand_id") -> dict:
    """cands: long table (s1_id, cand_id) restricted to eval S1s. truth: {s1: set} incl. singletons.
    n_pool: size of the S2+S3 pool searched (for reduction ratio)."""
    c = cands[cands[s1_col].isin(truth.keys())]
    cs = c.groupby(s1_col)[id_col].agg(set).to_dict()
    tot_true = hit = full_cov = nonsingle = 0
    per_c: dict[str, list] = {}
    for s1, t in truth.items():
        got = cs.get(s1, set())
        h = len(t & got)
        tot_true += len(t)
        hit += h
        if t:
            nonsingle += 1
            full_cov += h == len(t)
        if country is not None:
            acc = per_c.setdefault(country.get(s1, "?"), [0, 0])
            acc[0] += h
            acc[1] += len(t)
    sizes = np.array([len(cs.get(s1, ())) for s1 in truth])
    out = {
        "pair_recall": hit / tot_true if tot_true else 1.0,
        "entity_full_coverage": full_cov / nonsingle if nonsingle else 1.0,
        "cands_mean": float(sizes.mean()),
        "cands_p50": float(np.percentile(sizes, 50)),
        "cands_p95": float(np.percentile(sizes, 95)),
        "cands_max": int(sizes.max()) if len(sizes) else 0,
        "n_pairs": int(sizes.sum()),
        "empty_cand_rate": float((sizes == 0).mean()),
    }
    if n_pool:
        out["reduction_ratio"] = float(1.0 - sizes.sum() / (len(truth) * n_pool))
    if per_c:
        out["pair_recall_by_country"] = {k: v[0] / v[1] if v[1] else 1.0 for k, v in per_c.items()}
    return out


def label_candidates(cands: pd.DataFrame, pairs: pd.DataFrame,
                     s1_col: str = "s1_id", id_col: str = "cand_id") -> pd.Series:
    """Boolean label per candidate row from GT pairs (s1_id, match_id)."""
    key = pd.MultiIndex.from_frame(pairs[["s1_id", "match_id"]])
    return pd.MultiIndex.from_frame(cands[[s1_col, id_col]]).isin(key)
