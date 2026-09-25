"""Deterministic validation folds over TRAIN Source-1 entities.

fold = md5(s1_id) mod K  -> stable across machines/teammates, no randomness to sync.
Within fold 0 we also flag nested dev subsets for fast iteration:
  mini  ~20% of fold 0 (~88k S1)   -> default for quick experiments
  micro ~2%  of fold 0 (~9k S1)    -> smoke tests only, too noisy for decisions

Protocol (see docs/strategy.md):
  * evaluate on S1 queries of the eval subset, but ALWAYS retrieve candidates from the FULL
    train S2/S3 pool of that country, so the density of distractors/hard negatives matches test.
  * train models only on S1 entities from folds != eval fold (their positives + mined negatives).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

K_FOLDS = 5


def _h(s: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.md5((salt + s).encode("utf-8")).digest()[:8], "little")


def make_folds(gt: pd.DataFrame, s1: pd.DataFrame, k: int = K_FOLDS) -> pd.DataFrame:
    """gt: raw GT (source1_entity_id, matched_entity_ids); s1: train S1 table (entity_id, country)."""
    df = gt.rename(columns={"source1_entity_id": "s1_id"})
    m = df["matched_entity_ids"]
    df["n_matches"] = np.where(m.str.len() > 0, m.str.count(",") + 1, 0).astype("int16")
    df = df.drop(columns="matched_entity_ids")
    df = df.merge(s1[["entity_id", "country"]].rename(columns={"entity_id": "s1_id"}), on="s1_id", how="left")
    ids = df["s1_id"].tolist()
    df["fold"] = np.array([_h(x) % k for x in ids], dtype="int8")
    sub = np.array([_h(x, "sub") % 100 for x in ids], dtype="int16")
    df["mini"] = (df["fold"] == 0) & (sub < 20)
    df["micro"] = (df["fold"] == 0) & (sub < 2)
    return df


def eval_ids(folds: pd.DataFrame, subset: str = "mini") -> set[str]:
    """subset in {'micro','mini','fold0','fold0x'}; fold0x = fold0 minus mini (disjoint confirmation set)."""
    if subset == "fold0":
        return set(folds.loc[folds.fold == 0, "s1_id"])
    if subset == "fold0x":
        return set(folds.loc[(folds.fold == 0) & ~folds["mini"], "s1_id"])
    return set(folds.loc[folds[subset], "s1_id"])


def train_ids(folds: pd.DataFrame, eval_fold: int = 0) -> set[str]:
    return set(folds.loc[folds.fold != eval_fold, "s1_id"])
