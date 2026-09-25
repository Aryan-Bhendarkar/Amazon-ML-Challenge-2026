"""S1-context post-processing rules on ASSIGNED pair probabilities (no retraining).

Measured in experiments/runs/20260925-1236_aryan_baseline-v0-keys-lgbm/error_analysis.md §5.
Keys are counted over ALL S1 of the split, per country (country is only a partition key taken from
the data; unlabeled + transductive, so legal on test):
  n_key = sorted n_core tokens, a_key = sorted a_full tokens
  s1_same_name = #S1 of the country with the candidate's n_key, s1_same_addr = same for a_key
Rules (applied after assign_best_s1, base = prob >= t):
  ruleA (add):    cand address empty & n_key(S1) == n_key(cand) != '' & s1_same_name == 1
  dropA (remove): base & cand address empty & s1_same_name >= 2
  ruleB (add):    a_key(S1) == a_key(cand) != '' & s1_same_addr == 1 & name token_set < 50
  keep = (base | ruleA | ruleB) & ~dropA

    flags = context_flags(assigned, "test")       # assigned: s1_id, cand_id, prob (pandas)
    keep = keep_mask(flags, t)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz import fuzz, process

from . import paths

NAME_TSET_MAX_B = 50


def _norm_file(split: str, source: int, norm_v: int = 0):
    return paths.FEATURE_DIR / f"norm_v{norm_v}_{split}_s{source}.parquet"


def _sorted_key(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.sort().list.join(" "))


def with_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_sorted_key("n_core").alias("n_key"), _sorted_key("a_full").alias("a_key"))


def s1_key_counts(s1: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """s1: ALL S1 of the split with country, n_key, a_key -> (name counts, address counts)."""
    nk = s1.filter(pl.col("n_key") != "").group_by("country", "n_key").len("s1_same_name")
    ak = s1.filter(pl.col("a_key") != "").group_by("country", "a_key").len("s1_same_addr")
    return nk, ak


def rule_flags(A: pl.DataFrame, nk: pl.DataFrame, ak: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    """A: assigned pairs with S1 cols (n_key, a_key, n_core) and candidate cols (country_c, n_key_c,
    a_key_c, n_core_c, a_empty_c). Returns A + s1_same_name, s1_same_addr, ruleA, dropA_cond, ruleB."""
    A = (A.join(nk.rename({"country": "country_c", "n_key": "n_key_c"}), on=["country_c", "n_key_c"], how="left")
          .join(ak.rename({"country": "country_c", "a_key": "a_key_c"}), on=["country_c", "a_key_c"], how="left")
          .with_columns(pl.col("s1_same_name").fill_null(0), pl.col("s1_same_addr").fill_null(0)))
    same_a = (pl.col("a_key") == pl.col("a_key_c")) & (pl.col("a_key") != "")
    A = A.with_columns(same_a.alias("_same_a"))
    tset = np.full(A.height, 100.0, dtype=np.float32)
    idx = np.flatnonzero(A["_same_a"].to_numpy())
    if len(idx):
        sub = A[idx]
        tset[idx] = process.cpdist(sub["n_core"].fill_null("").to_list(), sub["n_core_c"].fill_null("").to_list(),
                                   scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32)
    A = A.with_columns(pl.Series("name_tset", tset))
    return A.with_columns(
        (pl.col("a_empty_c") & (pl.col("n_key") == pl.col("n_key_c")) & (pl.col("n_key") != "")
         & (pl.col("s1_same_name") == 1)).alias("ruleA"),
        (pl.col("a_empty_c") & (pl.col("s1_same_name") >= 2)).alias("dropA_cond"),
        (pl.col("_same_a") & (pl.col("s1_same_addr") == 1) & (pl.col("name_tset") < NAME_TSET_MAX_B)).alias("ruleB"),
    ).drop("_same_a")


def context_flags(assigned: pd.DataFrame, split: str, norm_v: int = 0) -> pd.DataFrame:
    """Load the split's normalized S1 (ALL of them, for the counts) and the candidates, return
    assigned + rule flags (pandas, same row order as `assigned`)."""
    s1 = with_keys(pl.read_parquet(_norm_file(split, 1, norm_v), columns=["entity_id", "country", "n_core", "a_full"]))
    nk, ak = s1_key_counts(s1)
    a = pl.from_pandas(assigned[["s1_id", "cand_id", "prob"]].reset_index(drop=True)).with_row_index("_r")
    ids = a.select(pl.col("cand_id").unique())
    C = pl.concat([pl.scan_parquet(_norm_file(split, s, norm_v))
                   .select("entity_id", "country", "n_core", "a_full", "a_empty")
                   .join(ids.lazy(), left_on="entity_id", right_on="cand_id").collect() for s in (2, 3)])
    C = with_keys(C).rename({"entity_id": "cand_id", "country": "country_c", "n_core": "n_core_c",
                             "a_full": "a_full_c", "a_empty": "a_empty_c", "n_key": "n_key_c", "a_key": "a_key_c"})
    A = (a.join(s1.select(pl.col("entity_id").alias("s1_id"), "n_core", "n_key", "a_key"), on="s1_id", how="left")
          .join(C.drop("a_full_c"), on="cand_id", how="left")
          .with_columns(pl.col("a_empty_c").fill_null(False)))
    out = rule_flags(A, nk, ak).sort("_r")
    return out.drop("_r").to_pandas()


def keep_mask(flags: pd.DataFrame, t: float) -> np.ndarray:
    base = flags.prob.to_numpy() >= t
    return ((base | flags.ruleA.to_numpy() | flags.ruleB.to_numpy()) & ~(base & flags.dropA_cond.to_numpy()))


def to_matches(flags: pd.DataFrame, t: float) -> dict[str, list[str]]:
    k = flags[keep_mask(flags, t)]
    return k.groupby("s1_id").cand_id.agg(list).to_dict()
