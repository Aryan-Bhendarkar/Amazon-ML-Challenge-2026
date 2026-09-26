"""Orphan simulation (HANDOFF EXP-A/B): make train look like test's suspected S1 pruning.

Hypothesis: test was built at train scale, then ~19% of S1 were removed while their S2/S3 records were kept
(records per S1: train 4.67, test 5.76 = 4.67 / 0.81). The orphaned records then look like matches for the
remaining similar S1 (chains, co-located, templated names). To simulate on train:
  * remove a deterministic subset of ALL train S1 (eval queries and the competing pool alike),
  * keep every S2/S3 record (records of removed S1 become unmatched: label 0 against every survivor),
  * recompute the S1-side split statistics (ctx G1/G2/G3) on the surviving S1 only,
  * score and assign only surviving eval S1.

Removal masks are pure functions of the S1 id (md5), so every run/model sees the same survivors.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import polars as pl

from .ctx_features import key_exprs
from .metric import f05_entity

SALT = "|orphan"


def md5_unit(ids, salt: str = SALT) -> np.ndarray:
    """Uniform [0,1) value per id: (md5(id + salt) mod 1000) / 1000, reproducible across machines."""
    return np.fromiter((int(hashlib.md5((s + salt).encode()).hexdigest(), 16) % 1000 for s in ids),
                       dtype=np.float64, count=len(ids)) / 1000.0


def shared_flag(s1: pl.DataFrame) -> np.ndarray:
    """True for S1 whose sorted core-name key or full-address key is shared with another S1 of the same country.
    s1: entity_id, country, n_core, a_full, a_street, a_house (normalized)."""
    k = s1.select("country", "n_core", "a_full", "a_street", "a_house").with_columns(key_exprs()[:2])
    k = k.with_columns(pl.len().over("country", "n_key").alias("_nn"), pl.len().over("country", "a_key").alias("_na"))
    return k.select(((pl.col("_nn") > 1) & (pl.col("n_key") != "")) | ((pl.col("_na") > 1) & (pl.col("a_key") != ""))
                    ).to_series().to_numpy()


def twin_flag(s1: pl.DataFrame) -> np.ndarray:
    """True for S1 whose sorted core-name key is shared with another S1 of the same country at a DIFFERENT house
    number (leading zeros stripped): the 'name twin' whose removal leaves same-name, shifted-house records behind."""
    k = s1.select("country", "n_core", "a_full", "a_street", "a_house").with_columns(key_exprs()[:1]).with_columns(
        pl.col("a_house").fill_null("").str.strip_chars_start("0").alias("_h"))
    k = k.with_columns(pl.col("_h").n_unique().over("country", "n_key").alias("_nh"))
    return k.select((pl.col("n_key") != "") & (pl.col("_nh") > 1)).to_series().to_numpy()


def removal_mask(s1: pl.DataFrame, rate: float = 0.19, variant: str = "uniform", salt: str = SALT) -> pl.DataFrame:
    """entity_id, removed (bool) for every S1 row.
    uniform: remove iff md5 unit < rate.
    twin   : only S1 that have a same-name S1 at a different house (same country), each removed with prob `rate`.
    biased : per-S1 rate x2 if its name or address key is shared with another S1 (same country), x0.5 otherwise,
             rescaled so the overall expected rate is `rate` (chains / co-located S1 are removed more often)."""
    ids = s1["entity_id"].to_list()
    u = md5_unit(ids, salt)
    if rate <= 0:
        rem = np.zeros(len(ids), bool)
    elif variant == "uniform":
        rem = u < rate
    elif variant == "biased":
        w = np.where(shared_flag(s1), 2.0, 0.5)
        p = np.minimum(1.0, rate * w / w.mean())
        rem = u < p
    elif variant == "twin":                  # remove only name twins (different house), at `rate` among them
        rem = twin_flag(s1) & (u < rate)
    else:
        raise ValueError(variant)
    return pl.DataFrame({"entity_id": ids, "removed": rem})


def loss_decomposition(pred: dict, truth: dict) -> dict:
    """Split the macro-F0.5 loss (1 - F) into: FP on true singletons, FP on non-singletons (score gained if the
    wrong records were dropped), FN (the rest: missing true records, incl. empty predictions). Sums to 1 - F."""
    n = len(truth)
    fs = fn_ = fp_ns = 0.0
    for s1, t in truth.items():
        p = set(pred.get(s1, ()))
        f = f05_entity(p, t)
        if not t:
            fs += 1.0 - f
            continue
        f_clean = f05_entity(p & t, t)
        fp_ns += f_clean - f
        fn_ += 1.0 - f_clean
    return {"loss_total": (fs + fp_ns + fn_) / n, "loss_fp_singleton": fs / n, "loss_fp_nonsingleton": fp_ns / n,
            "loss_fn": fn_ / n}


def fp_sources(pred: dict, truth: dict, owner: pd.Series, removed: set) -> dict:
    """Classify predicted false-positive records by their GT owner: orphan (owner removed), owned by another
    surviving S1, or unmatched in GT."""
    fps = [c for s1, cs in pred.items() for c in cs if c not in truth.get(s1, ())]
    own = owner.reindex(fps)
    orphan = own.isin(removed)
    unm = own.isna()
    return {"fp_total": int(len(fps)), "fp_orphan": int(orphan.sum()), "fp_owned_other": int((~orphan & ~unm).sum()),
            "fp_unmatched": int(unm.sum())}
