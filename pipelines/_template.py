"""TEMPLATE pipeline: copy to pipelines/<slug>.py and replace the three marked stages.

It runs end-to-end as-is (deliberately naive: one blocking key + one similarity, no training),
so it doubles as a plumbing test:
    python pipelines/_template.py --subset micro                  # val run, logs a tracked Run
    python pipelines/_template.py --test --threshold 0.8 --run-id <val run_id>
                                                                  # test inference -> artifacts for /submit
Stages to replace:  build_candidates()  ->  score_pairs()  (features + model)  ->  decision (harness)
"""
import _bootstrap  # noqa: F401  (adds src/ to sys.path, UTF-8 console)

import argparse
import time

import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz import fuzz, process

from ber import harness, paths
from ber.decision import assign_best_s1, threshold_matches
from ber.tracking import Run

NORM_V = 0
COLS = ["entity_id", "country", "n_core", "n_compact", "a_house", "a_street"]


def norm_scan(split: str, source: int) -> pl.LazyFrame:
    p = paths.FEATURE_DIR / f"norm_v{NORM_V}_{split}_s{source}.parquet"
    return pl.scan_parquet(p).select(COLS)


# ----------------------------------------------------------------------------- STAGE 1: blocking
def build_candidates(split: str, s1_ids: set | None) -> tuple[pd.DataFrame, int]:
    """Return (candidates[s1_id, cand_id], pool_size). REPLACE with real multi-retriever blocking."""
    s1 = norm_scan(split, 1)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(list(s1_ids)))
    pool = pl.concat([norm_scan(split, 2), norm_scan(split, 3)])
    key = pl.col("country") + "|" + pl.col("n_compact").str.slice(0, 8)
    s1k = s1.select(pl.col("entity_id").alias("s1_id"), key.alias("k")).filter(pl.col("k").str.len_chars() > 4)
    pk = pool.select(pl.col("entity_id").alias("cand_id"), key.alias("k"))
    big = pk.group_by("k").len().filter(pl.col("len") > 300).select("k")      # cap huge blocks
    cands = s1k.join(pk.join(big, on="k", how="anti"), on="k").select("s1_id", "cand_id")
    n_pool = pool.select(pl.len()).collect().item()
    return cands.collect().to_pandas(), n_pool


# ----------------------------------------------------------------------------- STAGE 2: scoring
def score_pairs(split: str, cands: pd.DataFrame) -> pd.DataFrame:
    """Return cands + prob. REPLACE with features + trained model (LightGBM etc.)."""
    ids = pl.DataFrame({"entity_id": pd.unique(pd.concat([cands.s1_id, cands.cand_id])).tolist()})
    rec = pl.concat([norm_scan(split, s) for s in (1, 2, 3)]).join(ids.lazy(), on="entity_id").collect().to_pandas()
    rec = rec.set_index("entity_id")
    a = rec.loc[cands.s1_id]
    b = rec.loc[cands.cand_id]
    name = process.cpdist(a.n_core.to_numpy(), b.n_core.to_numpy(), scorer=fuzz.token_set_ratio, workers=-1) / 100
    house = (a.a_house.to_numpy() == b.a_house.to_numpy()) & (a.a_house.to_numpy() != "")
    out = cands.copy()
    out["prob"] = (0.7 * name + 0.3 * house).astype("float32")      # NOT a model — placeholder
    return out


# ----------------------------------------------------------------------------- main
def main(a) -> None:
    t0 = time.time()
    if not a.test:
        ctx = harness.EvalContext.load(a.subset)
        with Run("template-naive", hypothesis="plumbing test: compact-name key + token_set",
                 params={"subset": a.subset, "norm_v": NORM_V}, tags=["baseline"]) as run:
            cands, n_pool = build_candidates("train", ctx.ids)
            harness.log_blocking(run, cands, ctx, n_pool)
            pred = score_pairs("train", cands)
            harness.log_predictions(run, pred, ctx)
            run.log(timing_min=round((time.time() - t0) / 60, 2))
            run.note("Template run. Replace blocking + scoring before drawing conclusions.")
    else:
        assert a.run_id and a.threshold is not None, "--test needs --run-id (val run) and --threshold (tuned on val)"
        out = paths.ART_DIR / a.run_id
        out.mkdir(parents=True, exist_ok=True)
        cands, _ = build_candidates("test", None)
        cands.to_parquet(out / "test_candidates.parquet")               # exact set scored
        pred = score_pairs("test", cands)
        m = threshold_matches(assign_best_s1(pred), a.threshold)
        pd.DataFrame([(s, x) for s, xs in m.items() for x in xs], columns=["s1_id", "match_id"]) \
          .to_parquet(out / "test_matches.parquet")
        print(f"test artifacts in {out} ({(time.time() - t0) / 60:.1f} min) -> now run /submit {a.run_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="micro", choices=["micro", "mini", "fold0"])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    main(ap.parse_args())
