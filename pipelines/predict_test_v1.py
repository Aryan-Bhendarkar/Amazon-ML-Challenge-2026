"""Test inference for a features_v1 run: ctx features on the TEST cache -> stage-1 probs -> decision.

Streams data/cands/<cache>/test.parquet row groups (each row group holds complete S1 candidate lists),
adds ber.ctx_features with statistics computed over the TEST split (unsupervised, same code as val),
scores with the run's model and applies the run's val-tuned threshold after assign_best_s1.

    python pipelines/predict_test_v1.py --run <features_v1 run_id> --cache v1_n1 --threads 4 [--limit-rows 8000000]
Outputs (artifacts/<run_id>/): test_pred.parquet (s1_id, cand_id, prob >= 0.01), test_matches.parquet.
Candidates for /submit = data/cands/<cache>/test.parquet (the exact set scored here).
Use --limit-rows on a slice first to estimate time and memory for the full run.
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from ber import ctx_features as cf
from ber import paths
from ber.decision import assign_best_s1, threshold_matches

PMIN = 0.01


def main(a):
    t0 = time.time()
    art = paths.ART_DIR / a.run
    feats = json.loads((art / "features.json").read_text())
    t = json.loads((art / "decision.json").read_text())["threshold"] if a.threshold is None else a.threshold
    model = lgb.Booster(model_file=str(art / "model.lgb"))
    src = paths.DATA_DIR / "cands" / a.cache / "test.parquet"
    pf = pq.ParquetFile(src)
    have = set(pf.schema_arrow.names)
    base = [f for f in feats if f in have]
    need_ctx = [f for f in feats if f not in have]
    print(f"{pf.metadata.num_rows:,} test pairs in {pf.num_row_groups} row groups; {len(base)} base + "
          f"{len(need_ctx)} ctx features; threshold {t}")
    sctx = cf.SplitContext.build("test") if need_ctx else None
    print(f"  test context built {time.time() - t0:.0f}s")
    seen, parts, n_done, batch = set(), [], 0, []

    def flush(batch):
        nonlocal n_done
        tb = pq.ParquetFile(src).read_row_groups(batch, columns=["s1_id", "cand_id"] + base)
        X = pl.from_arrow(tb)
        ids = set(X["s1_id"].unique().to_list())
        assert not (ids & seen), "an S1 spans row-group batches -> G5/rank features would be wrong"
        seen.update(ids)
        if need_ctx:
            F = cf.add_features(cf.attach_norm(X.select("s1_id", "cand_id", "name_tset"), "test"), sctx,
                                workers=a.threads)
            X = X.join(F.select(["s1_id", "cand_id"] + need_ctx), on=["s1_id", "cand_id"], how="left")
            del F
        Xp = X.select(["s1_id", "cand_id"] + feats).to_pandas()
        p = model.predict(Xp[feats], num_threads=a.threads).astype(np.float32)
        keep = p >= PMIN
        parts.append(Xp.loc[keep, ["s1_id", "cand_id"]].assign(prob=p[keep]))
        n_done += len(Xp)
        print(f"  {n_done:,} pairs scored, {len(seen):,} S1, {time.time() - t0:.0f}s", flush=True)
        del X, Xp
        gc.collect()

    rows = 0
    for rg in range(pf.num_row_groups):
        batch.append(rg)
        rows += pf.metadata.row_group(rg).num_rows
        if rows >= a.batch_rows:
            flush(batch)
            batch, rows = [], 0
            if a.limit_rows and n_done >= a.limit_rows:
                break
    if batch and not (a.limit_rows and n_done >= a.limit_rows):
        flush(batch)
    pred = pd.concat(parts, ignore_index=True)
    suffix = "_slice" if a.limit_rows else ""
    pred.to_parquet(art / f"test_pred{suffix}.parquet")
    m = threshold_matches(assign_best_s1(pred), t)
    pd.DataFrame([(s, x) for s, xs in m.items() for x in xs], columns=["s1_id", "match_id"]) \
      .to_parquet(art / f"test_matches{suffix}.parquet")
    el = (time.time() - t0) / 60
    print(f"done {el:.1f} min: {n_done:,} pairs, {len(seen):,} S1, {sum(map(len, m.values())):,} matches for "
          f"{len(m):,} S1 (t={t}); est. full run {el * pf.metadata.num_rows / max(n_done, 1):.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default="v1_n1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=3_000_000)
    ap.add_argument("--limit-rows", type=int, default=0, help="score only a slice (timing/memory estimate)")
    ap.add_argument("--threshold", type=float, default=None)
    main(ap.parse_args())
