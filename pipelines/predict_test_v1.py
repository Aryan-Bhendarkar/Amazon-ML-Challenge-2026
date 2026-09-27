"""Test inference for a features_v1 run: ctx features on the TEST cache -> stage-1 probs -> decision.

Streams data/cands/<cache>/test.parquet row groups (each row group holds complete S1 candidate lists),
adds ber.ctx_features with statistics computed over the TEST split (unsupervised, same code as val),
scores with the run's model and applies the run's val-tuned threshold after assign_best_s1.

    python pipelines/predict_test_v1.py --run <features_v1 run_id> --cache v1_n1 --threads 4 [--limit-rows 8000000]
    python pipelines/predict_test_v1.py --run <run_id> --from-feats      # model + threshold only (minutes)
Outputs (artifacts/<run_id>/): test_pred.parquet (s1_id, cand_id, prob >= 0.01), test_matches.parquet.
Candidates for /submit = data/cands/<cache>/test.parquet (the exact set scored here).
Use --limit-rows on a slice first to estimate time and memory for the full run.

Feature matrix: a featurizing run also saves EVERY test.parquet column + EVERY ctx_features column (not only the
run's features), one row group per batch (complete S1s), to data/cands/<cache>/test_feats_g15.parquet
(written to .tmp, renamed when complete; --limit-rows writes test_feats_g15_slice.parquet). --from-feats then skips
the test context + featurization and only streams those columns into the model (any run whose features.json is a
subset of the saved columns). --no-save-feats disables saving.
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
import pyarrow as pa
import pyarrow.parquet as pq

from ber import bag
from ber import ctx_features as cf
from ber import paths
from ber.decision import assign_best_s1, threshold_matches

PMIN = 0.01
FEATS_NAME = "test_feats_g15"


def feats_path(cache: str, sliced: bool, ctx_ver: int = 1):
    ver = "" if ctx_ver == 1 else f"_ctx{ctx_ver}"          # v1 name kept for the existing matrix
    return paths.DATA_DIR / "cands" / cache / f"{FEATS_NAME}{ver}{'_slice' if sliced else ''}.parquet"


def run_ctx_ver(run_id: str) -> int:
    """ctx_features version the run was trained with (meta params.ctx_ver; runs before v2 have none -> 1)."""
    meta = paths.EXP_DIR / run_id / "meta.json"
    return int(json.loads(meta.read_text()).get("params", {}).get("ctx_ver", 1)) if meta.exists() else 1


def main(a):
    t0 = time.time()
    art = paths.ART_DIR / a.run
    feats = json.loads((art / "features.json").read_text())
    t = json.loads((art / "decision.json").read_text())["threshold"] if a.threshold is None else a.threshold
    model = bag.load_model(art)                  # single model.lgb or seed bag (bag.json)
    cv = run_ctx_ver(a.run)                                   # never mix ctx versions between train and test
    nv = cf.norm_of(cv) if a.norm is None else a.norm      # --norm: same ctx features on a newer norm cache
    print(f"ctx_features version {cv} (norm_v{nv})")
    if a.from_feats:
        src = feats_path(a.cache, False, cv) if not a.feats_file else paths.ROOT / a.feats_file
    else:
        src = paths.DATA_DIR / "cands" / a.cache / f"{a.src_tag}.parquet"
    pf = pq.ParquetFile(src)
    all_cols = pf.schema_arrow.names
    have = set(all_cols)
    raw = cf.expand_derived(feats)                             # derived (coarsened) feats come from raw columns
    base = [f for f in raw if f in have]
    need_ctx = [f for f in raw if f not in have]
    if a.from_feats:
        assert not need_ctx, f"{src.name} lacks model features {need_ctx}"
    print(f"{pf.metadata.num_rows:,} test pairs in {pf.num_row_groups} row groups from {src.name}; {len(base)} "
          f"saved + {len(need_ctx)} ctx features to compute; threshold {t}")
    featurize = not a.from_feats
    save = featurize and not a.no_save_feats
    # featurizing runs compute ALL ctx groups anyway (add_features), so build the context whenever saving
    sctx = cf.SplitContext.build("test", cv, nv) if featurize and (need_ctx or save) else None
    print(f"  test context built {time.time() - t0:.0f}s")
    seen, parts, n_done, batch = set(), [], 0, []
    fw = {"writer": None, "schema": None}
    fpath = feats_path(a.cache, bool(a.limit_rows), cv)
    if nv != cf.norm_of(cv) or a.src_tag != "test":         # never overwrite the default matrix
        fpath = fpath.with_name(f"{fpath.stem}_{a.src_tag}_n{nv}.parquet")
    ftmp = fpath.with_suffix(".tmp")

    def flush(batch):
        nonlocal n_done
        cols = all_cols if save else ["s1_id", "cand_id"] + base
        X = pl.from_arrow(pq.ParquetFile(src).read_row_groups(batch, columns=list(dict.fromkeys(cols))))
        ids = set(X["s1_id"].unique().to_list())
        assert not (ids & seen), "an S1 spans row-group batches -> G5/rank features would be wrong"
        seen.update(ids)
        if sctx is not None:
            F = cf.add_features(cf.attach_norm(X.select("s1_id", "cand_id", "name_tset"), "test", nv), sctx,
                                workers=a.threads)
            new = cf.new_feature_cols(F, ["name_tset"]) if save else need_ctx
            X = X.join(F.select(["s1_id", "cand_id"] + [c for c in new if c not in X.columns]),
                       on=["s1_id", "cand_id"], how="left")
            del F
        if save:
            tb = X.to_arrow()
            if fw["writer"] is None:
                fw["schema"] = tb.schema
                fw["writer"] = pq.ParquetWriter(str(ftmp), tb.schema, compression="zstd")
            fw["writer"].write_table(tb.select(fw["schema"].names).cast(fw["schema"]), row_group_size=tb.num_rows)
            del tb
        Xp = cf.add_derived(cf.join_emb(X, a.cache, "test", feats), feats).select(["s1_id", "cand_id"] + feats).to_pandas()
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
    if fw["writer"] is not None:
        fw["writer"].close()
        os.replace(ftmp, fpath)
        print(f"  saved feature matrix {fpath} ({fpath.stat().st_size / 1e9:.2f} GB, "
              f"{len(fw['schema'].names)} columns)")
    pred = pd.concat(parts, ignore_index=True)
    suffix = ("_slice" if a.limit_rows else "") + a.out_suffix
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
    ap.add_argument("--from-feats", action="store_true", help=f"score data/cands/<cache>/{FEATS_NAME}.parquet only")
    ap.add_argument("--feats-file", default="", help="--from-feats: explicit matrix path (e.g. the _slice file)")
    ap.add_argument("--no-save-feats", action="store_true")
    ap.add_argument("--src-tag", default="test", help="candidate file data/cands/<cache>/<tag>.parquet (e.g. test_n2 "
                    "from pipelines/refeat_norm.py)")
    ap.add_argument("--norm", type=int, default=None, help="norm cache version for ctx features (default: the run's)")
    ap.add_argument("--out-suffix", default="", help="extra suffix for test_pred/test_matches (tests)")
    main(ap.parse_args())
