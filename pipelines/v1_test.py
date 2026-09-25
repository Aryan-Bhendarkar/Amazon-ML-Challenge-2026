"""Test inference for blocking v1 + a model trained on the v1 cache (e.g. the Phase 1 gate run).

Same code path as the cache build (pipelines/build_cache.build_v1): blocking_v1.run_country -> cap per S1 ->
baseline_v0.featurize (+ rbits, cos_name, cos_na) -> model.predict. IDF / key counts are fit on the TEST split's own
S1 + pool (unlabeled, transductive). Countries are discovered from the data (France included).

Outputs (artifacts/<model_run>/, or --out-dir):
  test_candidates.parquet  s1_id, cand_id  (EXACT scored set; one row group per S1 chunk -> streamable)
  test_pred.parquet        s1_id, cand_id, prob (prob >= FLOOR)
  test_matches.parquet     s1_id, match_id (assign_best_s1 + the run's tuned threshold)
  test_timing.json         per country: queries, pairs, seconds, peak RSS
  data/cands/v1_n<norm_v>/test.parquet  (--save-feats) all scored pairs + features, so later models on the v1 cache
                           can predict test without re-blocking/featurizing

    python pipelines/v1_test.py --model-run <run_id> --frac 0.05 --out-dir <scratch>   # timing slice
    python pipelines/v1_test.py --model-run <run_id>                                    # full test
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import baseline_v0 as bv0
import blocking_v1 as bv1
from ber import blocking as B
from ber import paths
from ber.decision import assign_best_s1, threshold_matches

FLOOR = 0.05
SEED = 42


def main(a):
    t_all = time.time()
    art = paths.ART_DIR / a.model_run
    out = Path(a.out_dir) if a.out_dir else art
    out.mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_file=str(art / "model.lgb"))
    feats = json.loads((art / "features.json").read_text())
    t = json.loads((art / "decision.json").read_text())["threshold"]
    bv0.NORM_V = a.norm_v
    s1_all = pl.read_parquet(B.norm_file("test", 1, a.norm_v), columns=["entity_id", "country"])
    schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string())])
    cw = pq.ParquetWriter(str(out / "test_candidates.parquet"), schema, compression="zstd")
    fw, fschema = None, None
    preds, timing = [], {}
    for ctry in B.countries_of("test", a.norm_v):
        t0 = time.time()
        ids = s1_all.filter(pl.col("country") == ctry)["entity_id"]
        if a.frac < 1.0:
            ids = ids.sample(fraction=a.frac, seed=SEED)
        _, U, tim, q, ptbl = bv1.run_country("test", ctry, set(ids.to_list()), print, a.norm_v,
                                             keep_frames=True, retriever_frames=False)
        U = B.cap_per_query(U, "score", bv1.P["cap"])
        t_blk = time.time() - t0
        qx = q.with_row_index("idx")
        n_pairs = 0
        for o in range(0, q.height, a.chunk):
            pr = U.filter((pl.col("i") >= o) & (pl.col("i") < o + a.chunk))
            if pr.height == 0:
                continue
            pool_part = bv0.fetch_pool(ptbl, pr["j"].to_numpy())
            f = bv0.featurize(pr.select("i", "j", "kmask"), qx, pool_part)
            f = f.merge(pr.select("s1_id", "cand_id", "rbits", "cos_name", "cos_na").to_pandas(),
                        on=["s1_id", "cand_id"], how="left", validate="one_to_one")
            cw.write_table(pa.Table.from_pandas(f[["s1_id", "cand_id"]], schema=schema, preserve_index=False),
                           row_group_size=max(1, len(f)))
            if a.save_feats:
                if fw is None:
                    fschema = pa.Table.from_pandas(f, preserve_index=False).schema
                    d = paths.DATA_DIR / "cands" / f"v1_n{a.norm_v}"
                    d.mkdir(parents=True, exist_ok=True)
                    fw = pq.ParquetWriter(str(d / "test.parquet"), fschema, compression="zstd")
                fw.write_table(pa.Table.from_pandas(f, schema=fschema, preserve_index=False), row_group_size=max(1, len(f)))
            p = model.predict(f[feats], num_threads=0).astype(np.float32)
            keep = p >= FLOOR
            preds.append(f.loc[keep, ["s1_id", "cand_id"]].assign(prob=p[keep]))
            n_pairs += len(f)
            del f, pool_part, pr
            gc.collect()
        timing[ctry] = dict(queries=int(q.height), pairs=n_pairs, blocking_s=round(t_blk), total_s=round(time.time() - t0),
                            peak_rss_gb=round(bv1.rss_gb(), 1), retriever_s={k: round(v) for k, v in tim.items()})
        print(f"[{ctry}] {timing[ctry]}", flush=True)
        del U, q, qx, ptbl
        gc.collect()
    cw.close()
    if fw is not None:
        fw.close()
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(out / "test_pred.parquet")
    m = threshold_matches(assign_best_s1(pred), t)
    pd.DataFrame([(s, x) for s, xs in m.items() for x in xs], columns=["s1_id", "match_id"]) \
      .to_parquet(out / "test_matches.parquet")
    timing["_all"] = dict(frac=a.frac, minutes=round((time.time() - t_all) / 60, 1), threshold=t,
                          n_matches=int(sum(map(len, m.values()))), n_s1_nonempty=len(m))
    (out / "test_timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing["_all"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", required=True)
    ap.add_argument("--norm-v", type=int, default=1)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=40_000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-feats", action="store_true")
    main(ap.parse_args())
