"""Test-time run of the model_v2 pipeline (blocking v1 retrievers minus tf_name -> learned pruner top-K ->
baseline + blocking + ctx features -> matcher -> assign_best_s1 over ALL test S1 -> threshold).

Per country (discovered from the data, France included): TF-IDF views / pool keys built once, S1 queries
processed in chunks (complete S1 groups). IDF, key counts and ctx statistics come from the TEST split only
(unlabeled, transductive). Candidates are written one row group per chunk (-> candidate_pairs.tsv).

    python pipelines/test_v2.py --run-id <model_v2 run_id> [--keep 40] [--chunk 100000]
Out: artifacts/<run_id>/test_candidates.parquet, test_pred.parquet, test_matches.parquet
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import baseline_v0 as bv0
import cands_v2 as cv
import model_v2 as mv
from ber import blocking as B
from ber import ctx_features as cf
from ber import paths
from ber.decision import assign_best_s1, threshold_matches

NORM_V = mv.NORM_V
SPLIT = "test"


def main(a):
    t00 = time.time()
    log = lambda s: print(f"[{(time.time() - t00) / 60:5.1f}m] {s}", flush=True)  # noqa: E731
    art = paths.ART_DIR / a.run_id
    model = lgb.Booster(model_file=str(art / "model.lgb"))
    feats = json.loads((art / "features.json").read_text())
    t = a.threshold if a.threshold is not None else json.loads((art / "decision.json").read_text())["threshold"]
    pruner = lgb.Booster(model_file=str(mv.CANDS / "pruner_nt.lgb"))
    sctx = cf.SplitContext.build(SPLIT)
    P = cv.P
    cand_schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string())])
    cw = pq.ParquetWriter(str(art / "test_candidates.parquet"), cand_schema, compression="zstd")
    preds, n_cand = [], 0
    keepcols = ["s1_id", "cand_id", "p_prune", "r_prune"] + mv.PRUNE_FEATS
    for ctry in B.countries_of(SPLIT, NORM_V):
        s1 = (pl.scan_parquet(B.norm_file(SPLIT, 1, NORM_V)).filter(pl.col("country") == ctry)
                .select(bv0.COLS).collect().with_row_index("idx"))
        ptbl = B.pool_table(SPLIT, ctry, NORM_V)
        pool = pl.from_arrow(ptbl.select(["entity_id", "n_core", "a_street", "a_full", "a_house", "a_empty"]))
        log(f"[{ctry}] s1={s1.height:,} pool={pool.height:,}")
        freq = B.token_freq(ptbl)
        pk = B.cap_keys(B.build_keys(ptbl, freq))
        pak = pool.select(B.akey_expr()).to_series()
        phs = pool.select(B.hskey_expr()).to_series()
        psk = B.skeleton(pool["n_core"])
        V = B.TfViews(s1, pool, P["max_df_name"], P["max_df_addr"])
        Pna = V.na("pool", None, P["w_na"])
        emp = np.flatnonzero(pool["a_empty"].to_numpy())
        PnE = V.Pn[emp]
        pid = pool["entity_id"].to_numpy()
        log(f"[{ctry}] views + pool keys ready")
        for o in range(0, s1.height, a.chunk):
            qrow = np.arange(o, min(o + a.chunk, s1.height))
            q = s1[qrow]
            parts = {}
            qk = B.build_keys(q.select(B.KEY_COLS).to_arrow(), freq)
            c = qk.join(pk, on=["key", "kt"], how="inner", suffix="_p")
            parts["keys_v0"] = (c.select(pl.col("idx").cast(pl.UInt32).alias("i"), pl.col("idx_p").cast(pl.UInt32).alias("j"))
                                  .unique())
            Qn = V.Sn[qrow]
            Qa = V.Sa[qrow]
            Qna = V.na("s1", qrow, P["w_na"])
            parts["tf_na"] = B.topk_frame(Qna, Pna, P["k_na"]).select("i", "j")
            parts["tf_empty"] = B.topk_frame(Qn, PnE, P["k_empty"], p_map=emp).select("i", "j")
            parts["akey"] = B.key_join(q.select(B.akey_expr()).to_series(), pak, P["cap_akey"])
            parts["hskey"] = B.key_join(q.select(B.hskey_expr()).to_series(), phs, P["cap_hskey"])
            parts["skel"] = B.key_join(B.skeleton(q["n_core"]), psk, P["cap_skel"], min_len=4)
            parts = {k: v.with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)) for k, v in parts.items()}
            U = B.union(parts)
            del parts, qk, c
            i, j = U["i"].to_numpy().astype(np.int64), U["j"].to_numpy().astype(np.int64)
            U = U.with_columns(pl.Series("cos_name", B.rowdot(Qn, V.Pn, i, j)),
                               pl.Series("cos_addr", B.rowdot(Qa, V.Pa, i, j)),
                               pl.Series("cos_na", B.rowdot(Qna, Pna, i, j)),
                               pl.Series("s1_id", q["entity_id"].to_numpy()[i]),
                               pl.Series("cand_id", pid[j]))
            n_union = U.height
            U = mv.add_prune_feats(U)
            pp = pruner.predict(U.select(mv.PRUNE_FEATS).to_pandas(), num_threads=0).astype(np.float32)
            U = U.with_columns(pl.Series("p_prune", pp))
            U = U.with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("s1_id").alias("r_prune"))
            K = U.filter(pl.col("r_prune") <= a.keep)
            del U
            pairs = pl.DataFrame({"i": qrow[K["i"].to_numpy()].astype(np.uint32), "j": K["j"].to_numpy().astype(np.uint32),
                                  "kmask": K["rbits"].to_numpy().astype(np.int32)})
            f = bv0.featurize(pairs, s1, bv0.fetch_pool(ptbl, pairs["j"].to_numpy()))
            F = pl.from_pandas(f).join(K.select(keepcols), on=["s1_id", "cand_id"], how="left")
            del f, pairs, K
            X = cf.add_features(cf.attach_norm(F.select("s1_id", "cand_id", "name_tset"), SPLIT), sctx)
            new = cf.new_feature_cols(X, ["name_tset"])
            F = F.join(X.select(["s1_id", "cand_id"] + new), on=["s1_id", "cand_id"], how="left")
            del X
            missing = [c for c in feats if c not in F.columns]
            assert not missing, missing
            prob = model.predict(F.select(feats).to_pandas(), num_threads=0).astype(np.float32)
            ct = pa.Table.from_arrays([F["s1_id"].to_arrow(), F["cand_id"].to_arrow()], schema=cand_schema)
            cw.write_table(ct, row_group_size=max(1, ct.num_rows))
            n_cand += ct.num_rows
            keep = prob >= 0.05
            preds.append(pd.DataFrame({"s1_id": F["s1_id"].to_numpy()[keep], "cand_id": F["cand_id"].to_numpy()[keep],
                                       "prob": prob[keep]}))
            log(f"[{ctry}] {qrow[-1] + 1:,}/{s1.height:,} S1: union {n_union / len(qrow):.0f}/q, kept {ct.num_rows:,}, "
                f"prob>={t}: {(prob >= t).sum():,}")
            del F, ct, prob, Qn, Qa, Qna
            gc.collect()
        del V, Pna, PnE, pk, pak, phs, psk, pool, ptbl, s1
        gc.collect()
    cw.close()
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(art / "test_pred.parquet")
    m = threshold_matches(assign_best_s1(pred), t)
    pd.DataFrame([(s, x) for s, xs in m.items() for x in xs], columns=["s1_id", "match_id"]) \
      .to_parquet(art / "test_matches.parquet")
    log(f"done: {n_cand:,} candidate pairs, {sum(map(len, m.values())):,} matches for {len(m):,} S1 (t={t})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--threshold", type=float, default=None)
    main(ap.parse_args())
