"""EXP-030: add the `nkey_num` retriever (ber.blocking.nkey_num_keys, block cap 50) to an existing v1 cache,
incrementally -> data/cands/<dst>/<file>.parquet. New pairs (not already in the source cache) are featurized with the
SAME baseline featurizer (kmask 0, rbits = RBITS['nkey_num'], cos_* null) and appended OUTSIDE the 100/S1 cap; the
candidate-set-dependent columns (name_rank_in_s1, addr_rank_in_s1, name_gap_s1, n_cand_s1) are recomputed per S1.
The same procedure runs for train / eval subsets / test, so train, val and test stay consistent.

Query sets: train -> data/cands/train_ids.parquet; mini/fold0x -> ber.split eval ids; test -> ALL test S1.
test.parquet is streamed row group by row group (S1-complete groups are preserved; S1s that had no candidates
and now get some are appended as extra row groups), so it doubles as the candidate set for the submission.

    python pipelines/augment_nkey_num.py --files mini            # measure (recall before/after is logged)
    python pipelines/augment_nkey_num.py --files train mini fold0x test
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import baseline_v0 as bv0
from ber import blocking as B
from ber import io, paths, split
from ber.tracking import Run

CAP = 50
CTX_COLS = ["name_rank_in_s1", "addr_rank_in_s1", "name_gap_s1", "n_cand_s1"]


def query_ids(name: str) -> set:
    if name == "train":
        return set(pd.read_parquet(paths.DATA_DIR / "cands" / "train_ids.parquet").s1_id)
    if name == "test":
        return set(pl.read_parquet(B.norm_file("test", 1, 1), columns=["entity_id"])["entity_id"].to_list())
    return split.eval_ids(io.load_folds(), name)


def new_pairs(split_name: str, qids: set, existing: pl.DataFrame, norm_v: int, log) -> pd.DataFrame:
    """Featurized nkey_num pairs for the query S1s that are NOT already in `existing` (s1_id, cand_id)."""
    bv0.NORM_V = norm_v
    out = []
    s1_all = pl.read_parquet(B.norm_file(split_name, 1, norm_v), columns=B.POOL_COLS)
    for ctry in B.countries_of(split_name, norm_v):
        t0 = time.time()
        q = s1_all.filter((pl.col("country") == ctry) & pl.col("entity_id").is_in(list(qids)))
        if q.height == 0:
            continue
        ptbl = B.pool_table(split_name, ctry, norm_v)
        pool = pl.from_arrow(ptbl.select(["entity_id", "n_core", "a_numbers"]))
        pr = B.multikey_join(B.nkey_num_keys(q, "i"), B.nkey_num_keys(pool, "j"), CAP)
        pr = pr.with_columns(pl.Series("s1_id", q["entity_id"].to_numpy()[pr["i"].to_numpy()]),
                             pl.Series("cand_id", pool["entity_id"].to_numpy()[pr["j"].to_numpy()]))
        n_all = pr.height
        pr = pr.join(existing, on=["s1_id", "cand_id"], how="anti")
        log(f"  [{ctry}] q={q.height:,} nkey_num pairs {n_all:,} (+{n_all / q.height:.2f}/S1), new {pr.height:,} "
            f"(+{pr.height / q.height:.2f}/S1) {time.time() - t0:.0f}s")
        if pr.height:
            pool_part = bv0.fetch_pool(ptbl, pr["j"].to_numpy())
            f = bv0.featurize(pr.select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32),
                                        pl.lit(0, dtype=pl.Int8).alias("kmask")),
                              q.with_row_index("idx"), pool_part)
            f["rbits"] = np.int32(B.RBITS["nkey_num"])
            f["cos_name"] = np.float32(np.nan)
            f["cos_na"] = np.float32(np.nan)
            out.append(f)
        del ptbl, pool, pr
        gc.collect()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def recompute_ctx(f: pd.DataFrame) -> pd.DataFrame:
    """Same definitions as baseline_v0.featurize (per S1 over its full candidate set)."""
    g = f.groupby("s1_id")
    f["name_rank_in_s1"] = g["name_tset"].rank(ascending=False, method="min").astype(np.float32)
    f["addr_rank_in_s1"] = g["addr_tset"].rank(ascending=False, method="min").astype(np.float32)
    f["name_gap_s1"] = g["name_tset"].transform("max") - f["name_tset"]
    f["n_cand_s1"] = g["cand_id"].transform("size").astype(np.int32)
    return f


def align(new: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    for c in like.columns:
        if c not in new.columns:
            new[c] = pd.NA
    return new[like.columns].astype(like.dtypes.to_dict())


def label(f: pd.DataFrame) -> np.ndarray:
    g = io.load_gt_pairs(set(f.s1_id))
    pos = pd.MultiIndex.from_arrays([g.s1_id, g.match_id])
    return pd.MultiIndex.from_arrays([f.s1_id, f.cand_id]).isin(pos)


def do_file(name: str, a, run, log) -> dict:
    t0 = time.time()
    src = paths.DATA_DIR / "cands" / a.src / f"{name}.parquet"
    dst_dir = paths.DATA_DIR / "cands" / a.dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    split_name = "test" if name == "test" else "train"
    qids = query_ids(name)
    existing = pl.read_parquet(src, columns=["s1_id", "cand_id"])
    new = new_pairs(split_name, qids, existing, a.norm_v, log)
    rep = {"queries": len(qids), "old_pairs": existing.height, "new_pairs": len(new),
           "new_per_s1": round(len(new) / max(1, len(qids)), 3)}
    if name != "test":
        old = pd.read_parquet(src)
        if len(new):
            new["label"] = label(new)
            if "role" in old.columns:
                role = old.drop_duplicates("s1_id").set_index("s1_id")["role"]
                new["role"] = new.s1_id.map(role).fillna("eval" if name != "train" else "train")
                if name == "train":      # S1s that had no candidates before: role from the sample file
                    ids = pd.read_parquet(paths.DATA_DIR / "cands" / "train_ids.parquet").set_index("s1_id")["role"]
                    new["role"] = new.s1_id.map(ids).fillna(new["role"])
            f = recompute_ctx(pd.concat([old, align(new, old)], ignore_index=True))
        else:
            f = old
        if name != "train":
            n_true = sum(len(v) for v in io.load_gt_sets(qids).values())
            rep.update(recall_before=round(float(old.label.sum()) / n_true, 5), recall_after=round(float(f.label.sum()) / n_true, 5),
                       true_new=int(new.label.sum()) if len(new) else 0)
        f.to_parquet(dst_dir / f"{name}.parquet", index=False)
        rep["pairs"] = len(f)
    else:
        pf = pq.ParquetFile(src)
        writer, schema, seen, n = None, None, set(), 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg).to_pandas()
            s1s = t.s1_id.unique()
            seen.update(s1s)
            add = new[new.s1_id.isin(s1s)] if len(new) else new
            if len(add):
                t = recompute_ctx(pd.concat([t, align(add.copy(), t)], ignore_index=True))
            if writer is None:
                schema = pa.Table.from_pandas(t, preserve_index=False).schema
                writer = pq.ParquetWriter(str(dst_dir / "test.parquet"), schema, compression="zstd")
            writer.write_table(pa.Table.from_pandas(t, schema=schema, preserve_index=False), row_group_size=max(1, len(t)))
            n += len(t)
        rest = new[~new.s1_id.isin(seen)] if len(new) else new
        if len(rest):
            t = recompute_ctx(align(rest.copy(), pf.read_row_group(0).to_pandas()))
            writer.write_table(pa.Table.from_pandas(t, schema=schema, preserve_index=False), row_group_size=max(1, len(t)))
            n += len(t)
            rep["s1_newly_nonempty"] = int(t.s1_id.nunique())
        writer.close()
        rep["pairs"] = n
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    log(f"[{name}] {rep}")
    return rep


def main(a):
    with Run(f"blocking-nkey-num-{a.dst}", hypothesis="EXP-030: exact name key + shared address number recovers "
             "native-script generic-name blocking misses (+0.0015 mini upper bound)",
             params={"src": a.src, "dst": a.dst, "cap": CAP, "norm_v": a.norm_v, "files": a.files},
             tags=["blocking"], parent="20260925-2015_aryan-bhendarkar_gate-blocking-v1-n1") as run:
        reps = {}
        for name in a.files:
            reps[name] = do_file(name, a, run, lambda s: print(s, flush=True))
            run.log(files=reps)
        print(json.dumps(reps, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v1_n1")
    ap.add_argument("--dst", default="v1_n2")
    ap.add_argument("--norm-v", type=int, default=1)
    ap.add_argument("--files", nargs="+", default=["mini"])
    main(ap.parse_args())
