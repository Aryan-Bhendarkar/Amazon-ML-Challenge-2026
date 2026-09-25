"""Candidate + feature cache (Phase 0.3) so model/decision experiments skip blocking.

cand_ver=keys_v0: baseline_v0 blocking + its 28 features, for
  train.parquet  : baseline training sample (seed 42: 150k fold>=2 'train' + 30k fold-1 'es'), label from GT
  <subset>.parquet: eval subset S1 (role 'eval'), candidates from the FULL train pool of the country
Columns: s1_id, cand_id, label, role, + baseline feature columns (kmask included).
Also writes data/cands/train_ids.parquet (s1_id, role).
cand_ver=v1: blocking v1 union (pipelines/blocking_v1.run_country, capped per S1) for train_ids + subset in ONE
pass per country, the SAME baseline featurizer, plus rbits (retriever bitmask), cos_name, cos_na.
Written to data/cands/v1_n<norm_v>/{train,<subset>}.parquet.

    python pipelines/build_cache.py --cand-ver keys_v0 [--subset mini] [--n-train 150000]
    python pipelines/build_cache.py --cand-ver v1 --norm-v 1 [--subset mini]
"""
import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
import pandas as pd

import baseline_v0 as bv0
from ber import harness, io, paths

SEED = 42


def train_ids(n_train: int) -> pd.DataFrame:
    folds = io.load_folds()
    rng = np.random.default_rng(SEED)
    tr_pool = np.sort(folds[folds.fold >= 2].s1_id.to_numpy())
    es_pool = np.sort(folds[folds.fold == 1].s1_id.to_numpy())
    tr = rng.choice(tr_pool, size=min(n_train, len(tr_pool)), replace=False)
    es = rng.choice(es_pool, size=min(n_train // 5, len(es_pool)), replace=False)
    return pd.concat([pd.DataFrame({"s1_id": tr, "role": "train"}), pd.DataFrame({"s1_id": es, "role": "es"})],
                     ignore_index=True)


def label(f: pd.DataFrame, ids) -> pd.Series:
    g = io.load_gt_pairs(ids)
    pos = pd.MultiIndex.from_arrays([g.s1_id, g.match_id])
    return pd.Series(pd.MultiIndex.from_arrays([f.s1_id, f.cand_id]).isin(pos), index=f.index)


def build_v1(a):
    import gc

    import polars as pl

    import blocking_v1 as bv1
    from ber import blocking as B
    bv0.NORM_V = a.norm_v
    out = paths.ROOT / "data" / "cands" / f"v1_n{a.norm_v}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ids = train_ids(a.n_train)
    ctx = harness.EvalContext.load(a.subset)
    role = dict(zip(ids.s1_id, ids.role))
    role.update({s: "eval" for s in ctx.ids})
    all_ids = set(role)
    parts = {"train": [], "eval": []}
    for ctry in B.countries_of("train", a.norm_v):
        ids_c = all_ids & set(pl.read_parquet(B.norm_file("train", 1, a.norm_v), columns=["entity_id", "country"])
                              .filter(pl.col("country") == ctry)["entity_id"].to_list())
        _, U, tim, q, ptbl = bv1.run_country("train", ctry, ids_c, print, a.norm_v, keep_frames=True)
        U = B.cap_per_query(U, "score", bv1.P["cap"])
        qx = q.with_row_index("idx")
        for o in range(0, q.height, a.chunk):
            pr = U.filter((pl.col("i") >= o) & (pl.col("i") < o + a.chunk))
            if pr.height == 0:
                continue
            pool_part = bv0.fetch_pool(ptbl, pr["j"].to_numpy())
            f = bv0.featurize(pr.select("i", "j", "kmask"), qx, pool_part)
            f = f.merge(pr.select("s1_id", "cand_id", "rbits", "cos_name", "cos_na").to_pandas(), on=["s1_id", "cand_id"],
                        how="left", validate="one_to_one")
            f["role"] = f.s1_id.map(role)
            for r, key in (("eval", "eval"), ("train", "train"), ("es", "train")):
                parts[key].append(f[f.role == r])
            del f, pool_part
            gc.collect()
        print(f"[{ctry}] featurized {(time.time() - t0) / 60:.1f} min", flush=True)
        del U, q, qx, ptbl
        gc.collect()
    for key, name in (("train", "train"), ("eval", a.subset)):
        f = pd.concat(parts[key], ignore_index=True)
        f["label"] = label(f, set(f.s1_id)).to_numpy()
        f.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(f):,} pairs, pos rate {f.label.mean():.4f}", flush=True)
    print(f"done {(time.time() - t0) / 60:.1f} min")


def main(a):
    if a.cand_ver == "v1":
        return build_v1(a)
    assert a.cand_ver == "keys_v0"
    out = paths.ROOT / "data" / "cands" / a.cand_ver
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not a.skip_train:
        ids = train_ids(a.n_train)
        ids.to_parquet(out.parent / "train_ids.parquet", index=False)
        f, _ = bv0.collect("train", set(ids.s1_id), a.chunk)
        f["label"] = label(f, set(ids.s1_id)).to_numpy()
        f["role"] = f.s1_id.map(dict(zip(ids.s1_id, ids.role)))
        f.to_parquet(out / "train.parquet", index=False)
        print(f"train: {len(f):,} pairs, pos rate {f.label.mean():.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)
        del f
    ctx = harness.EvalContext.load(a.subset)
    f, _ = bv0.collect("train", ctx.ids, a.chunk)
    f["label"] = label(f, ctx.ids).to_numpy()
    f["role"] = "eval"
    f.to_parquet(out / f"{a.subset}.parquet", index=False)
    print(f"{a.subset}: {len(f):,} pairs, pos rate {f.label.mean():.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-ver", default="keys_v0")
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0", "fold0x"])
    ap.add_argument("--n-train", type=int, default=150_000)
    ap.add_argument("--chunk", type=int, default=40_000)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--norm-v", type=int, default=0)
    main(ap.parse_args())
