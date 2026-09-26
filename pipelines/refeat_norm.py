"""Lane C / EXP-F2: re-featurize a candidate cache's BASE pair features on a newer normalization.

Only for countries whose normalization changed (data-derived: S1 rows of norm_v<new> that differ from
norm_v<old>; with norm v2 that is France only, US/India are byte-identical). Retrieval columns
(kmask, rbits, cos_name, cos_na) and the candidate SET are kept from the cache, so candidate_pairs.tsv
is unchanged. Per-S1 rank features are recomputed over each S1's full candidate list (baseline_v0
definitions, complete S1 groups per chunk). Output keeps the cache schema and writes one row group per
chunk of complete S1 (what predict_test_v1 expects):

    data/cands/<cache>/<tag>_n<new>.parquet

Check mode (run this on train/mini, never needs test): recompute ALL rows of the given countries and
assert they equal the cached features (proves the recompute path is faithful before touching test):

    python pipelines/refeat_norm.py --cache v1_n2 --tag mini --check --countries US,India --limit-s1 3000
    python pipelines/refeat_norm.py --cache v1_n2 --tag test          # test lane only (France pairs)

Then score with ctx features on the same norm (US/India ctx inputs unchanged under v2):
    python pipelines/predict_test_v1.py --run <run> --cache v1_n2 --src-tag test_n2 --norm 2 --out-suffix _n2
"""
import _bootstrap  # noqa: F401

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import baseline_v0 as bv0
from ber import paths

CANDS = paths.DATA_DIR / "cands"
KEEP = ["s1_id", "cand_id", "kmask", "rbits", "cos_name", "cos_na", "role", "label"]   # never recomputed
CMP = ["n_core", "n_compact", "n_alias", "n_legal", "n_kind", "a_full", "a_street", "a_numbers", "a_house",
       "a_state", "a_empty"]


def norm_file(split: str, s: int, v: int):
    return paths.FEATURE_DIR / f"norm_v{v}_{split}_s{s}.parquet"


def changed_countries(split: str, old: int, new: int) -> list[str]:
    """Countries with at least one S1 or pool row whose normalized fields differ between versions."""
    out = set()
    for s in (1, 2, 3):
        a = pl.scan_parquet(norm_file(split, s, old)).select(["entity_id", "country"] + CMP)
        b = pl.scan_parquet(norm_file(split, s, new)).select(["entity_id"] + CMP)
        d = (a.join(b, on="entity_id", suffix="_n")
              .filter(pl.any_horizontal([pl.col(c).ne_missing(pl.col(f"{c}_n")) for c in CMP]))
              .select("country").unique().collect())
        out |= set(d["country"].to_list())
    return sorted(out)


def load_norm(split: str, v: int, ids: pl.Series, sources) -> pl.DataFrame:
    parts = [pl.scan_parquet(norm_file(split, s, v)).filter(pl.col("entity_id").is_in(ids.implode()))
               .select(bv0.COLS) for s in sources]
    return pl.concat(parts).collect()


def refeat(pairs: pl.DataFrame, split: str, v: int) -> pl.DataFrame:
    """Base features for complete S1 groups on norm v (baseline_v0.featurize, the cache's featurizer)."""
    s1 = load_norm(split, v, pairs["s1_id"].unique(), (1,)).with_row_index("idx")
    pool = load_norm(split, v, pairs["cand_id"].unique(), (2, 3)).with_row_index("idx")
    pr = (pairs.select("s1_id", "cand_id", "kmask")
               .join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("i")), on="s1_id")
               .join(pool.select(pl.col("entity_id").alias("cand_id"), pl.col("idx").alias("j")), on="cand_id"))
    assert pr.height == pairs.height, f"norm rows missing for {pairs.height - pr.height} pairs"
    f = bv0.featurize(pr.select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32), "kmask"), s1, pool)
    return pl.from_pandas(f.drop(columns=["kmask"]))


def _n_diff(a: pl.Series, b: pl.Series) -> int:
    """#rows that differ (null == null, NaN == NaN)."""
    x, y = a.to_numpy(), b.to_numpy()
    same = (x == y) | (pd.isna(x) & pd.isna(y))
    return int((~same).sum())


def check(a, split: str, cs: list[str]) -> None:
    """Recompute every row of countries `cs` (load-all) and assert equality with the cache."""
    t0 = time.time()
    cache = pl.read_parquet(CANDS / a.cache / f"{a.tag}.parquet")
    schema = cache.schema
    ctry = pl.scan_parquet(norm_file(split, 1, a.old)).select("entity_id", "country").collect()
    todo = cache.join(ctry.rename({"entity_id": "s1_id", "country": "_ctry"}), on="s1_id", how="left") \
                .filter(pl.col("_ctry").is_in(cs))
    if a.limit_s1:
        keep = todo["s1_id"].unique().sort().sample(min(a.limit_s1, todo["s1_id"].n_unique()), seed=42)
        todo = todo.filter(pl.col("s1_id").is_in(keep.implode()))
    feat_cols = [c for c in schema if c not in KEEP]
    todo = todo.with_columns((pl.col("s1_id").hash(seed=7) % a.chunks).alias("_ch"))
    for ch in range(a.chunks):
        P = todo.filter(pl.col("_ch") == ch)
        if not P.height:
            continue
        new = recompute(P.select(list(schema)), schema, split, a.new)
        old = P.select(list(schema)).sort(["s1_id", "cand_id"])
        nw = new.sort(["s1_id", "cand_id"])
        bad = {c: n for c in feat_cols if (n := _n_diff(old[c], nw[c]))}
        print(f"  chunk {ch}: {P.height:,} pairs, mismatching cols {bad or 'NONE'}", flush=True)
        assert not bad, f"recompute differs from cache: {bad}"
    print(f"CHECK PASSED: recompute == cache for {todo.height:,} pairs of {cs} ({time.time() - t0:.0f}s)")


def recompute(P: pl.DataFrame, schema, split: str, v: int) -> pl.DataFrame:
    """Rows P (complete S1 lists) with base features recomputed on norm v; KEEP columns from the cache."""
    F = refeat(P, split, v)
    new = P.select([c for c in schema if c in KEEP]).join(F, on=["s1_id", "cand_id"], how="left")
    return new.select(list(schema)).cast(dict(schema))


def main(a):
    t0 = time.time()
    split = "test" if a.tag.startswith("test") else "train"
    cs = a.countries.split(",") if a.countries else changed_countries(split, a.old, a.new)
    print(f"[{a.tag}] re-featurize countries {cs} on norm_v{a.new} {time.time() - t0:.0f}s", flush=True)
    if a.check:
        return check(a, split, cs)
    # streaming write: row-group batches of complete S1 lists (the layout predict_test_v1 relies on)
    src = CANDS / a.cache / f"{a.tag}.parquet"
    pf = pq.ParquetFile(src)
    ctry = pl.scan_parquet(norm_file(split, 1, a.old)).select("entity_id", "country").collect() \
             .rename({"entity_id": "s1_id", "country": "_ctry"})
    dst = CANDS / a.cache / f"{a.tag}_n{a.new}{'_slice' if a.limit_rows else ''}.parquet"
    tmp = dst.with_suffix(".tmp")
    w, schema, seen, n_in, n_re, batch, rows = None, None, set(), 0, 0, [], 0

    def flush(batch):
        nonlocal w, schema, n_in, n_re
        X = pl.from_arrow(pf.read_row_groups(batch))
        schema = schema or X.schema
        ids = set(X["s1_id"].unique().to_list())
        assert not (ids & seen), "an S1 spans row-group batches -> per-S1 rank features would be wrong"
        seen.update(ids)
        X = X.join(ctry, on="s1_id", how="left")
        P = X.filter(pl.col("_ctry").is_in(cs)).drop("_ctry")
        inc = P.group_by("s1_id").agg(pl.len().alias("n"), pl.col("n_cand_s1").first()).filter(pl.col("n") != pl.col("n_cand_s1"))
        assert not inc.height, f"{inc.height} S1 with incomplete candidate lists in this batch (row groups not S1-aligned)"
        R = X.filter(~pl.col("_ctry").is_in(cs) | pl.col("_ctry").is_null()).drop("_ctry")
        out = pl.concat([recompute(P, schema, split, a.new), R]) if P.height else R
        assert out.height == X.height
        tb = out.select(list(schema)).to_arrow()
        if w is None:
            w = pq.ParquetWriter(str(tmp), tb.schema, compression="zstd")
        w.write_table(tb.cast(w.schema), row_group_size=tb.num_rows)
        n_in += X.height
        n_re += P.height
        print(f"  {n_in:,} pairs ({n_re:,} re-featurized) {time.time() - t0:.0f}s", flush=True)
        del X, P, R, out, tb
        gc.collect()

    for rg in range(pf.num_row_groups):
        batch.append(rg)
        rows += pf.metadata.row_group(rg).num_rows
        if rows >= a.batch_rows:
            flush(batch)
            batch, rows = [], 0
            if a.limit_rows and n_in >= a.limit_rows:
                break
    if batch and not (a.limit_rows and n_in >= a.limit_rows):
        flush(batch)
    w.close()
    n = pq.ParquetFile(tmp).metadata.num_rows
    if not a.limit_rows:
        assert n == pf.metadata.num_rows, f"row count {n} != cache {pf.metadata.num_rows}"
    os.replace(tmp, dst)
    print(f"wrote {dst} ({n:,} pairs, {n_re:,} re-featurized, candidate set unchanged) in "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v1_n2")
    ap.add_argument("--tag", default="mini")
    ap.add_argument("--old", type=int, default=1)
    ap.add_argument("--new", type=int, default=2)
    ap.add_argument("--countries", default="", help="override the data-derived changed-country list")
    ap.add_argument("--check", action="store_true", help="assert recompute == cache (use unchanged countries)")
    ap.add_argument("--limit-s1", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=16, help="--check only")
    ap.add_argument("--batch-rows", type=int, default=3_000_000)
    ap.add_argument("--limit-rows", type=int, default=0, help="stream only a slice (writes <tag>_n<v>_slice)")
    main(ap.parse_args())
