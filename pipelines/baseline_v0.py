"""EXP-001 baseline: key blocking + rapidfuzz pair features + LightGBM + assign/threshold.

Memory: processed per country (discovered from data, never hard-coded), S1 queries in chunks,
keys hashed to u64, pool attributes fetched only for candidate rows. Peak memory is key
building for the largest country (~2 GB, US train) + training pairs (~0.25 KB/pair).
The team laptop has only ~5 GB truly free RAM with apps open -> run `micro` locally; run
`mini`/test on SageMaker or a Kaggle notebook (30 GB RAM) if memory is tight.

Val run (trains on folds 1-4 sample, evaluates on the eval subset, logs a tracked Run):
    python pipelines/baseline_v0.py --subset mini --n-train 120000
Test run (reuses the model + threshold saved by the val run):
    python pipelines/baseline_v0.py --test --run-id <val run_id>
Outputs for /submit: artifacts/<run_id>/test_candidates.parquet, test_matches.parquet
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ber import harness, io, paths
from ber.decision import assign_best_s1, threshold_matches
from ber.normalize import STREET_CANON
from ber.tracking import Run

NORM_V = 0
SEED = 42
COLS = ["entity_id", "country", "n_core", "n_compact", "n_alias", "n_legal", "n_kind", "n_script",
        "a_full", "a_street", "a_numbers", "a_house", "a_state", "a_empty"]
STREET_STOP = sorted(set(STREET_CANON.values()) | {"road", "street", "near", "opp", "post", "office", "city",
                                                     "main", "cross", "floor", "shop", "unit", "pmb", "box", "rue"})
CAPS = {"A": 150, "N1": 200, "N2": 100, "C": 60}      # max POOL records per key (drop bigger blocks)
KTYPE = {"A": 1, "N1": 2, "N2": 4, "C": 8}
SLICE = 400_000     # rows per slice when exploding tokens into keys (memory knob)


# ============================================================================ loading
KEY_COLS = ["entity_id", "n_core", "n_compact", "n_alias", "a_house", "a_street"]


def _norm_file(split: str, source: int):
    return paths.FEATURE_DIR / f"norm_v{NORM_V}_{split}_s{source}.parquet"


def load_norm(split: str, source: int, country: str, cols=COLS) -> pl.DataFrame:
    return (pl.scan_parquet(_norm_file(split, source)).filter(pl.col("country") == country)
              .select(cols).collect())


def pool_table(split: str, country: str) -> pa.Table:
    """S2+S3 of one country as a MEMORY-MAPPED Arrow IPC table (built once, streamed, cached).
    Row number = pool idx. mmap pages are file-backed, so they don't eat Windows commit memory."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    p = paths.FEATURE_DIR / f"pool_v{NORM_V}_{split}_{safe}.arrow"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        writer = None
        with pa.OSFile(str(tmp), "wb") as sink:
            for s in (2, 3):
                for b in pq.ParquetFile(_norm_file(split, s)).iter_batches(batch_size=200_000, columns=COLS):
                    t = pa.Table.from_batches([b])
                    t = t.filter(pc.equal(t["country"], country))
                    if writer is None:
                        writer = pa.ipc.new_file(sink, t.schema)
                    writer.write_table(t)
            if writer is not None:
                writer.close()
        os.replace(tmp, p)
    return pa.ipc.open_file(pa.memory_map(str(p), "r")).read_all()


def fetch_pool(tbl: pa.Table, idx: np.ndarray) -> pl.DataFrame:
    """Full attributes only for the candidate rows (zero-copy mmap + take)."""
    u = np.unique(idx).astype(np.int64)
    return pl.from_arrow(tbl.take(pa.array(u))).with_columns(pl.Series("idx", u.astype(np.uint32)))


def countries_of(split: str) -> list[str]:
    p = paths.FEATURE_DIR / f"norm_v{NORM_V}_{split}_s1.parquet"
    return sorted(pl.read_parquet(p, columns=["country"])["country"].unique().to_list())


# ============================================================================ blocking
def _hk(df: pl.DataFrame, expr: pl.Expr, kt: int) -> pl.DataFrame:
    """Hash string keys immediately (u64) and tag the key type (int8) -> low memory."""
    return df.select(pl.col("idx"), expr.hash(seed=7).alias("key"), pl.lit(kt, dtype=pl.Int8).alias("kt"))


def _core_tokens(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select("idx", pl.col("n_core").str.split(" ").alias("t")).explode("t", empty_as_null=True)
              .filter(pl.col("t").str.len_chars() >= 3).unique(["idx", "t"]))


def _slices(tbl: pa.Table, slice_rows: int):
    """Yield polars slices (with global row index `idx`) of the key columns of an Arrow table."""
    for o in range(0, tbl.num_rows, slice_rows):
        part = pl.from_arrow(tbl.slice(o, slice_rows).select(KEY_COLS[1:]))
        yield part.with_row_index("idx", offset=o)


def token_freq(tbl: pa.Table, slice_rows: int) -> pl.DataFrame:
    """Core-token document frequency, computed slice by slice (bounded memory)."""
    parts = [_core_tokens(d).group_by("t").len() for d in _slices(tbl, slice_rows)]
    return pl.concat(parts).group_by("t").agg(pl.col("len").sum())


def make_keys(df: pl.DataFrame, freq: pl.DataFrame) -> pl.DataFrame:
    """Return keys[idx, key:u64, kt:i8] for the rows of df (call per slice). df must have column idx."""
    parts = []
    # A: house number + each significant street token (robust to component reordering)
    a = (df.filter(pl.col("a_house") != "")
           .select("idx", "a_house", pl.col("a_street").str.split(" ").alias("t")).explode("t", empty_as_null=True)
           .filter((pl.col("t").str.len_chars() >= 4) & ~pl.col("t").is_in(STREET_STOP)))
    parts.append(_hk(a, pl.col("a_house") + "|" + pl.col("t"), KTYPE["A"]))
    del a
    # N1/N2: the two RAREST core-name tokens (rarity within this country)
    tok = _core_tokens(df)
    tok = tok.join(freq, on="t", how="left").with_columns(pl.col("len").fill_null(0))
    top2 = tok.sort(["idx", "len", "t"]).group_by("idx", maintain_order=True).head(2)
    del tok
    parts.append(_hk(top2.group_by("idx", maintain_order=True).first(), pl.col("t"), KTYPE["N1"]))
    n2 = top2.group_by("idx").agg(pl.col("t").sort()).filter(pl.col("t").list.len() == 2)
    parts.append(_hk(n2, pl.col("t").list.join("|"), KTYPE["N2"]))
    del top2, n2
    # C: compact name prefix (domains/handles/squashed names) + alias compact
    parts.append(_hk(df.filter(pl.col("n_compact").str.len_chars() >= 6),
                     pl.col("n_compact").str.slice(0, 10), KTYPE["C"]))
    parts.append(_hk(df.filter(pl.col("n_alias").str.len_chars() >= 6),
                     pl.col("n_alias").str.replace_all(" ", "").str.slice(0, 10), KTYPE["C"]))
    return pl.concat(parts).unique()


def build_keys(tbl: pa.Table, freq: pl.DataFrame, slice_rows: int) -> pl.DataFrame:
    return pl.concat([make_keys(d, freq) for d in _slices(tbl, slice_rows)])


def cap_keys(pool_keys: pl.DataFrame) -> pl.DataFrame:
    """Drop keys whose block holds more pool records than CAPS[kt] (per key type, bounded memory)."""
    out = []
    for name, kt in KTYPE.items():
        k = pool_keys.filter(pl.col("kt") == kt)
        ok = k.group_by("key").len().filter(pl.col("len") <= CAPS[name]).select("key")
        out.append(k.join(ok, on="key", how="semi"))
    return pl.concat(out)


def candidates_for(s1_keys: pl.DataFrame, pool_keys_capped: pl.DataFrame) -> pl.DataFrame:
    """Join S1 keys to (capped) pool keys -> unique (i, j) with a bitmask of key types that hit."""
    c = s1_keys.join(pool_keys_capped, on=["key", "kt"], how="inner", suffix="_p")
    return (c.group_by(["idx", "idx_p"]).agg(pl.col("kt").unique().sum().alias("kmask"))
             .rename({"idx": "i", "idx_p": "j"}).sort(["i", "j"]))          # sorted -> deterministic


# ============================================================================ features
_HOUSE_CODES = ["equal", "suffix", "prefix", "zeros", "edit1", "near", "diff", "miss1", "miss2"]


def house_rel(a: str, b: str) -> int:
    if not a and not b:
        return 8
    if not a or not b:
        return 7
    if a == b:
        return 0
    if a.lstrip("0") == b.lstrip("0"):
        return 3
    if a.endswith(b) or b.endswith(a):
        return 1
    if a.startswith(b) or b.startswith(a):
        return 2
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return 4
    da, db = "".join(ch for ch in a if ch.isdigit()), "".join(ch for ch in b if ch.isdigit())
    if da and db and len(da) < 9 and len(db) < 9 and abs(int(da) - int(db)) <= 10:
        return 5
    return 6


def _legal_rel(a: str, b: str) -> int:
    if not a and not b:
        return 0
    if a == b:
        return 1
    if not a or not b:
        return 2
    return 3 if not (set(a.split()) & set(b.split())) else 4


def _tok_diff(a: str, b: str) -> tuple[int, int, int]:
    ta, tb = set(a.split()), set(b.split())
    return len(tb - ta), len(ta - tb), len(ta & tb)


def _num_jacc(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return -1.0
    return len(sa & sb) / len(sa | sb)


def featurize(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame) -> pd.DataFrame:
    """pairs[i, j, kmask] -> feature frame (plus s1_id, cand_id)."""
    A = pairs.join(s1, left_on="i", right_on="idx").join(pool, left_on="j", right_on="idx", suffix="_c")
    need = [c for c in COLS if c != "country"]
    g = {c: A[c].fill_null("").to_numpy() for c in need + [x + "_c" for x in need] if A[c].dtype == pl.String}
    W = dict(workers=-1, dtype=np.float32)
    f = pd.DataFrame({"s1_id": g["entity_id"], "cand_id": g["entity_id_c"]})
    f["kmask"] = A["kmask"].to_numpy()
    for k, sc in {"tset": fuzz.token_set_ratio, "tsort": fuzz.token_sort_ratio, "ratio": fuzz.ratio,
                  "partial": fuzz.partial_ratio}.items():
        f[f"name_{k}"] = process.cpdist(g["n_core"], g["n_core_c"], scorer=sc, **W)
    f["comp_jw"] = process.cpdist(g["n_compact"], g["n_compact_c"], scorer=JaroWinkler.normalized_similarity, **W)
    f["comp_partial"] = process.cpdist(g["n_compact"], g["n_compact_c"], scorer=fuzz.partial_ratio, **W)
    f["alias_tset"] = process.cpdist(g["n_core"], g["n_alias_c"], scorer=fuzz.token_set_ratio, **W)
    f["addr_tset"] = process.cpdist(g["a_full"], g["a_full_c"], scorer=fuzz.token_set_ratio, **W)
    f["addr_ratio"] = process.cpdist(g["a_full"], g["a_full_c"], scorer=fuzz.ratio, **W)
    f["street_tset"] = process.cpdist(g["a_street"], g["a_street_c"], scorer=fuzz.token_set_ratio, **W)
    f["house_rel"] = np.fromiter((house_rel(a, b) for a, b in zip(g["a_house"], g["a_house_c"])), np.int8, len(f))
    f["num_jacc"] = np.fromiter((_num_jacc(a, b) for a, b in zip(g["a_numbers"], g["a_numbers_c"])), np.float32, len(f))
    f["legal_rel"] = np.fromiter((_legal_rel(a, b) for a, b in zip(g["n_legal"], g["n_legal_c"])), np.int8, len(f))
    d = np.array([_tok_diff(a, b) for a, b in zip(g["n_core"], g["n_core_c"])], dtype=np.int16).reshape(-1, 3)
    f["extra_tok"], f["missing_tok"], f["common_tok"] = d[:, 0], d[:, 1], d[:, 2]
    f["state_rel"] = np.where((g["a_state"] == "") | (g["a_state_c"] == ""), 2,
                              (g["a_state"] == g["a_state_c"]).astype(np.int8)).astype(np.int8)
    f["cand_addr_empty"] = A["a_empty_c"].to_numpy().astype(np.int8)
    f["cand_kind"] = pd.Series(g["n_kind_c"]).map({"name": 0, "domain": 1, "handle": 2, "empty": 3}).fillna(0).astype(np.int8).to_numpy()
    f["cand_native"] = (g["n_script_c"] != "latin").astype(np.int8)
    f["cand_src"] = np.char.startswith(g["entity_id_c"].astype(str), "S3").astype(np.int8)
    f["len_core_s1"] = np.fromiter((len(x.split()) for x in g["n_core"]), np.int16, len(f))
    f["len_core_c"] = np.fromiter((len(x.split()) for x in g["n_core_c"]), np.int16, len(f))
    # context: ranks / competition (within S1 and within record)
    f["name_rank_in_s1"] = f.groupby("s1_id")["name_tset"].rank(ascending=False, method="min").astype(np.float32)
    f["addr_rank_in_s1"] = f.groupby("s1_id")["addr_tset"].rank(ascending=False, method="min").astype(np.float32)
    f["name_gap_s1"] = f.groupby("s1_id")["name_tset"].transform("max") - f["name_tset"]
    f["n_cand_s1"] = f.groupby("s1_id")["cand_id"].transform("size").astype(np.int32)
    # NOTE: record-side competition features (how many / which S1s want this record) are NOT
    # computed here: within a chunk only a subset of S1s is visible, so they would differ between
    # val (few queries) and test (all S1). Compute them over ALL S1 of the country (stage-2, EXP-010).
    return f


FEATS = None  # set after first featurize


def feature_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in ("s1_id", "cand_id", "label")]


# ============================================================================ driver
def run_split(split: str, query_ids: set | None, chunk: int, fn, log=print):
    """For each country: build keys once, then process S1 queries in chunks; call fn(pairs_features)."""
    total_pairs, n_pool_total = 0, 0
    for ctry in countries_of(split):
        t0 = time.time()
        q = pl.scan_parquet(_norm_file(split, 1)).filter(pl.col("country") == ctry)
        if query_ids is not None:
            q = q.filter(pl.col("entity_id").is_in(sorted(query_ids)))
        s1 = q.select(COLS).collect()
        if s1.height == 0:
            continue
        ptbl = pool_table(split, ctry)                 # mmap; key columns are read slice by slice
        s1 = s1.with_row_index("idx")
        n_pool_c = ptbl.num_rows
        n_pool_total += n_pool_c
        freq = token_freq(ptbl, SLICE)
        pool_keys = cap_keys(build_keys(ptbl, freq, SLICE))
        gc.collect()
        s1_keys = build_keys(s1.drop("idx").to_arrow(), freq, SLICE)
        log(f"  [{ctry}] s1={s1.height:,} pool={n_pool_c:,} keys ready {time.time() - t0:.0f}s")
        for start in range(0, s1.height, chunk):
            sk = s1_keys.filter((pl.col("idx") >= start) & (pl.col("idx") < start + chunk))
            pairs = candidates_for(sk, pool_keys)
            if pairs.height == 0:
                continue
            pool_part = fetch_pool(ptbl, pairs["j"].to_numpy())
            f = featurize(pairs, s1, pool_part)
            del pool_part
            total_pairs += len(f)
            fn(f)
            del f, pairs
            gc.collect()
        log(f"  [{ctry}] done {time.time() - t0:.0f}s, pairs so far {total_pairs:,}")
        del s1, pool_keys, s1_keys, ptbl
        gc.collect()
    return total_pairs, n_pool_total


def collect(split, ids, chunk, model=None, log=print):
    """Run blocking+features for query ids. Without a model: return full feature frames (training).
    With a model: predict per chunk and keep only (s1_id, cand_id, prob) to save memory."""
    parts = []

    def fn(f):
        if model is None:
            parts.append(f)
        else:
            parts.append(f[["s1_id", "cand_id"]].assign(prob=model.predict(f[FEATS], num_threads=0).astype(np.float32)))
    _, n_pool = run_split(split, ids, chunk, fn, log)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), n_pool


def main(a):
    global FEATS
    t0 = time.time()
    if not a.test:
        folds = io.load_folds()
        rng = np.random.default_rng(SEED)
        tr_pool = np.sort(folds[folds.fold >= 2].s1_id.to_numpy())
        es_pool = np.sort(folds[folds.fold == 1].s1_id.to_numpy())
        tr_ids = set(rng.choice(tr_pool, size=min(a.n_train, len(tr_pool)), replace=False))
        es_ids = set(rng.choice(es_pool, size=min(a.n_train // 5, len(es_pool)), replace=False))
        del folds, tr_pool, es_pool
        gc.collect()
        ctx = harness.EvalContext.load(a.subset)
        pairs = io.load_gt_pairs(tr_ids | es_ids)          # only the training entities (memory)
        pos = set(zip(pairs.s1_id, pairs.match_id))
        del pairs
        with Run("baseline-v0-keys-lgbm", hypothesis="key blocking + rapidfuzz features + LightGBM baseline",
                 params={"subset": a.subset, "n_train": a.n_train, "caps": CAPS, "norm_v": NORM_V,
                         "chunk": a.chunk}, tags=["baseline", "blocking", "model"]) as run:
            print("[train pairs]")
            tr, _ = collect("train", tr_ids | es_ids, a.chunk)
            tr["label"] = [(s, c) in pos for s, c in zip(tr.s1_id, tr.cand_id)]
            FEATS = feature_cols(tr)
            is_es = tr.s1_id.isin(es_ids).to_numpy()
            run.log(train_pairs=int((~is_es).sum()), train_pos_rate=float(tr.label[~is_es].mean()))
            dtr = lgb.Dataset(tr.loc[~is_es, FEATS], tr.loc[~is_es, "label"].astype(int),
                              categorical_feature=["house_rel", "legal_rel", "state_rel", "cand_kind"])
            des = lgb.Dataset(tr.loc[is_es, FEATS], tr.loc[is_es, "label"].astype(int), reference=dtr)
            params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                          verbose=-1, seed=SEED, num_threads=0, deterministic=True, force_row_wise=True)
            model = lgb.train(params, dtr, 3000, valid_sets=[des],
                              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
            model.save_model(str(run.art_dir / "model.lgb"))
            (run.art_dir / "features.json").write_text(json.dumps(FEATS))
            imp = dict(sorted(zip(FEATS, model.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
            run.log(best_iter=model.best_iteration, feature_gain=imp)
            del tr, dtr, des
            gc.collect()
            print("[eval pairs]")
            ev, n_pool = collect("train", ctx.ids, a.chunk, model=model)
            harness.log_blocking(run, ev[["s1_id", "cand_id"]], ctx, n_pool)
            out = harness.log_predictions(run, ev, ctx)
            (run.art_dir / "decision.json").write_text(json.dumps({"threshold": out["val"]["threshold"]}))
            run.log(timing_min=round((time.time() - t0) / 60, 1))
            run.note(f"mini F0.5={out['val']['f05_macro']:.4f}. Top features: {list(imp)[:8]}")
    else:
        art = paths.ART_DIR / a.run_id
        model = lgb.Booster(model_file=str(art / "model.lgb"))
        FEATS = json.loads((art / "features.json").read_text())
        t = a.threshold if a.threshold is not None else json.loads((art / "decision.json").read_text())["threshold"]
        pred_parts, n_cand = [], [0]
        cand_schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string())])
        cw = pq.ParquetWriter(str(art / "test_candidates.parquet"), cand_schema, compression="zstd")

        def fn(f):
            # one row group per S1 chunk (complete S1 groups) -> streamable to candidate_pairs.tsv
            ct = pa.Table.from_pandas(f[["s1_id", "cand_id"]], schema=cand_schema, preserve_index=False)
            cw.write_table(ct, row_group_size=max(1, ct.num_rows))
            n_cand[0] += ct.num_rows
            p = model.predict(f[FEATS], num_threads=0).astype(np.float32)
            keep = p >= min(0.05, t)
            pred_parts.append(f.loc[keep, ["s1_id", "cand_id"]].assign(prob=p[keep]))
        run_split("test", None, a.chunk, fn)
        cw.close()
        pred = pd.concat(pred_parts, ignore_index=True)
        pred.to_parquet(art / "test_pred.parquet")
        m = threshold_matches(assign_best_s1(pred), t)
        pd.DataFrame([(s, x) for s, xs in m.items() for x in xs], columns=["s1_id", "match_id"]) \
          .to_parquet(art / "test_matches.parquet")
        print(f"test done in {(time.time() - t0) / 60:.1f} min: {n_cand[0]:,} candidate pairs, "
              f"{sum(map(len, m.values())):,} matches for {len(m):,} S1 (threshold {t})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="mini", choices=["micro", "mini", "fold0"])
    ap.add_argument("--n-train", type=int, default=20_000, help="train S1 queries (laptop ~20k; SageMaker 200k+)")
    ap.add_argument("--chunk", type=int, default=20_000, help="S1 queries per chunk (memory knob)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    main(ap.parse_args())
