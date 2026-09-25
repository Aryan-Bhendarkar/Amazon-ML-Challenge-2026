"""Blocking v1: multi-retriever candidate generation (per country; queries = S1, pool = S2+S3).

Every retriever returns a polars frame (i: query row, j: pool row, s: float32 score or null) and
has a bit in RBITS. `union` ORs the bits and keeps the max score. Retrievers:
  keys_v0     baseline keys (house+street-token, 2 rarest name tokens, compact prefix), capped blocks
  tf_name     char-3gram TF-IDF on n_core, top-k                                   (Sparkly-style)
  tf_na       TF-IDF name char3 view (w .35) + address word view (w .65), top-k (main retriever)
  akey        exact sorted a_full token key                   (co-located / unrelated-name positives)
  hskey       house number (leading zeros stripped) + sorted significant street tokens
  tf_empty    tf_name top-k restricted to pool records with an EMPTY address (name-only records)
  skel        phonetic consonant skeleton of the first two core tokens (native-script transliterations)
  rev         reverse: each pool record -> top-k S1 over ALL S1 of the country (tf_name)
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from . import paths, tfidf
from .normalize import STREET_CANON

RBITS = {"keys_v0": 1, "tf_name": 2, "tf_na": 4, "akey": 8, "hskey": 16, "tf_empty": 32, "skel": 64, "rev": 128}

POOL_COLS = ["entity_id", "country", "n_core", "n_compact", "n_alias", "n_legal", "n_kind", "n_script",
             "a_full", "a_street", "a_numbers", "a_house", "a_state", "a_empty"]
STREET_STOP = sorted(set(STREET_CANON.values()) | {"road", "street", "near", "opp", "post", "office", "city",
                                                     "main", "cross", "floor", "shop", "unit", "pmb", "box", "rue"})
KEY_CAPS = {"A": 150, "N1": 200, "N2": 100, "C": 60}          # baseline_v0 caps (unchanged)
KTYPE = {"A": 1, "N1": 2, "N2": 4, "C": 8}
KEY_COLS = ["entity_id", "n_core", "n_compact", "n_alias", "a_house", "a_street"]
SLICE = 400_000


# ============================================================================ data
def norm_file(split: str, source: int, norm_v: int = 0):
    return paths.FEATURE_DIR / f"norm_v{norm_v}_{split}_s{source}.parquet"


def pool_table(split: str, country: str, norm_v: int = 0) -> pa.Table:
    """S2+S3 of one country as a memory-mapped Arrow IPC table (cached). Row number = pool idx.
    Same file + row order as baseline_v0.pool_table."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    p = paths.FEATURE_DIR / f"pool_v{norm_v}_{split}_{safe}.arrow"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        writer = None
        with pa.OSFile(str(tmp), "wb") as sink:
            for s in (2, 3):
                for b in pq.ParquetFile(norm_file(split, s, norm_v)).iter_batches(batch_size=200_000, columns=POOL_COLS):
                    t = pa.Table.from_batches([b])
                    t = t.filter(pc.equal(t["country"], country))
                    if writer is None:
                        writer = pa.ipc.new_file(sink, t.schema)
                    writer.write_table(t)
            if writer is not None:
                writer.close()
        os.replace(tmp, p)
    return pa.ipc.open_file(pa.memory_map(str(p), "r")).read_all()


def countries_of(split: str, norm_v: int = 0) -> list[str]:
    return sorted(pl.read_parquet(norm_file(split, 1, norm_v), columns=["country"])["country"].unique().to_list())


# ============================================================================ baseline keys (verbatim from baseline_v0)
def _hk(df: pl.DataFrame, expr: pl.Expr, kt: int) -> pl.DataFrame:
    return df.select(pl.col("idx"), expr.hash(seed=7).alias("key"), pl.lit(kt, dtype=pl.Int8).alias("kt"))


def _core_tokens(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select("idx", pl.col("n_core").str.split(" ").alias("t")).explode("t", empty_as_null=True)
              .filter(pl.col("t").str.len_chars() >= 3).unique(["idx", "t"]))


def _slices(tbl: pa.Table, slice_rows: int = SLICE):
    for o in range(0, tbl.num_rows, slice_rows):
        part = pl.from_arrow(tbl.slice(o, slice_rows).select(KEY_COLS[1:]))
        yield part.with_row_index("idx", offset=o)


def token_freq(tbl: pa.Table, slice_rows: int = SLICE) -> pl.DataFrame:
    parts = [_core_tokens(d).group_by("t").len() for d in _slices(tbl, slice_rows)]
    return pl.concat(parts).group_by("t").agg(pl.col("len").sum())


def make_keys(df: pl.DataFrame, freq: pl.DataFrame) -> pl.DataFrame:
    parts = []
    a = (df.filter(pl.col("a_house") != "")
           .select("idx", "a_house", pl.col("a_street").str.split(" ").alias("t")).explode("t", empty_as_null=True)
           .filter((pl.col("t").str.len_chars() >= 4) & ~pl.col("t").is_in(STREET_STOP)))
    parts.append(_hk(a, pl.col("a_house") + "|" + pl.col("t"), KTYPE["A"]))
    tok = _core_tokens(df).join(freq, on="t", how="left").with_columns(pl.col("len").fill_null(0))
    top2 = tok.sort(["idx", "len", "t"]).group_by("idx", maintain_order=True).head(2)
    parts.append(_hk(top2.group_by("idx", maintain_order=True).first(), pl.col("t"), KTYPE["N1"]))
    n2 = top2.group_by("idx").agg(pl.col("t").sort()).filter(pl.col("t").list.len() == 2)
    parts.append(_hk(n2, pl.col("t").list.join("|"), KTYPE["N2"]))
    parts.append(_hk(df.filter(pl.col("n_compact").str.len_chars() >= 6),
                     pl.col("n_compact").str.slice(0, 10), KTYPE["C"]))
    parts.append(_hk(df.filter(pl.col("n_alias").str.len_chars() >= 6),
                     pl.col("n_alias").str.replace_all(" ", "").str.slice(0, 10), KTYPE["C"]))
    return pl.concat(parts).unique()


def build_keys(tbl: pa.Table, freq: pl.DataFrame, slice_rows: int = SLICE) -> pl.DataFrame:
    return pl.concat([make_keys(d, freq) for d in _slices(tbl, slice_rows)])


def cap_keys(pool_keys: pl.DataFrame) -> pl.DataFrame:
    out = []
    for name, kt in KTYPE.items():
        k = pool_keys.filter(pl.col("kt") == kt)
        ok = k.group_by("key").len().filter(pl.col("len") <= KEY_CAPS[name]).select("key")
        out.append(k.join(ok, on="key", how="semi"))
    return pl.concat(out)


def keys_v0(q_tbl: pa.Table, p_tbl: pa.Table) -> pl.DataFrame:
    freq = token_freq(p_tbl)
    pk = cap_keys(build_keys(p_tbl, freq))
    qk = build_keys(q_tbl, freq)
    c = qk.join(pk, on=["key", "kt"], how="inner", suffix="_p")
    return (c.group_by(["idx", "idx_p"]).agg(pl.col("kt").unique().sum().cast(pl.Int8).alias("kmask"))
             .rename({"idx": "i", "idx_p": "j"}).with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)))


# ============================================================================ exact keys
def sorted_tokens(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.sort().list.join(" "))


def akey_expr() -> pl.Expr:
    return sorted_tokens("a_full")


def hskey_expr() -> pl.Expr:
    """house (leading zeros stripped) | sorted significant street tokens ('' if either side is missing)."""
    street = (pl.col("a_street").fill_null("").str.split(" ")
              .list.eval(pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().is_in(STREET_STOP)))
              .list.unique().list.sort().list.join(" "))
    house = pl.col("a_house").fill_null("").str.strip_chars_start("0")
    return pl.when((house != "") & (street != "")).then(house + "|" + street).otherwise(pl.lit(""))


# phonetic consonant skeleton (strategy_v2 P1.5), applied per token of the anyascii n_core
_SKEL_SUBS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("sh", "s"),
              ("ck", "k"), ("x", "ks"), ("q", "k"), ("c", "k"), ("w", "v"), ("z", "j")]


def skeleton_expr(col: str = "n_core", n_tokens: int = 2) -> pl.Expr:
    e = pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[^a-z ]", "")
    for a, b in _SKEL_SUBS:
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"[aeiouy]", "")
    toks = e.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.head(n_tokens)
    return toks.list.join("|")


def _dedup_letters(s: pl.Series) -> pl.Series:
    """collapse repeated letters (polars regex has no backreferences)."""
    out = s
    for ch in "bdfgjklmnprstv":
        out = out.str.replace_all(ch + "+", ch)
    return out


def skeleton(series: pl.Series, n_tokens: int = 2) -> pl.Series:
    s = pl.DataFrame({"n_core": series}).select(skeleton_expr("n_core", n_tokens)).to_series()
    return _dedup_letters(s)


def key_join(qk: pl.Series, pk: pl.Series, cap: int, min_len: int = 1) -> pl.DataFrame:
    """Exact-key join. Pool blocks with more than `cap` records are dropped."""
    q = pl.DataFrame({"key": qk}).with_row_index("i").filter(pl.col("key").str.len_chars() >= min_len)
    p = pl.DataFrame({"key": pk}).with_row_index("j").filter(pl.col("key").str.len_chars() >= min_len)
    ok = p.group_by("key").len().filter(pl.col("len") <= cap).select("key")
    p = p.join(ok, on="key", how="semi")
    return q.join(p, on="key").select("i", "j")


# ============================================================================ TF-IDF
class TfViews:
    """TF-IDF matrices for S1 (all of the country) and pool; idf fit on S1_all ∪ pool (unlabeled).
    name = char_wb 3-grams of n_core; addr = whitespace tokens of a_full. Columns with df > max_df are
    dropped (bounds the top-k matmul; measured on micro: address-heavy word view is ~10x faster than
    char 3-grams on the address and has better recall)."""

    def __init__(self, s1_all: pl.DataFrame, pool: pl.DataFrame, max_df_name: float = 0.01,
                 max_df_addr: float = 0.02, n_jobs: int | None = None):
        cn_s = tfidf.raw_counts(s1_all["n_core"].fill_null("").to_list(), n_jobs)
        cn_p = tfidf.raw_counts(pool["n_core"].fill_null("").to_list(), n_jobs)
        idf = tfidf.fit_idf([cn_s, cn_p], max_df_name)
        self.Sn, self.Pn = tfidf.weight(cn_s, idf), tfidf.weight(cn_p, idf)
        del cn_s, cn_p
        ca_s = tfidf.raw_counts(s1_all["a_full"].fill_null("").to_list(), n_jobs, analyzer="word")
        ca_p = tfidf.raw_counts(pool["a_full"].fill_null("").to_list(), n_jobs, analyzer="word")
        idf = tfidf.fit_idf([ca_s, ca_p], max_df_addr)
        self.Sa, self.Pa = tfidf.weight(ca_s, idf), tfidf.weight(ca_p, idf)
        del ca_s, ca_p

    def na(self, which: str, rows=None, w=(0.35, 0.65)) -> sp.csr_matrix:
        n, a = (self.Sn, self.Sa) if which == "s1" else (self.Pn, self.Pa)
        if rows is not None:
            n, a = n[rows], a[rows]
        return tfidf.combine([n, a], list(w))


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, i: np.ndarray, j: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """cosine of A[i[k]] . B[j[k]] for each pair (l2 rows), chunked."""
    out = np.empty(len(i), dtype=np.float32)
    for o in range(0, len(i), chunk):
        a, b = A[i[o:o + chunk]], B[j[o:o + chunk]]
        out[o:o + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def topk_frame(Q, P, k, n_threads=None, q_map=None, p_map=None) -> pl.DataFrame:
    i, j, s = tfidf.topk(Q, P, k=k, n_threads=n_threads)
    if q_map is not None:
        i = q_map[i]
    if p_map is not None:
        j = p_map[j]
    return pl.DataFrame({"i": i.astype(np.uint32), "j": j.astype(np.uint32), "s": s})


def union(parts: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """parts: retriever name -> (i, j[, s]). Returns i, j, rbits (int32)."""
    fr = [df.select("i", "j").with_columns(pl.lit(RBITS[name], dtype=pl.Int32).alias("b")) for name, df in parts.items()]
    return (pl.concat(fr).group_by("i", "j").agg(pl.col("b").unique().sum().alias("rbits")))


def cap_per_query(c: pl.DataFrame, score: str, cap: int) -> pl.DataFrame:
    return (c.sort(["i", score], descending=[False, True])
             .with_columns(pl.int_range(pl.len()).over("i").alias("_r"))
             .filter(pl.col("_r") < cap).drop("_r"))
