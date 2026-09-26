"""C1 (lead decision 18:50): France rows re-featurized on norm v2, US/India rows untouched.

  filter : python pipelines/splice_fr.py filter            test_n2.parquet -> test_n2fr.parquet (France S1 only,
                                                           one row group per source row group: complete S1 lists)
  splice : python pipelines/splice_fr.py splice --run <run> --fr-suffix _n2fr
           C1 matches = C0 test_matches (non-France S1) + test_matches<fr-suffix> (France S1), written next to them as
           test_matches_c1.parquet, plus a label-free France before/after report (json).
Country comes from the data (test S1 table) and is only used to select rows, never as a feature.
"""
import _bootstrap  # noqa: F401

import argparse
import json

import polars as pl
import pyarrow.parquet as pq

from ber import ctx_features as cf
from ber import paths

CANDS = paths.DATA_DIR / "cands"


def fr_ids() -> set:
    s1 = pl.read_parquet(cf._norm_file("test", 1, 1), columns=["entity_id", "country"])
    return set(s1.filter(pl.col("country") == "France")["entity_id"].to_list())


def do_filter(a):
    fr = fr_ids()
    src = pq.ParquetFile(CANDS / a.cache / "test_n2.parquet")
    out = CANDS / a.cache / "test_n2fr.parquet"
    tmp = out.with_suffix(".tmp")
    w, n = None, 0
    for rg in range(src.num_row_groups):
        t = pl.from_arrow(src.read_row_group(rg)).filter(pl.col("s1_id").is_in(list(fr)))
        if t.height == 0:
            continue
        tb = t.to_arrow()
        if w is None:
            w = pq.ParquetWriter(str(tmp), tb.schema, compression="zstd")
        w.write_table(tb.cast(w.schema), row_group_size=tb.num_rows)
        n += t.height
    w.close()
    tmp.rename(out)
    print(f"France pairs {n:,} -> {out}")


def report(m: pl.DataFrame, pred: pl.DataFrame, fr: set, t: float) -> dict:
    mf = m.filter(pl.col("s1_id").is_in(list(fr)))
    k = mf.group_by("s1_id").len("k")
    strong = pred.filter(pl.col("s1_id").is_in(list(fr)) & (pl.col("prob") >= 0.99))["s1_id"].n_unique()
    return {"fr_matches": mf.height, "fr_matches_per_s1": round(mf.height / len(fr), 4),
            "fr_empty_rate": round(1 - k.height / len(fr), 4),
            "fr_s1_with_candidate_p>=0.99": round(strong / len(fr), 4),
            "fr_band_pairs_per_s1": round(pred.filter(pl.col("s1_id").is_in(list(fr)) &
                                                      pl.col("prob").is_between(0.2, 0.975)).height / len(fr), 4)}


def do_splice(a):
    art = paths.ART_DIR / a.run
    fr = fr_ids()
    m0 = pl.read_parquet(art / "test_matches.parquet")
    mf = pl.read_parquet(art / f"test_matches{a.fr_suffix}.parquet")
    extra = set(mf["s1_id"].unique().to_list()) - fr
    assert not extra, f"{len(extra)} non-France S1 in the France matches"
    m1 = pl.concat([m0.filter(~pl.col("s1_id").is_in(list(fr))), mf.select(m0.columns)])
    assert m1["match_id"].is_unique().all(), "a record assigned twice"
    m1.write_parquet(art / "test_matches_c1.parquet")
    p0 = pl.read_parquet(art / "test_pred.parquet")
    p1 = pl.read_parquet(art / f"test_pred{a.fr_suffix}.parquet")
    rep = {"C0_norm_v1": report(m0, p0, fr, a.t), "C1_norm_v2": report(m1, p1, fr, a.t),
           "non_fr_rows_identical": m0.filter(~pl.col("s1_id").is_in(list(fr))).sort(["s1_id", "match_id"]).equals(
               m1.filter(~pl.col("s1_id").is_in(list(fr))).sort(["s1_id", "match_id"]))}
    (art / "c1_fr_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["filter", "splice"])
    ap.add_argument("--cache", default="v1_n2")
    ap.add_argument("--run", default="20260926-1737_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3-psemb")
    ap.add_argument("--fr-suffix", default="_n2fr")
    ap.add_argument("--t", type=float, default=0.775)
    a = ap.parse_args()
    do_filter(a) if a.cmd == "filter" else do_splice(a)
