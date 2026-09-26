"""Stack cross-encoder logits into the GBDT as a feature (lead 02:35): one or more xenc sources, each becomes feature
`xenc_<name>` (the source's OOF/averaged `xenc_logit`; NaN for pairs that were not scored = stage-1 p < 0.01).

    python pipelines/xenc_join.py minilm=artifacts/kaggle/amlc-xenc2-score [xlmr=<dir> ...] [--map data/kaggle/xenc2_map]

Each <dir> holds logits_score_<tag>.parquet (pair_id, ..., xenc_logit) for tag in {train, mini, fold0x, test}, OR (Lane G
format) logits_h{k}_<tag>.parquet (pair_id, xenc_h{k}) from cross-fitted halves: the value is the mean of the halves present
(train: exactly one half = the OOF model, es rows both; val/test: both). Several dirs may be given for one name separated
by '+' (e.g. XLM-R h0 and h1 delivered in separate dirs): their files are pooled. Pair ids are
resolved with the box-local map (<map>/<tag>.parquet: pair_id, s1_id, cand_id). Writes
data/cands/<cache>/xenc_<name>_<tag>.parquet (s1_id, cand_id, xenc_<name>), read by ctx_features.join_emb (all consumers:
features_v1 --xenc, dm_val, predict_test_v1). Leakage: train logits are out-of-fold by construction (cross-fit halves; es
rows = mean of both models); checked: train-role AUC 0.9932 vs never-trained es 0.9934.
"""
import _bootstrap  # noqa: F401

import argparse

import polars as pl

from ber import paths

TAGS = ("train", "mini", "fold0x", "test")


def main(a):
    for spec in a.sources:
        name, d = spec.split("=", 1)
        col = f"xenc_{name}"
        for tag in TAGS:
            src = paths.ROOT / d.split("+")[0] / f"logits_score_{tag}.parquet"
            halves = sorted(f for dd in d.split("+") for f in (paths.ROOT / dd).glob(f"logits_h*_{tag}.parquet"))
            if src.exists():
                L = pl.read_parquet(src, columns=["pair_id", "xenc_logit"])
            elif halves:
                parts = [pl.read_parquet(f, columns=["pair_id", f.name.split("_")[1].replace("h", "xenc_h")])
                         .rename(lambda c: c if c == "pair_id" else "v") for f in halves]
                L = pl.concat(parts).group_by("pair_id").agg(pl.col("v").mean().alias("xenc_logit"),
                                                             pl.len().alias("n_halves"))
                print(f"{name}/{tag}: halves {[f.name for f in halves]} -> n_halves {L.group_by('n_halves').len().sort('n_halves').rows()}")
                L = L.drop("n_halves")
            else:
                print(f"{name}/{tag}: no logits in {d}, skipped")
                continue
            M = pl.read_parquet(paths.ROOT / a.map / f"{tag}.parquet")
            J = M.join(L, on="pair_id", how="inner").select("s1_id", "cand_id", pl.col("xenc_logit").cast(pl.Float32).alias(col))
            assert J.height == L.height, (name, tag, J.height, L.height)
            out = paths.DATA_DIR / "cands" / a.cache / f"{col}_{tag}.parquet"
            J.write_parquet(out, compression="zstd")
            print(f"{name}/{tag}: {J.height:,} pairs -> {out.name} (mean {J[col].mean():.3f}, NaN {J[col].is_nan().sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="name=dir (dir relative to the repo root)")
    ap.add_argument("--map", default="data/kaggle/xenc2_map")
    ap.add_argument("--cache", default="v1_n2")
    main(ap.parse_args())
