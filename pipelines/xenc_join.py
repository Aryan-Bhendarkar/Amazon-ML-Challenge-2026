"""Stack cross-encoder logits into the GBDT as a feature (lead 02:35): one or more xenc sources, each becomes feature
`xenc_<name>` (the source's OOF/averaged `xenc_logit`; NaN for pairs that were not scored = stage-1 p < 0.01).

    python pipelines/xenc_join.py minilm=artifacts/kaggle/amlc-xenc2-score [xlmr=<dir> ...] [--map data/kaggle/xenc2_map]

Each <dir> holds logits_score_<tag>.parquet (pair_id, ..., xenc_logit) for tag in {train, mini, fold0x, test}; pair ids are
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
            src = paths.ROOT / d / f"logits_score_{tag}.parquet"
            if not src.exists():
                print(f"{name}/{tag}: missing {src}, skipped")
                continue
            L = pl.read_parquet(src, columns=["pair_id", "xenc_logit"])
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
