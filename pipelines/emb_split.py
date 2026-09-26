"""Lane B: split the Kaggle bge-m3 output (pair_cos.parquet: tag, s1_id, cand_id, emb_cos_*) into
data/cands/<cache>/emb_<tag>.parquet for ctx_features.join_emb (NaN = outside band / empty text)."""
import _bootstrap  # noqa: F401

import sys

import polars as pl

from ber import paths

src = sys.argv[1] if len(sys.argv) > 1 else str(paths.ART_DIR / "kaggle" / "amlc-emb-bgem3" / "pair_cos.parquet")
cache = sys.argv[2] if len(sys.argv) > 2 else "v1_n2"
d = pl.read_parquet(src)
for tag, g in d.group_by("tag"):
    tag = tag[0] if isinstance(tag, tuple) else tag
    out = paths.DATA_DIR / "cands" / cache / f"emb_{tag}.parquet"
    g.drop("tag").write_parquet(out, compression="zstd")
    print(tag, g.height, out, g.select(pl.col("^emb_cos_.*$").mean()).row(0))
