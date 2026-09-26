"""Verify Lane G cross-fit logits are out-of-fold on train (lead 04:50): model h{k} trained on md5(s1_id) % 2 == k, so
logits_h{k}_train must hold only train-role pairs with md5 half 1-k; es (fold-1) rows may appear in both.
Exit code 1 on any violation.     python pipelines/xenc_verify_halves.py <dir>[+<dir>...]"""
import _bootstrap  # noqa: F401

import hashlib
import sys

import polars as pl

from ber import paths

dirs = sys.argv[1].split("+")
M = pl.read_parquet(paths.DATA_DIR / "kaggle" / "xenc2_map" / "train.parquet").join(
    pl.read_parquet(paths.DATA_DIR / "cands" / "v1_n2" / "train.parquet", columns=["s1_id", "cand_id", "role"]),
    on=["s1_id", "cand_id"], how="left")
h = M.select("s1_id").unique().with_columns(pl.col("s1_id").map_elements(
    lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 2, return_dtype=pl.Int8).alias("md5h"))
M = M.join(h, on="s1_id")
bad, seen = 0, None
for d in dirs:
    for f in sorted((paths.ROOT / d).glob("logits_h*_train.parquet")):
        k = int(f.name.split("_")[1][1:])
        ids = pl.read_parquet(f, columns=["pair_id"])
        x = M.join(ids, on="pair_id", how="inner")
        ins = x.filter((pl.col("role") == "train") & (pl.col("md5h") == k)).height
        print(f"{f.relative_to(paths.ROOT)}: {x.height:,} pairs (roles {x.group_by('role').len().sort('role').rows()}), in-sample {ins}")
        bad += ins
        seen = ids if seen is None else pl.concat([seen, ids])
tr = M.filter(pl.col("role") == "train")
cnt = seen.group_by("pair_id").len("n_files")
multi = tr.join(cnt, on="pair_id", how="left").with_columns(pl.col("n_files").fill_null(0))
print("train-role pairs by #files:", multi.group_by("n_files").len().sort("n_files").rows())
sys.exit(1 if bad else 0)
