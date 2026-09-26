"""Lane B (monitor note 12) export: uncertain-band pairs + unique record texts for frozen bge-m3 cosines on Kaggle.

Band = stage-1 p in [LO, HI]:
  mini / fold0x / test : the 0710 model's probs (val_pred, dm-val fold0x_pred, test_pred)
  train                : decision_v1 4-fold OOF probs (v1_n1 train pairs; nkey_num-only pairs have no OOF -> out of band)
Text = norm_v1 n_full / a_full (the same normalization for every split; no country token). Record key = "<split>|<id>".
Outputs data/kaggle/emb/{records,pairs}.parquet (zstd). Low CPU (POLARS_MAX_THREADS=2).
"""
import _bootstrap  # noqa: F401

import os

import polars as pl

from ber import ctx_features as cf
from ber import paths

LO, HI = 0.05, 0.98
R0710 = paths.ART_DIR / "20260926-0710_aryan-bhendarkar_feat-v1-v1-n2-g1-g2-g3-g4-g5-ctx3"
SRC = {"train": (paths.ART_DIR / "20260925-2235_aryan-bhendarkar_decision-v1-v1-n1" / "oof_train.parquet", "train"),
       "mini": (R0710 / "val_pred.parquet", "train"),
       "fold0x": (paths.ART_DIR / "20260926-1441_aryan-bhendarkar_dm-val-fold0x" / "fold0x_pred.parquet", "train"),
       "test": (R0710 / "test_pred.parquet", "test")}
OUT = paths.DATA_DIR / "kaggle" / "emb"


def texts(split: str) -> pl.DataFrame:
    cols = ["entity_id", "n_full", "a_full"]
    df = pl.concat([pl.read_parquet(cf._norm_file(split, s, 1), columns=cols) for s in (1, 2, 3)])
    return df.select((pl.lit(split + "|") + pl.col("entity_id")).alias("key"),
                     pl.col("n_full").fill_null("").alias("name"), pl.col("a_full").fill_null("").alias("addr"))


def main():
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = []
    for tag, (p, split) in SRC.items():
        d = (pl.scan_parquet(p).filter(pl.col("prob").is_between(LO, HI)).select("s1_id", "cand_id").collect()
             .with_columns(pl.lit(tag).alias("tag"), (pl.lit(split + "|") + pl.col("s1_id")).alias("k1"),
                           (pl.lit(split + "|") + pl.col("cand_id")).alias("k2")))
        print(tag, d.height, "band pairs", flush=True)
        pairs.append(d)
    P = pl.concat(pairs).unique(["tag", "s1_id", "cand_id"])
    keys = pl.concat([P.select(pl.col("k1").alias("key")), P.select(pl.col("k2").alias("key"))]).unique()
    rec = pl.concat([texts("train"), texts("test")]).join(keys, on="key", how="semi")
    rec = rec.with_columns(pl.when(pl.col("addr") != "").then(pl.col("name") + " | " + pl.col("addr"))
                           .otherwise(pl.col("name")).alias("full"))
    assert rec.height == keys.height, (rec.height, keys.height)
    P.write_parquet(OUT / "pairs.parquet", compression="zstd")
    rec.write_parquet(OUT / "records.parquet", compression="zstd")
    print(f"pairs {P.height:,} ({P.group_by('tag').len().rows()}), records {rec.height:,}; "
          f"sizes {[(f.name, round(f.stat().st_size / 1e6, 1)) for f in OUT.glob('*.parquet')]} MB")


if __name__ == "__main__":
    main()
