"""Lane G / G3 prep: scoring inputs for kaggle/xenc_g3.py from box1's G3 pair set (data/kaggle/g3/pairs.parquet).

- half: the cross-fit half the G2 models were trained on = md5(s1_id) % 2 (pipelines/xenc_g1_export.py), NOT the
  g3 `cf` column (polars hash(42), the halves of the OLD MiniLM ckpts). Train-role pairs get their half; fold-1
  ('es', cf == -1) and mini / fold0x / test pairs get -1 (scored by every model, averaged downstream).
  Checked: md5 half == G1 export cf for every S1 that appears in both.
- records: key -> text in the G2 training serialization on norm_v2 (train == norm_v1; test France uses the
  shipped France rules): "[COL] name [VAL] <n_full> [COL] address [VAL] <a_full> [NUM] <house>".
Upload dir: data/kaggle/g3_up/{pairs,records}.parquet (no entity ids: keys are replaced by int ids).
Local map : data/kaggle/g3_up_map.parquet (pair_id, tag, k1, k2).

    python pipelines/xenc_g3_prep.py
"""
import _bootstrap  # noqa: F401

import hashlib
import time

import numpy as np
import polars as pl

from ber import paths

G3 = paths.DATA_DIR / "kaggle" / "g3"
UP = paths.DATA_DIR / "kaggle" / "g3_up"


def md5_half(ids: list[str]) -> np.ndarray:
    return np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 for s in ids], dtype=np.int8)


def main():
    t0 = time.time()
    UP.mkdir(parents=True, exist_ok=True)
    P = pl.read_parquet(G3 / "pairs.parquet")
    P = P.with_columns(pl.col("k1").str.split("|").list.get(1).alias("s1_id"))
    tr = P.filter((pl.col("tag") == "train") & (pl.col("cf") >= 0))
    h = md5_half(tr["s1_id"].to_list())
    P = P.join(tr.select("pair_id").with_columns(pl.Series("half", h)), on="pair_id", how="left") \
         .with_columns(pl.col("half").fill_null(-1).cast(pl.Int8))
    # consistency with the G1 export halves (the halves the G2 models were actually trained on)
    g1 = (pl.read_parquet(paths.DATA_DIR / "kaggle" / "g1_map" / "train.parquet", columns=["pair_id", "s1_id"])
            .join(pl.read_parquet(paths.DATA_DIR / "kaggle" / "g1" / "train.parquet", columns=["pair_id", "cf"]),
                  on="pair_id").select("s1_id", "cf").unique())
    chk = P.filter(pl.col("half") >= 0).select("s1_id", "half").unique().join(g1, on="s1_id")
    bad = chk.filter(pl.col("half") != pl.col("cf")).height
    print(f"[half] {chk.height:,} S1 shared with G1, mismatches {bad}", flush=True)
    assert bad == 0 and chk.height > 0
    keys = pl.concat([P.select(pl.col("k1").alias("key")), P.select(pl.col("k2").alias("key"))]).unique()
    recs = []
    for split in ("train", "test"):
        d = pl.concat([pl.read_parquet(paths.FEATURE_DIR / f"norm_v2_{split}_s{s}.parquet",
                                       columns=["entity_id", "n_full", "a_full", "a_house"]) for s in (1, 2, 3)])
        recs.append(d.select((pl.lit(split + "|") + pl.col("entity_id")).alias("key"), pl.concat_str([
            pl.lit("[COL] name [VAL] "), pl.col("n_full").fill_null(""),
            pl.lit(" [COL] address [VAL] "), pl.col("a_full").fill_null(""),
            pl.when(pl.col("a_house").fill_null("") != "").then(pl.lit(" [NUM] ") + pl.col("a_house"))
              .otherwise(pl.lit(""))]).alias("text")))
    R = pl.concat(recs).join(keys, on="key", how="semi").with_row_index("rid")
    assert R.height == keys.height, f"missing records {keys.height - R.height}"
    rid = R.select("key", "rid")
    out = (P.join(rid.rename({"key": "k1", "rid": "r1"}), on="k1").join(rid.rename({"key": "k2", "rid": "r2"}), on="k2")
             .select("pair_id", "tag", "half", "r1", "r2").sort("pair_id"))
    assert out.height == P.height
    out.write_parquet(UP / "pairs.parquet", compression="zstd")
    R.select("rid", "text").write_parquet(UP / "records.parquet", compression="zstd")
    P.select("pair_id", "tag", "k1", "k2").write_parquet(paths.DATA_DIR / "kaggle" / "g3_up_map.parquet")
    print(out.group_by("tag", "half").len().sort("tag", "half"))
    print(f"records {R.height:,}; sizes MB", {f.name: round(f.stat().st_size / 1e6, 1) for f in UP.glob("*.parquet")},
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
